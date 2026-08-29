# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import copy
import codecs
import heapq
import ipaddress
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Tuple, AsyncIterator
from collections import OrderedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import requests

import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

@dataclass
class RouterMethod:
    LEAST_LOAD: str = "least_load"
    SESSION_AFFINITY: str = "session_affinity"

@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"

@dataclass
class RequestRecoveryState:
    request_id: str
    api: str
    original_request: dict[str, Any]
    # 当前准备发送给 P/D 节点的请求。
    current_request: dict[str, Any]
    # 请求目前所处阶段。
    # created、prefill、decode、decode_generating、waiting_prefill、waiting_decode、completed、cancelled、failed
    phase: str = "created"
    # 用户是否开启模型思考。
    # 只有明确传入 chat_template_kwargs.enable_thinking=false 才关闭；
    # 未传或传入 true 时统一按开启处理。
    thinking_enabled: bool = True
    # 是否已经从后端响应中检测到 reasoning/reasoning_content 字段。
    # 检测到后，直接按照 reasoning parser 的结构累计和恢复。
    reasoning_parser_detected: bool = False
    # reasoning parser 模式下已经发送给客户端的思考内容。
    reasoning_text: str = ""
    # reasoning parser 模式下已经发送给客户端的正式回答内容。
    content_text: str = ""
    # 未开启 reasoning parser 时，后端 content 中累计的全部原始模型输出。
    # thinking 开启时，其中可能同时包含思考文本、</think> 和正式回答。
    raw_content_text: str = ""
    # 用于恢复续推的最终 assistant.content 前缀。
    generated_text: str = ""
    # 当前已经生成的 Token 数。
    completion_tokens: int = 0
    # 该请求已经恢复过多少次。
    recovery_count: int = 0
    # 当前这一轮发送给 P/D 的内部请求 ID。
    # Decode 恢复时会重新生成，但逻辑 request_id 不变。
    backend_request_id: str = ""

TAINT_PRIORITY = 1e15


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        
        # Selection locks to prevent race conditions in concurrent async requests
        # Split into two independent locks so prefiller and decoder paths can run in parallel
        self._prefiller_selection_lock = threading.Lock()
        self._decoder_selection_lock = threading.Lock()
        
        # 保存所有正在执行或等待恢复的请求。
        self.recovery_requests: dict[str, RequestRecoveryState] = {}
        # 防止多个线程或请求同时修改字典。
        self.recovery_requests_lock = threading.RLock()
        # 保存 FastAPI 当前运行的 asyncio 事件循环。
        self.event_loop = asyncio.get_running_loop()
        # Prefill 可用信号。
        self.prefiller_available_event = asyncio.Event()
        # Decode 可用信号。
        self.decoder_available_event = asyncio.Event()
        
        # Removed selection locks - no longer needed for synchronous methods
        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = [(0.0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        
        # Session affinity mapping for SESSION_AFFINITY strategy
        # Maps session_id -> instance_idx (using OrderedDict for LRU)
        self.session_prefill_map: OrderedDict = OrderedDict()
        self.session_decoder_map: OrderedDict = OrderedDict()
        self._session_lock = threading.Lock()
        self.SESSION_MAP_MAX_SIZE = 10000  # LRU capacity limit
        
        # 根据初始节点列表，设置 Prefill 和 Decode 的可用状态。
        self._sync_availability_events()
        # 所有属性初始化完成后，再启动节点监听线程。
        self.node_listener = NodeListener(self)

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        if server in self.tainted_prefillers:
            priority = TAINT_PRIORITY
        else:
            priority = (server.active_tokens + server.active_kv_cache * 0.3)
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        if server in self.tainted_decoders:
            priority = TAINT_PRIORITY
        else:
            priority = server.active_tokens
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())
        
    def register_recovery_request(self, state: RequestRecoveryState):
        with self.recovery_requests_lock:
            self.recovery_requests[state.request_id] = state
        logger.info("[REQUEST REGISTERED] request_id=%s phase=%s", state.request_id, state.phase)
        
    def get_recovery_request(self, request_id: str) -> RequestRecoveryState | None:
        with self.recovery_requests_lock:
            return self.recovery_requests.get(request_id)
    
    def remove_recovery_request(self, request_id: str) -> RequestRecoveryState | None:
        with self.recovery_requests_lock:
            state = self.recovery_requests.pop(request_id, None)
        if state is not None:
            logger.info("[REQUEST REMOVED] request_id=%s phase=%s", state.request_id, state.phase)
        return state
    
    def has_available_prefiller(self) -> bool:
        return any(priority < TAINT_PRIORITY for priority, _, _ in self.prefiller_heap)

    def has_available_decoder(self) -> bool:
        return any(priority < TAINT_PRIORITY for priority, _, _ in self.decoder_heap)
        
    async def wait_for_prefiller(self, timeout: float = 5.0) -> bool:
        """等待 Prefill 可用，超时返回 False。"""
        if self.has_available_prefiller():
            return True

        # 节点刚被隔离时，available_event 可能还保留之前的 set 状态。
        # 主动清理旧状态，避免 wait() 立即返回并连续刷出 keep-alive。
        self.prefiller_available_event.clear()

        # clear 与节点恢复可能并发发生，因此清理后重新检查一次，
        # 防止丢失刚刚发生的恢复通知。
        if self.has_available_prefiller():
            self.prefiller_available_event.set()
            return True

        try:
            await asyncio.wait_for(self.prefiller_available_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

        available = self.has_available_prefiller()
        if not available:
            # 防止旧事件或虚假唤醒导致下一轮继续立即返回。
            self.prefiller_available_event.clear()
        return available

    async def wait_for_decoder(self, timeout: float = 5.0) -> bool:
        """等待 Decode 可用，超时返回 False。"""
        if self.has_available_decoder():
            return True

        # 清理节点被隔离前遗留的 set 状态，避免恢复等待循环空转，
        # 从而保证 keep-alive 按 timeout 周期输出，而不是瞬间刷屏。
        self.decoder_available_event.clear()

        # 防止 clear 与节点恢复并发时丢失恢复通知。
        if self.has_available_decoder():
            self.decoder_available_event.set()
            return True

        try:
            await asyncio.wait_for(self.decoder_available_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

        available = self.has_available_decoder()
        if not available:
            self.decoder_available_event.clear()
        return available

    def _sync_availability_events(self) -> None:
        """同步 Prefill/Decode 可用事件，避免事件状态延迟造成空转。"""

        def apply_event_states() -> None:
            if self.has_available_prefiller():
                self.prefiller_available_event.set()
            else:
                self.prefiller_available_event.clear()

            if self.has_available_decoder():
                self.decoder_available_event.set()
            else:
                self.decoder_available_event.clear()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            # NodeListener 后台线程中没有正在运行的 asyncio 事件循环。
            running_loop = None

        if running_loop is self.event_loop:
            # 当前已经在 Proxy 主事件循环中，立即同步，避免 clear/set
            # 被延迟到后续循环才执行。
            apply_event_states()
        else:
            # NodeListener 等后台线程通过线程安全方式提交到主事件循环。
            self.event_loop.call_soon_threadsafe(apply_event_states)
    
    def list_recovery_requests(self) -> list[RequestRecoveryState]:
        with self.recovery_requests_lock:
            return list(self.recovery_requests.values())

    def select_prefiller(self, token_count):
        """Select the least loaded prefiller instance. Thread-safe for concurrent async calls."""
        with self._prefiller_selection_lock:
            if not self.has_available_prefiller():
                raise RuntimeError("No prefiller servers available")

            priority, chosen, server = heapq.heappop(self.prefiller_heap)

            # Update the chosen server
            self.prefillers[chosen].active_tokens += token_count
            self.prefillers[chosen].active_kv_cache += token_count

            # Update priority and re-add to heap
            self._update_prefiller_priority(chosen)

            return chosen

    def release_prefiller(self, idx, token_count):
        """Release a prefiller instance. Thread-safe for concurrent async calls."""
        with self._prefiller_selection_lock:
            if idx >= len(self.prefillers):
                return
            self.prefillers[idx].active_tokens -= token_count
            # Update priority queue after releasing
            self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        """Release prefiller KV cache. Thread-safe for concurrent async calls."""
        with self._prefiller_selection_lock:
            if idx >= len(self.prefillers):
                return
            if self.prefillers[idx].active_kv_cache > 0:
                self.prefillers[idx].active_kv_cache -= token_count
            # Update priority queue after releasing
            self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):
        """Select the least loaded decoder instance. Thread-safe for concurrent async calls."""
        with self._decoder_selection_lock:
            if not self.has_available_decoder():
                raise RuntimeError("No decoder servers available")

            priority, chosen, server = heapq.heappop(self.decoder_heap)

            # Update the chosen server
            self.decoders[chosen].active_tokens += token_count

            # Update priority and re-add to heap
            self._update_decoder_priority(chosen)

            return chosen

    def release_decoder(self, idx, token_count):
        """Release a decoder instance. Thread-safe for concurrent async calls."""
        with self._decoder_selection_lock:
            if idx >= len(self.decoders):
                return
            self.decoders[idx].active_tokens -= token_count
            # Update priority queue after releasing
            self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = self.node_listener.check_instance_status(server)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)
                self.prefiller_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.prefiller_heap
                ]
                heapq.heapify(self.prefiller_heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)
                # prefiller_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.prefiller_heap, (0, len(self.prefillers) - 1, server))
        self._sync_availability_events()
        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)
                self.decoder_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.decoder_heap
                ]
                heapq.heapify(self.decoder_heap)
            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.decoder_heap, (0, len(self.decoders) - 1, server))
        self._sync_availability_events()
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning("Start to taint prefill instances %s.", instances)
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)
        self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
        prefiller_heap_copy = self.prefiller_heap.copy()
        prefiller_heap_copy.sort(key=lambda x: x[1])  # sorted by key: prefiller_idx
        prefiller_heap = []
        idx = 0
        for priority, _, server in prefiller_heap_copy:
            if server not in instances_to_remove:
                prefiller_heap.append((priority, idx, server))
                idx += 1

        # prefiller_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.prefiller_heap = prefiller_heap
        heapq.heapify(self.prefiller_heap)
        self._sync_availability_events()
        self.print_status(f"Remove prefiller instances: {instances}.")
        return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning("Start to taint decode instances %s.", instances)
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x[1])  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for priority, _, server in decoder_heap_copy:
            if server not in instances_to_remove:
                decoder_heap.append((priority, idx, server))
                idx += 1

        # decoder_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        self._sync_availability_events()
        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        self.prefiller_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.prefiller_heap
        ]
        heapq.heapify(self.prefiller_heap)
        self._sync_availability_events()

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        self.decoder_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.decoder_heap
        ]
        heapq.heapify(self.decoder_heap)
        self._sync_availability_events()

    def mark_prefiller_unavailable(self, server: ServerState) -> None:
        """立即将一个 Prefill 标记为不可用，并加入恢复监听。"""
        self._taint_prefillers([server])
        self.node_listener.waiting_nodes[str(server)] = (InstanceType.PREFILL, server, 0)
        logger.warning("[PREFILL UNAVAILABLE] instance=%s has been moved to waiting nodes", server)

    def mark_decoder_unavailable(self, server: ServerState) -> None:
        """立即隔离故障 Decode，并加入恢复监听。"""
        self._taint_decoders([server])
        self.node_listener.waiting_nodes[str(server)] = (InstanceType.DECODE, server, 0)
        logger.warning("[DECODE UNAVAILABLE] instance=%s has been moved to waiting nodes", server)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.check_timeout = 10.0      # 将超时时间延长到 10 秒
        self.max_failures = 3          # 连续失败 3 次才判定为 DOWN
        self.failure_counters = {}     # 记录正常节点的连续失败次数: {server_url: count}

        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

        logger.info("NodeListener background thread started.")
    def _node_listener(self) -> None:
        while True:
            if not hasattr(self.proxy_state, 'prefillers') or not hasattr(self.proxy_state, 'decoders'):
                logger.debug("ProxyState is not fully initialized yet. Waiting...")
                time.sleep(1)
                continue

            if self.waiting_nodes:
                logger.debug(f"Checking {len(self.waiting_nodes)} nodes in waiting_nodes list.")
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = self.check_instance_status(server)
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    logger.info(f"[RECOVERY] {instance_type} instance {server.host}:{server.port} recovered. Adding back to proxy pool.")
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)
                    if (check_times + 1) % 10 == 0:  # 每失败 10 次打印一次警告，避免日志爆炸
                        logger.warning(f"[WAITING] {instance_type} instance {server.host}:{server.port} is still down. Checked {check_times} times.")
            # 2. 检查正常服务中的 Prefill 节点（发现异常则踢出）
            for server in list(self.proxy_state.prefillers):
                if server not in self.proxy_state.tainted_prefillers:
                    self._check_and_handle_active_node(server, InstanceType.PREFILL)

            # 3. 检查正常服务中的 Decode 节点（发现异常则踢出）
            for server in list(self.proxy_state.decoders):
                if server not in self.proxy_state.tainted_decoders:
                    self._check_and_handle_active_node(server, InstanceType.DECODE)

            if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                if not need_waiting:
                    self.proxy_state.tainted_prefillers.clear()

            if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                if not need_waiting:
                    self.proxy_state.tainted_decoders.clear()
            time.sleep(global_args.waiting_retry_interval)

    def _check_and_handle_active_node(self, server: Any, instance_type: str) -> None:
        """
        检查活跃节点，加入连续失败容错机制
        """
        server_key = f"{server.host}:{server.port}"
        is_valid = self.check_instance_status(server, self.check_timeout)
        
        if is_valid:
            # 如果健康，清零失败计数
            if server_key in self.failure_counters:
                self.failure_counters[server_key] = 0
        else:
            # 如果不健康，增加失败计数
            current_failures = self.failure_counters.get(server_key, 0) + 1
            self.failure_counters[server_key] = current_failures
            
            if current_failures >= self.max_failures:
                logger.error(f"[DOWN] {instance_type} instance {server_key} failed {current_failures} times. Removing from active pool.")
                if instance_type == InstanceType.PREFILL:
                    self.proxy_state.remove_prefillers([server])
                else:
                    self.proxy_state.remove_decoders([server])
                self.waiting_nodes[str(server)] = (instance_type, server, 0)
                # 移出活跃池后，清理计数器
                self.failure_counters.pop(server_key, None)
            else:
                logger.warning(f"[WARNING] {instance_type} instance {server_key} check failed ({current_failures}/{self.max_failures}).")

    @staticmethod
    def check_instance_status(server: Any, timeout: float = 10) -> bool:
        """
        请求 /metrics。
        如果包含 999999，表示故障 (返回 False)。
        如果连不上，表示故障 (返回 False)。
        正常响应且不包含 999999，表示健康 (返回 True)。
        """
        url = f"http://{server.host}:{server.port}/metrics"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            if "999999.0" in response.text:
                logger.warning(f"[HEALTH CHECK] {url} reported failure code '999999'.")
                return False
            else:
                return True
                
        except Exception as e:
            logger.warning(f"[HEALTH CHECK] Failed to connect to {url}. Error: {type(e).__name__} - {str(e)}")
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--router-method",
        type=str,
        choices=[RouterMethod.LEAST_LOAD, RouterMethod.SESSION_AFFINITY],
        default=RouterMethod.LEAST_LOAD,
        help="Router method for selecting backend servers",
    )
    parser.add_argument(
        "--custom-session-id-headers",
        type=str,
        nargs="+",
        default=[],
        help="Custom HTTP header names for session ID extraction (highest priority)",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if 400 <= status_code < 500:
                logger.error(
                    "Prefill request was rejected by %s with status %s: %s; "
                    "response_body=%s",
                    endpoint,
                    status_code,
                    e,
                    e.response.text[:4000],
                )
                raise
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, e)
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc
        except httpx.RequestError as e:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, e)
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    decoder: ServerState | None = None,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    # 整个函数生命周期内，是否已经向外层返回过 chunk。
    first_chunk_sent = False
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                # HTTP 状态码不是 2xx 时抛出异常。
                response.raise_for_status()

                # 不直接使用 ``async for``，而是周期性检查当前 Decode 是否
                # 已经被 NodeListener 隔离。这样既能处理 APIServer 被 kill 后的
                # EOF/连接重置，也能处理端口已关闭但旧 HTTP 流一直不结束的半关闭状态。
                byte_stream = response.aiter_bytes().__aiter__()
                while True:
                    next_chunk_task = asyncio.create_task(byte_stream.__anext__())
                    try:
                        while True:
                            done, _ = await asyncio.wait(
                                {next_chunk_task},
                                timeout=1.0,
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            if next_chunk_task in done:
                                try:
                                    chunk = next_chunk_task.result()
                                except StopAsyncIteration:
                                    # HTTP 流自然结束，交给外层根据 [DONE] 或
                                    # 正常 finish_reason 判断是否属于完整响应。
                                    return

                                first_chunk_sent = True
                                yield chunk
                                break

                            if (decoder is not None and proxy_state is not None and decoder in proxy_state.tainted_decoders):
                                raise RuntimeError(f"decode instance {decoder} was marked unavailable while its response stream was still open")
                    finally:
                        if not next_chunk_task.done():
                            next_chunk_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await next_chunk_task
        except asyncio.CancelledError:
            # 用户主动断开时，必须把取消异常继续向外抛。
            raise
        except Exception as error:
            if (isinstance(error, httpx.HTTPStatusError) and 400 <= error.response.status_code < 500):
                # 请求参数或模板被后端拒绝，不应把健康 Decode 标记为故障。
                raise
            # 已经向用户输出过内容后，绝对不能原样重试。
            if first_chunk_sent:
                logger.error(
                    "[DECODE STREAM INTERRUPTED] "
                    "request_id=%s attempt=%s decoder=%s error=%s",
                    request_id,
                    attempt,
                    decoder,
                    error,
                )
                raise DecodeStreamError(original_error=error, partial_response_sent=True) from error
            # 一个 chunk 都没返回，可以进行短暂重试。
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[DECODE RETRY] request_id=%s "
                    "attempt=%s/%s delay=%s decoder=%s error=%s",
                    request_id,
                    attempt,
                    max_retries,
                    delay,
                    decoder,
                    error,
                )
                await asyncio.sleep(delay)
                continue
            # 一个 chunk 都没返回，但重试次数已经耗尽。
            logger.error(
                "[DECODE REQUEST FAILED] "
                "request_id=%s attempts=%s decoder=%s error=%s",
                request_id,
                max_retries,
                decoder,
                error,
            )
            raise DecodeStreamError(original_error=error, partial_response_sent=False) from error
        
        
def extract_session_id(req_data: dict, req_header):
    """
    Extract hash key with priority:
    1. HTTP Headers (case-insensitive): custom headers (from --custom-session-id-headers) > x-session-id > x-user-id > x-tenant-id > x-correlation-id > x-request-id > x-trace-id
    2. Body: session_params.session_id > user > session_id > user_id
    3. Fallback: json.dumps(req_data, sort_keys=True)
    """
    # HTTP Headers priority (case-insensitive): custom headers + built-in headers
    custom_headers = getattr(global_args, 'custom_session_id_headers', []) or []
    header_priority = custom_headers + [
        "x-session-id", "x-user-id", "x-tenant-id",
        "x-correlation-id", "x-request-id", "x-trace-id"
    ]
    if req_header:
        headers_lower = {k.lower(): v for k, v in req_header.items()}
        for hdr in header_priority:
            if hdr in headers_lower and headers_lower.get(hdr):
                return f"header:{hdr}:{headers_lower.get(hdr)}"

    # Body fields priority
    if req_data:
        # session_params.session_id (nested)
        session_params = req_data.get("session_params") or {}
        if isinstance(session_params, dict) and session_params.get("session_id"):
            return f"session:{session_params['session_id']}"
        # user field
        if req_data.get("user"):
            return f"user:{req_data['user']}"
        # legacy session_id
        if req_data.get("session_id"):
            return f"session:{req_data['session_id']}"
        # legacy user_id
        if req_data.get("user_id"):
            return f"user:{req_data['user_id']}"

    return ''


async def get_idx_router(pd_state: Any, req_data: Any, req_header: Any, request_length: int) -> Tuple[int, float, bool]:
    """
    Returns (select_idx, instance_score, token_accounting_done).
    token_accounting_done indicates whether select_* was called (so release_* should be called).
    """
    if global_args.router_method == RouterMethod.LEAST_LOAD:
        if pd_state == InstanceType.PREFILL:
            instance_score = proxy_state.calculate_prefill_scores(request_length)
            select_idx = proxy_state.select_prefiller(instance_score)
            logger.debug(f"instance_score:{instance_score},select_idx:{select_idx}")
        elif pd_state == InstanceType.DECODE:
            instance_score = proxy_state.calculate_decode_scores(request_length)
            select_idx = proxy_state.select_decoder(instance_score)
        logger.debug(f"Request length: {request_length}, {pd_state} score: {instance_score}")
        return select_idx, instance_score, True
    elif global_args.router_method == RouterMethod.SESSION_AFFINITY:
        session_id = extract_session_id(req_data, req_header)
        instance_score = 0.0

        # Determine instance list and tainted list based on pd_state
        if pd_state == InstanceType.PREFILL:
            instance_score = proxy_state.calculate_prefill_scores(request_length)
            session_map = proxy_state.session_prefill_map
            instance_list = proxy_state.prefillers
            tainted_list = proxy_state.tainted_prefillers
        elif pd_state == InstanceType.DECODE:
            instance_score = proxy_state.calculate_decode_scores(request_length)
            session_map = proxy_state.session_decoder_map
            instance_list = proxy_state.decoders
            tainted_list = proxy_state.tainted_decoders
        else:
            raise ValueError(f"Unknown pd_state: {pd_state}")

        # -----------------------------------------------------------
        # SESSION_AFFINITY: Route based on session existence
        # Design principle:
        #   - Existing session: route directly to sticky node (no token accounting)
        #   - New session: select via Least Load, then record mapping (token accounting done)
        #
        # IMPORTANT: select_prefiller/select_decoder internally acquire
        # _prefiller_selection_lock / _decoder_selection_lock respectively,
        # so we must NOT hold _session_lock when calling them.
        # Lock ordering: _session_lock before select_* locks to avoid deadlock.
        # -----------------------------------------------------------

        # Step 1: Check if session already has a sticky mapping
        select_idx = None
        if session_id:
            with proxy_state._session_lock:
                if session_id in session_map:
                    cached_idx = session_map[session_id]
                    # Validate cached instance is still available
                    if cached_idx < len(instance_list) and instance_list[cached_idx] not in tainted_list:
                        session_map.move_to_end(session_id)
                        logger.debug(f"SESSION_AFFINITY: {pd_state} session {session_id} -> instance {cached_idx} (cached, NO token accounting)")
                        return cached_idx, instance_score, False  # NO token accounting for cached session

        # Step 2: New session or no session_id - use Least Load (select_* acquires its own lock internally)
        if pd_state == InstanceType.PREFILL:
            select_idx = proxy_state.select_prefiller(instance_score)
        else:
            select_idx = proxy_state.select_decoder(instance_score)

        # Step 3: Record mapping for new session
        if session_id:
            with proxy_state._session_lock:
                # LRU eviction if needed
                if len(session_map) >= proxy_state.SESSION_MAP_MAX_SIZE:
                    evict_count = len(session_map) - proxy_state.SESSION_MAP_MAX_SIZE + 1
                    for _ in range(evict_count):
                        session_map.popitem(last=False)
                session_map[session_id] = select_idx
                logger.debug(f"SESSION_AFFINITY: {pd_state} session {session_id} -> instance {select_idx} (new, tokens accounted)")
                # Note: token accounting already done by select_prefiller/select_decoder inside their own lock

        logger.debug(f"SESSION_AFFINITY: {pd_state} instance_score:{instance_score}, select_idx:{select_idx}")
        return select_idx, instance_score, True  # Token accounting done for new session


async def _handle_select_instance(api: str, req_data: Any, request_length: int, request_id: str, req_header: Any):
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug("Request length: %s, Prefiller score: %s", request_length, prefiller_score)
    
    try:
        prefiller_idx, prefiller_score, prefill_token_acct = await get_idx_router(InstanceType.PREFILL, req_data, req_header, request_length)
        logger.debug(f"content-length:{req_header.get('content-length')},user_id:{req_header.get('x-user-id')},prefiller_idx:{prefiller_idx}")
    except Exception as error:
        raise PrefillSelectionError(error) from error

    prefiller = proxy_state.prefillers[prefiller_idx]
    # 记录 prefill 开始时间
    prefill_start_time = time.time()

    try:
        response = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
        # 记录 prefill 结束时间
        prefill_end_time = time.time()
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

    except asyncio.CancelledError:
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise

    except httpx.HTTPStatusError:
        # 4xx 一般表示请求本身无效；send_request_to_service 已经区分 4xx。
        # 释放本地计数，但不要隔离健康 Prefill。
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise

    except Exception as error:
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        proxy_state.mark_prefiller_unavailable(prefiller)
        raise PrefillUnavailableError(prefiller, error) from error

    finally:
        # Release prefiller after prefill completes (only if select_prefiller was called - not for cached SESSION_AFFINITY)
        if proxy_state and prefill_token_acct:
            proxy_state.release_prefiller(prefiller_idx, prefiller_score)

    decoder_score = proxy_state.calculate_decode_scores(request_length)
    logger.debug("Decoder score: %f", decoder_score)

    try:
        decoder_idx, decoder_score, decoder_token_acct = await get_idx_router(InstanceType.DECODE, req_data, req_header, request_length)
        
    except Exception as error:
        # Prefill 已经成功，但此时 Decode 可能刚好被 NodeListener 隔离。
        # 旧 KV 不能悬挂，下一轮需要使用新的 backend_request_id 重做 Prefill。
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise DecoderUnavailableError(error) from error

    decoder = proxy_state.decoders[decoder_idx]
    logger.info(f"Using {prefiller.url}, {decoder.url}")

    return InstanceInfo(
        request_id=request_id,
        prefiller_idx=prefiller_idx,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder=decoder,
        decoder_idx=decoder_idx,
        decoder_score=decoder_score,
        decoder_token_acct=decoder_token_acct,
        prefill_start_time=prefill_start_time,
        prefill_end_time=prefill_end_time,
    )


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState
    decoder_token_acct: bool
    prefill_start_time: float
    prefill_end_time: float
    
class PrefillUnavailableError(RuntimeError):
    """表示本次请求使用的 Prefill 实例已经不可用。"""
    def __init__(self, server: ServerState, original_error: Exception):
        self.server = server
        self.original_error = original_error
        super().__init__(f"Prefill instance {server} is unavailable: {original_error}")

class DecodeStreamError(RuntimeError):
    """表示 Decode 流式请求发生异常。"""
    def __init__(self, original_error: Exception, partial_response_sent: bool):
        # 保存真正的底层异常，例如 httpx.ReadError。
        self.original_error = original_error
        # 是否已经向用户发送过部分内容。
        self.partial_response_sent = partial_response_sent
        super().__init__(
            "Decode stream failed; "
            f"partial_response_sent={partial_response_sent}; original_error={original_error}"
        )

class PrefillSelectionError(RuntimeError):
    """节点状态在检查和选择之间发生变化，当前没有健康 Prefill。"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(f"No healthy prefiller is available: {original_error}")


class DecoderUnavailableError(RuntimeError):
    """Prefill 已完成，但暂时没有健康 Decode 可接收本轮请求。"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(f"No healthy decoder is available: {original_error}")


class SSEEventBuffer:
    """把任意 TCP/HTTP 字节分片重新组装成完整 SSE data 事件。"""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._decoder.decode(chunk)
        # SSE 既可能使用 CRLF，也可能使用 LF。
        self._buffer = self._buffer.replace("\r\n", "\n")

        payloads: list[str] = []
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            data_lines = []
            for line in event.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
            if data_lines:
                payloads.append("\n".join(data_lines))
        return payloads


# OpenAI 标准结束原因，以及部分后端常见的兼容值。
# finish_reason=None 表示仍在生成，不属于结束状态。
NORMAL_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "function_call",
    "content_filter",
    "eos",
    "end_turn",
    "completed",
}


def get_recoverable_decode_error_detail(chunk_json: dict[str, Any]) -> str | None:
    """识别 APIServer 以 HTTP 200 返回的 Decode 内部故障事件。"""
    error_info = chunk_json.get("error")
    if not isinstance(error_info, dict):
        return None

    error_message = str(error_info.get("message") or "")
    error_type = str(error_info.get("type") or "")
    error_code = error_info.get("code")

    try:
        numeric_error_code = int(error_code)
    except (TypeError, ValueError):
        numeric_error_code = None

    recoverable_error = (
        numeric_error_code is not None
        and numeric_error_code >= 500
    ) or error_type == "InternalServerError" or any(
        marker in error_message
        for marker in (
            "EngineCore encountered",
            "EngineDeadError",
            "EngineCore process",
            "engine is dead",
        )
    )
    if not recoverable_error:
        return None

    return (
        f"type={error_type!r}, code={error_code!r}, "
        f"message={error_message!r}"
    )


THINK_START_MARKER = "<think>\n"
THINK_END_MARKER = "</think>\n\n"


def extract_generated_parts(choice: dict[str, Any]) -> tuple[str, str, bool]:
    """提取当前 choice 的 reasoning、content 以及 reasoning 字段是否存在。

    reasoning_field_present 用来判断后端是否启用了 reasoning parser。
    即使字段值当前为空，只要响应结构中出现 reasoning 或
    reasoning_content，也视为 parser 已启用。
    """
    container = choice.get("delta")
    if not isinstance(container, dict):
        container = choice.get("message")

    if isinstance(container, dict):
        reasoning_field_present = (
            "reasoning_content" in container
            or "reasoning" in container
        )

        reasoning_value = container.get("reasoning_content")
        if not isinstance(reasoning_value, str) or not reasoning_value:
            reasoning_value = container.get("reasoning")

        content_value = container.get("content")
        reasoning_piece = (
            reasoning_value
            if isinstance(reasoning_value, str) and reasoning_value
            else ""
        )
        content_piece = (
            content_value
            if isinstance(content_value, str) and content_value
            else ""
        )
        return reasoning_piece, content_piece, reasoning_field_present

    text_value = choice.get("text")
    content_piece = text_value if isinstance(text_value, str) else ""
    return "", content_piece, False


def is_thinking_enabled(req_data: dict[str, Any]) -> bool:
    """只有用户明确传入 enable_thinking=false 时才关闭思考。"""
    chat_template_kwargs = req_data.get("chat_template_kwargs")
    return not (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.get("enable_thinking") is False
    )


def ensure_default_thinking(req_data: dict[str, Any]) -> bool:
    """为 Chat 请求补齐默认 enable_thinking=true，并返回最终开关值。"""
    if "messages" not in req_data:
        return False

    chat_template_kwargs = req_data.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
        req_data["chat_template_kwargs"] = chat_template_kwargs

    chat_template_kwargs.setdefault("enable_thinking", True)
    return chat_template_kwargs.get("enable_thinking") is not False


def compose_generated_text(state: RequestRecoveryState) -> str:
    """根据 parser 状态和 thinking 开关构造恢复用 assistant.content。

    1. 已检测到 reasoning parser：
       - content 已有值：丢弃 reasoning，只续写 content；
       - 只有 reasoning：在前面补 <think> 后继续思考。
    2. 未检测到 reasoning parser：
       - enable_thinking=false：原样续写累计 content；
       - enable_thinking=true：始终在完整累计 content 前补 <think>。

    未启用 reasoning parser 时，思考文本、</think> 和正式回答都在
    content 中，因此必须完整保留，不能删除或截取 </think>。
    """
    if state.reasoning_parser_detected:
        if state.content_text:
            return state.content_text
        if state.reasoning_text:
            return THINK_START_MARKER + state.reasoning_text
        return ""

    raw_content = state.raw_content_text
    if not raw_content:
        return ""

    if not state.thinking_enabled:
        return raw_content

    return THINK_START_MARKER + raw_content


def _has_unclosed_think_marker(content: str) -> bool:
    """判断原 assistant.content 是否处于未结束的 <think> 中。"""
    return content.rfind("<think>") > content.rfind("</think>")


def _extract_visible_answer_content(content: str) -> str:
    """提取 Qwen Chat Template 最终会保留的正式回答部分。"""
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1].lstrip("\n")
    if _has_unclosed_think_marker(content):
        return ""
    return content


def _build_resume_assistant_content(
    state: RequestRecoveryState,
    original_content: str,
) -> str:
    """将本次已生成内容追加到原始 assistant 前缀。"""
    if state.reasoning_parser_detected:
        if state.content_text:
            original_answer = _extract_visible_answer_content(original_content)
            return original_answer + state.content_text

        if state.reasoning_text:
            if _has_unclosed_think_marker(original_content):
                return original_content + state.reasoning_text
            return original_content + THINK_START_MARKER + state.reasoning_text

        return original_content

    raw_content = state.raw_content_text
    if not raw_content:
        return original_content

    if not state.thinking_enabled:
        return original_content + raw_content

    # 未启用 reasoning parser 时，raw_content 中可能同时包含：
    # 思考文本、</think> 和正式回答。无论中断发生在哪个阶段，
    # 都完整保留累计内容，只确保最前面存在一个未由响应返回的 <think>。
    if _has_unclosed_think_marker(original_content):
        return original_content + raw_content
    return original_content + THINK_START_MARKER + raw_content


def build_resume_request(state: RequestRecoveryState) -> dict[str, Any]:
    """根据已经发送给客户端的模型文本构造恢复续推请求。"""
    resume_request = copy.deepcopy(state.original_request)
    resume_request.pop("kv_transfer_params", None)

    if "messages" in resume_request:
        messages = resume_request.get("messages") or []

        if messages and messages[-1].get("role") == "assistant":
            last_assistant = messages[-1]
            original_content = last_assistant.get("content") or ""
            last_assistant["content"] = _build_resume_assistant_content(
                state,
                original_content,
            )
            last_assistant.pop("reasoning", None)
            last_assistant.pop("reasoning_content", None)
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": _build_resume_assistant_content(state, ""),
                }
            )

        resume_request["messages"] = messages
        resume_request["continue_final_message"] = True
        resume_request["add_generation_prompt"] = False

    elif "prompt" in resume_request:
        original_prompt = resume_request.get("prompt") or ""
        resume_request["prompt"] = original_prompt + state.generated_text

    original_max_tokens = resume_request.get("max_tokens")
    if isinstance(original_max_tokens, int):
        resume_request["max_tokens"] = max(1, original_max_tokens - state.completion_tokens)

    original_max_completion_tokens = resume_request.get("max_completion_tokens")
    if isinstance(original_max_completion_tokens, int):
        resume_request["max_completion_tokens"] = max(1, original_max_completion_tokens - state.completion_tokens)

    return resume_request


KEEP_ALIVE_CHUNK = b": keep-alive\n\n"

@dataclass
class CompletionExecutionContext:
    """保存一次 Completions/Chat Completions 请求的执行状态。"""

    api: str
    req_data: dict[str, Any]
    recovery_state: RequestRecoveryState
    stream_flag: bool
    req_header: Any
    instance_info: InstanceInfo | None = None
    released_kv: bool = True
    completion_tokens: int = 0
    completion_tokens_before_attempt: int = 0
    done_received: bool = False
    normal_finish_received: bool = False
    recompute_requested: bool = False

    def release_prefiller_kv_once(self) -> None:
        """当前轮次的 Prefill KV 只释放一次。"""
        if self.instance_info is None or self.released_kv:
            return
        if not self.released_kv and global_args.router_method == RouterMethod.LEAST_LOAD:
            proxy_state.release_prefiller_kv(self.instance_info.prefiller_idx, self.instance_info.prefiller_score)
            self.released_kv = True

    def release_current_instance(self) -> None:
        """释放当前轮次的 Prefill KV 和 Decode 负载。"""
        if self.instance_info is None:
            return
        self.release_prefiller_kv_once()
        if proxy_state and self.instance_info.decoder_token_acct:
            proxy_state.release_decoder(self.instance_info.decoder_idx,  self.instance_info.decoder_score)

        # 无论是否进行过负载计数，都必须清理当前实例引用。
        self.instance_info = None
        self.released_kv = True

    def reset_decode_attempt(self) -> None:
        """开始新一轮 Decode 前重置仅属于本轮的状态。"""
        self.completion_tokens_before_attempt = self.completion_tokens
        self.done_received = False
        self.normal_finish_received = False
        self.recompute_requested = False

    def apply_next_request(self) -> None:
        """根据当前恢复状态构造下一轮请求，并原地更新 req_data。"""
        if self.recovery_state.generated_text:
            next_request = build_resume_request(self.recovery_state)
        else:
            # 只发送过 role/usage/心跳等元数据时，重新执行原始请求。
            next_request = copy.deepcopy(self.recovery_state.original_request)

        self.req_data.clear()
        self.req_data.update(next_request)
        self.recovery_state.current_request = copy.deepcopy(self.req_data)


async def _wait_for_completion_instance(
    context: CompletionExecutionContext,
) -> AsyncIterator[bytes]:
    """等待可用 P/D，并完成当前轮次的 Prefill 与 Decode 选择。"""
    state = context.recovery_state

    while context.instance_info is None:
        while not proxy_state.has_available_prefiller():
            state.phase = "waiting_prefill"
            available = await proxy_state.wait_for_prefiller(timeout=5.0)
            if not available and context.stream_flag:
                yield KEEP_ALIVE_CHUNK

        while not proxy_state.has_available_decoder():
            state.phase = "waiting_decode"
            available = await proxy_state.wait_for_decoder(timeout=5.0)
            if not available and context.stream_flag:
                yield KEEP_ALIVE_CHUNK

        state.phase = "prefill"
        # 上一轮的 KV 信息不能带入新的 Prefill。
        context.req_data.pop("kv_transfer_params", None)
        request_length = len(json.dumps(context.req_data, ensure_ascii=False).encode("utf-8"))

        try:
            context.instance_info = await _handle_select_instance(
                context.api,
                context.req_data,
                request_length,
                state.backend_request_id,
                context.req_header
            )
        except PrefillSelectionError as error:
            state.phase = "waiting_prefill"
            state.recovery_count += 1
            state.backend_request_id = await proxy_state.next_req_id()
            logger.warning(
                "[WAITING PREFILL SELECTION] "
                "logical_request_id=%s "
                "new_backend_request_id=%s "
                "error=%s",
                state.request_id,
                state.backend_request_id,
                error,
            )
            if context.stream_flag:
                yield KEEP_ALIVE_CHUNK
            continue
        except PrefillUnavailableError as error:
            state.phase = "waiting_prefill"
            state.recovery_count += 1
            state.backend_request_id = await proxy_state.next_req_id()
            logger.warning(
                "[WAITING PREFILL] "
                "logical_request_id=%s "
                "new_backend_request_id=%s "
                "failed_instance=%s",
                state.request_id,
                state.backend_request_id,
                error.server,
            )
            if context.stream_flag:
                yield KEEP_ALIVE_CHUNK
            continue
        except DecoderUnavailableError as error:
            state.phase = "waiting_decode"
            state.recovery_count += 1
            state.backend_request_id = await proxy_state.next_req_id()
            logger.warning(
                "[WAITING DECODE] "
                "logical_request_id=%s "
                "new_backend_request_id=%s "
                "error=%s",
                state.request_id,
                state.backend_request_id,
                error,
            )
            if context.stream_flag:
                yield KEEP_ALIVE_CHUNK
            continue

        # Prefill 成功，当前轮次持有新的 KV。
        context.released_kv = False
        state.phase = "decode"
        state.current_request = copy.deepcopy(context.req_data)


def _process_decode_payload(context: CompletionExecutionContext, payload: str) -> list[bytes]:
    """解析一个完整 SSE/JSON payload，并更新恢复状态。"""
    if not payload:
        return []

    state = context.recovery_state

    if payload == "[DONE]":
        context.done_received = True
        return [b"data: [DONE]\n\n"] if context.stream_flag else []

    try:
        chunk_json = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Forwarding non-JSON payload without state update: %s", payload)
        if context.stream_flag:
            return [(f"data: {payload}\n\n").encode("utf-8")]
        return []

    # EngineCore 被强制结束但 APIServer 仍存活时，vLLM 可能以
    # HTTP 200 返回顶层 error SSE，随后再发送 [DONE]。
    error_info = chunk_json.get("error")
    if isinstance(error_info, dict):
        error_detail = get_recoverable_decode_error_detail(chunk_json)
        if error_detail is not None:
            decoder = (
                context.instance_info.decoder
                if context.instance_info is not None
                else None
            )
            logger.warning(
                "[DECODE ERROR EVENT INTERCEPTED] "
                "logical_request_id=%s "
                "backend_request_id=%s "
                "decoder=%s %s",
                state.request_id,
                state.backend_request_id,
                decoder,
                error_detail,
            )
            raise DecodeStreamError(
                original_error=RuntimeError("decode returned recoverable error event: " + error_detail),
                partial_response_sent=bool(state.generated_text),
            )

        # 非 5xx 的请求级错误不应隔离健康 Decode，保持原样转发。
        if context.stream_flag:
            return [(f"data: {payload}\n\n").encode("utf-8")]
        return []

    choices = chunk_json.get("choices") or []
    if not choices:
        if context.stream_flag:
            return [(f"data: {payload}\n\n").encode("utf-8")]
        return []

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    stop_reason = choice.get("stop_reason")

    # recomputed 是现有协议中的特殊重算信号，不属于节点故障。
    if finish_reason == "recomputed" or stop_reason == "recomputed":
        context.recompute_requested = True
        return []

    # finish_reason=None 表示仍在生成；白名单之外的非空值进入恢复。
    if finish_reason is not None:
        if finish_reason in NORMAL_FINISH_REASONS:
            context.normal_finish_received = True
        else:
            abnormal_detail = (f"finish_reason={finish_reason!r}, stop_reason={stop_reason!r}")
            decoder = (
                context.instance_info.decoder
                if context.instance_info is not None
                else None
            )
            logger.warning(
                "[DECODE ABNORMAL FINISH INTERCEPTED] "
                "logical_request_id=%s "
                "backend_request_id=%s "
                "decoder=%s %s",
                state.request_id,
                state.backend_request_id,
                decoder,
                abnormal_detail,
            )
            raise DecodeStreamError(
                original_error=RuntimeError("decode returned abnormal terminal state: " + abnormal_detail),
                partial_response_sent=bool(state.generated_text),
            )

    reasoning_piece, content_piece, reasoning_field_present = extract_generated_parts(choice)
    generated_piece = reasoning_piece + content_piece

    if reasoning_field_present or reasoning_piece:
        state.reasoning_parser_detected = True

    if state.reasoning_parser_detected:
        # reasoning parser 已启用：直接依赖 reasoning/content 字段。
        if reasoning_piece:
            state.reasoning_text += reasoning_piece
        if content_piece:
            state.content_text += content_piece
    elif content_piece:
        # reasoning parser 未启用：思考与回答全部原样存在 content 中。
        state.raw_content_text += content_piece

    if generated_piece:
        state.generated_text = compose_generated_text(state)
        state.phase = "decode_generating"

    usage = chunk_json.get("usage") or {}
    reported_completion_tokens = usage.get("completion_tokens")
    if isinstance(reported_completion_tokens, int):
        context.completion_tokens = max(context.completion_tokens, context.completion_tokens_before_attempt + reported_completion_tokens)
    elif generated_piece:
        # 没有 usage 时，暂时按有效生成事件近似累计。
        context.completion_tokens += 1

    state.completion_tokens = context.completion_tokens

    if context.stream_flag:
        normalized_payload = json.dumps(chunk_json, ensure_ascii=False, separators=(",", ":"))
        return [(f"data: {normalized_payload}\n\n").encode("utf-8")]
    return []

async def _stream_decode_attempt(context: CompletionExecutionContext) -> AsyncIterator[bytes]:
    """执行一轮 Decode，并将后端响应转换为客户端响应。"""
    if context.instance_info is None:
        raise RuntimeError("Decode instance is not selected")

    context.reset_decode_attempt()
    sse_buffer = SSEEventBuffer()

    async for chunk in stream_service_response_with_retry(
        context.instance_info.decoder.client,
        context.api,
        context.req_data,
        request_id=context.recovery_state.backend_request_id,
        decoder=context.instance_info.decoder,
        max_retries=global_args.max_retries,
        base_delay=global_args.retry_delay,
    ):
        if not context.released_kv and chunk:
            context.release_prefiller_kv_once()

        if context.stream_flag:
            payloads = sse_buffer.feed(chunk)
        else:
            try:
                payloads = [chunk.decode("utf-8").strip()]
            except UnicodeDecodeError:
                logger.debug("Forwarding undecodable non-stream chunk: %s", chunk)
                yield chunk
                continue

        for payload in payloads:
            for output_chunk in _process_decode_payload(context, payload):
                yield output_chunk
            if context.recompute_requested:
                break

        if not context.stream_flag:
            if payloads:
                try:
                    normalized_non_stream = json.dumps(json.loads(payloads[-1]), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                except (json.JSONDecodeError, TypeError):
                    normalized_non_stream = chunk
                yield normalized_non_stream
            else:
                yield chunk

        if context.recompute_requested:
            break

    # 流式连接自然结束时，必须收到 [DONE]，或明确的正常 finish_reason。
    if (context.stream_flag and not context.recompute_requested and not context.done_received):
        if context.normal_finish_received:
            yield b"data: [DONE]\n\n"
            context.done_received = True
        else:
            raise DecodeStreamError(
                original_error=RuntimeError("decode stream ended without [DONE] or a normal finish_reason"),
                partial_response_sent=bool(context.recovery_state.generated_text),
            )


async def _recover_decode_failure(context: CompletionExecutionContext, error: DecodeStreamError) -> bool:
    """处理 Decode 故障；返回 False 表示响应已在 [DONE] 后完成。"""
    state = context.recovery_state

    # 某些连接可能在结束帧后再出现协议层关闭异常，此时不能续推。
    if context.done_received:
        logger.warning("[DECODE POST-DONE ERROR IGNORED] logical_request_id=%s error=%s", state.request_id, error)
        state.phase = "completed"
        return False

    failed_instance_info = context.instance_info
    if failed_instance_info is None:
        raise error

    failed_decoder = failed_instance_info.decoder
    logger.error(
        "[DECODE RECOVERY START] "
        "logical_request_id=%s "
        "backend_request_id=%s "
        "decoder=%s "
        "partial_response_sent=%s "
        "generated_text_length=%s",
        state.request_id,
        state.backend_request_id,
        failed_decoder,
        error.partial_response_sent,
        len(state.generated_text),
    )

    proxy_state.abort_prefiller_request(failed_instance_info.prefiller_idx, state.backend_request_id)
    proxy_state.mark_decoder_unavailable(failed_decoder)
    context.release_current_instance()

    state.recovery_count += 1
    state.phase = "waiting_decode"
    state.backend_request_id = await proxy_state.next_req_id()
    context.apply_next_request()

    logger.info(
        "[DECODE WAITING] "
        "logical_request_id=%s "
        "new_backend_request_id=%s "
        "recovery_count=%s",
        state.request_id,
        state.backend_request_id,
        state.recovery_count,
    )
    return True

async def _prepare_recompute(context: CompletionExecutionContext,) -> None:
    """处理后端 recomputed 信号，不隔离 Decode，重新执行当前请求。"""
    state = context.recovery_state
    current_instance_info = context.instance_info
    if current_instance_info is not None:
        proxy_state.abort_prefiller_request(current_instance_info.prefiller_idx, state.backend_request_id)

    context.release_current_instance()
    state.recovery_count += 1
    state.backend_request_id = await proxy_state.next_req_id()
    context.apply_next_request()

    logger.info(
        "[RECOMPUTE] logical_request_id=%s "
        "new_backend_request_id=%s recovery_count=%s",
        state.request_id,
        state.backend_request_id,
        state.recovery_count,
    )


async def _generate_completion_stream(context: CompletionExecutionContext) -> AsyncIterator[bytes]:
    """运行完整的 P/D 选择、Decode、故障恢复和续推循环。"""
    state = context.recovery_state

    try:
        while True:
            async for heartbeat in _wait_for_completion_instance(context):
                yield heartbeat

            try:
                async for chunk in _stream_decode_attempt(context):
                    yield chunk
            except DecodeStreamError as error:
                should_continue = await _recover_decode_failure(context, error)
                if not should_continue:
                    break
                if context.stream_flag:
                    yield KEEP_ALIVE_CHUNK
                continue

            if context.recompute_requested:
                await _prepare_recompute(context)
                continue

            state.phase = "completed"
            break

    except asyncio.CancelledError:
        # StreamingResponse 在客户端主动断开时会取消当前生成器。
        state.phase = "cancelled"
        logger.info(
            "[CLIENT DISCONNECTED] logical_request_id=%s "
            "backend_request_id=%s",
            state.request_id,
            state.backend_request_id,
        )
        if context.instance_info is not None:
            proxy_state.abort_prefiller_request(context.instance_info.prefiller_idx, state.backend_request_id)
        raise
    except Exception as error:
        failed_phase = state.phase
        state.phase = "failed"
        logger.exception(
            "[REQUEST FAILED] request_id=%s "
            "failed_phase=%s error=%s",
            state.request_id,
            failed_phase,
            error,
        )
        if context.instance_info is not None:
            proxy_state.abort_prefiller_request(context.instance_info.prefiller_idx, state.backend_request_id)
        raise
    finally:
        context.release_current_instance()
        proxy_state.request_num -= 1
        if state.phase in {"completed", "cancelled", "failed"}:
            proxy_state.remove_recovery_request(state.request_id)


async def _handle_completions(api: str, request: Request):
    """初始化请求上下文，并返回统一的流式/非流式响应。"""
    request_counted = False
    recovery_registered = False
    request_id: str | None = None

    try:
        proxy_state.request_num += 1
        request_counted = True

        req_data = await request.json()
        thinking_enabled = ensure_default_thinking(req_data)
        request_id = await proxy_state.next_req_id()
        req_header = request.headers
        recovery_state = RequestRecoveryState(
            request_id=request_id,
            backend_request_id=request_id,
            api=api,
            original_request=copy.deepcopy(req_data),
            current_request=copy.deepcopy(req_data),
            phase="created",
            thinking_enabled=thinking_enabled,
        )
        proxy_state.register_recovery_request(recovery_state)
        recovery_registered = True

        stream_flag = bool(req_data.get("stream", False))
        context = CompletionExecutionContext(
            api=api,
            req_data=req_data,
            recovery_state=recovery_state,
            stream_flag=stream_flag,
            req_header=req_header
        )
        media_type = (
            "text/event-stream; charset=utf-8"
            if stream_flag
            else "application/json"
        )
        return StreamingResponse(_generate_completion_stream(context), media_type=media_type)
    except Exception:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if recovery_registered and request_id is not None:
            proxy_state.remove_recovery_request(request_id)
        if request_counted:
            proxy_state.request_num -= 1
        raise

async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error("Failed to %s instances: %s", adjust_mode, e)
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
