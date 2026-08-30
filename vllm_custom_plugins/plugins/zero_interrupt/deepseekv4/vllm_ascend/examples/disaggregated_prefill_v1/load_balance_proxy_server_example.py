# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server supports two mutually-exclusive deployment modes:
# 1. PD disaggregated: separate "prefiller" and "decoder" backend servers.
# 2. PD mixed: each "mixed" backend performs both Prefill and Decode.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests in PD disaggregated or PD mixed deployments.
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
# PD disaggregated example:
#
#   python load_balance_proxy_server_bracket_group_checked.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 [127.0.0.2 127.0.0.2] \
#     --prefiller-ports 8100 8101 [8100 8101] \
#     --decoder-hosts 127.0.0.3 [127.0.0.4 127.0.0.4] \
#     --decoder-ports 8200 [8200 8201]
#
# PD mixed example (do not configure prefiller/decoder options at the same time):
#
#   python load_balance_proxy_server_bracket_group_checked.py \
#     --host 0.0.0.0 --port 9000 \
#     --mixed-hosts 127.0.0.1 127.0.0.1 \
#     --mixed-ports 8300 8301
#
# Each hosts/ports option appears once. Bare host/port pairs are independent
# one-member groups. Items enclosed by an unquoted [ ... ] pair form one
# multi-member group. Host and port group boundaries must match exactly.
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
class DeploymentMode:
    DISAGGREGATED: str = "disaggregated"
    MIXED: str = "mixed"


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"
    MIXED: str = "mixed"

@dataclass
class RequestRecoveryState:
    request_id: str
    api: str
    original_request: dict[str, Any]
    # 当前准备发送给 P/D 节点的请求。
    current_request: dict[str, Any]
    # 请求目前所处阶段。
    # created、prefill、decode、decode_generating、mixed、mixed_generating、
    # waiting_prefill、waiting_decode、waiting_mixed、completed、cancelled、failed
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
    # 当前这一轮发送给后端（P/D 或 MIXED）的内部请求 ID。
    # 故障恢复时会重新生成，但逻辑 request_id 不变。
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


class ServerGroupState:
    """A logical DP/EP failure domain containing one or more HTTP rank endpoints."""

    def __init__(
        self,
        group_id: str,
        instance_type: str,
        servers: list[ServerState],
    ):
        if not servers:
            raise ValueError(f"{instance_type} group {group_id} must contain at least one server")
        self.group_id = group_id
        self.instance_type = instance_type
        self.servers = servers
        self.tainted = False
        self.pending_removal = False
        self.failed_member: ServerState | None = None

    def __repr__(self):
        members = ", ".join(str(server) for server in self.servers)
        return f"{self.group_id}({members})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "type": self.instance_type,
            "members": [str(server) for server in self.servers],
            "available": not self.tainted and not self.pending_removal,
            "tainted": self.tainted,
            "pending_removal": self.pending_removal,
            "failed_member": str(self.failed_member) if self.failed_member else None,
        }


class ProxyState:
    def __init__(
        self,
        prefiller_instance_groups,
        decoder_instance_groups,
        mixed_instance_groups=None,
        deployment_mode: str = DeploymentMode.DISAGGREGATED,
    ):
        self.request_num = 0
        self.deployment_mode = deployment_mode
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.tainted_mixed: list[ServerState] = []

        self.prefiller_groups: list[ServerGroupState] = []
        self.decoder_groups: list[ServerGroupState] = []
        self.mixed_groups: list[ServerGroupState] = []
        self.prefiller_group_by_server: dict[ServerState, ServerGroupState] = {}
        self.decoder_group_by_server: dict[ServerState, ServerGroupState] = {}
        self.mixed_group_by_server: dict[ServerState, ServerGroupState] = {}
        self._next_prefiller_group_id = 0
        self._next_decoder_group_id = 0
        self._next_mixed_group_id = 0

        for instances in prefiller_instance_groups:
            group = self._new_group(InstanceType.PREFILL, instances)
            self.prefiller_groups.append(group)
        for instances in decoder_instance_groups:
            group = self._new_group(InstanceType.DECODE, instances)
            self.decoder_groups.append(group)
        for instances in mixed_instance_groups or []:
            group = self._new_group(InstanceType.MIXED, instances)
            self.mixed_groups.append(group)

        self.prefillers: list[ServerState] = []
        self.decoders: list[ServerState] = []
        self.mixed_instances: list[ServerState] = []
        self._rebuild_prefiller_topology()
        self._rebuild_decoder_topology()
        self._rebuild_mixed_topology()

        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()

        # Selection locks prevent concurrent requests and the health thread from
        # mutating the same heap at the same time.
        self._prefiller_selection_lock = threading.Lock()
        self._decoder_selection_lock = threading.Lock()
        self._mixed_selection_lock = threading.Lock()

        # 保存所有正在执行或等待恢复的请求。
        self.recovery_requests: dict[str, RequestRecoveryState] = {}
        self.recovery_requests_lock = threading.RLock()
        self.event_loop = asyncio.get_running_loop()
        self.prefiller_available_event = asyncio.Event()
        self.decoder_available_event = asyncio.Event()
        self.mixed_available_event = asyncio.Event()

        # Each entry is (priority_score, server_index, server_reference).
        self.prefiller_heap = [(0.0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0.0, i, server) for i, server in enumerate(self.decoders)]
        self.mixed_heap = [(0.0, i, server) for i, server in enumerate(self.mixed_instances)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        heapq.heapify(self.mixed_heap)

        self.session_prefill_map: OrderedDict = OrderedDict()
        self.session_decoder_map: OrderedDict = OrderedDict()
        self.session_mixed_map: OrderedDict = OrderedDict()
        self._session_lock = threading.Lock()
        self.SESSION_MAP_MAX_SIZE = 10000

        self._sync_availability_events()
        self.node_listener = NodeListener(self)

    def _new_group(
        self,
        instance_type: str,
        instances: list[tuple[str, int]] | list[ServerState],
    ) -> ServerGroupState:
        servers: list[ServerState] = []
        for instance in instances:
            if isinstance(instance, ServerState):
                servers.append(instance)
            else:
                host, port = instance
                servers.append(ServerState(host, port))

        if instance_type == InstanceType.PREFILL:
            group_id = f"prefill-{self._next_prefiller_group_id}"
            self._next_prefiller_group_id += 1
        elif instance_type == InstanceType.DECODE:
            group_id = f"decode-{self._next_decoder_group_id}"
            self._next_decoder_group_id += 1
        elif instance_type == InstanceType.MIXED:
            group_id = f"mixed-{self._next_mixed_group_id}"
            self._next_mixed_group_id += 1
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        return ServerGroupState(group_id, instance_type, servers)

    def _rebuild_prefiller_topology(self) -> None:
        self.prefillers = [server for group in self.prefiller_groups for server in group.servers]
        self.prefiller_group_by_server = {
            server: group for group in self.prefiller_groups for server in group.servers
        }
        self.tainted_prefillers = [
            server
            for group in self.prefiller_groups
            if group.tainted or group.pending_removal
            for server in group.servers
        ]

    def _rebuild_decoder_topology(self) -> None:
        self.decoders = [server for group in self.decoder_groups for server in group.servers]
        self.decoder_group_by_server = {
            server: group for group in self.decoder_groups for server in group.servers
        }
        self.tainted_decoders = [
            server
            for group in self.decoder_groups
            if group.tainted or group.pending_removal
            for server in group.servers
        ]

    def _rebuild_mixed_topology(self) -> None:
        self.mixed_instances = [server for group in self.mixed_groups for server in group.servers]
        self.mixed_group_by_server = {
            server: group for group in self.mixed_groups for server in group.servers
        }
        self.tainted_mixed = [
            server
            for group in self.mixed_groups
            if group.tainted or group.pending_removal
            for server in group.servers
        ]

    def _rebuild_prefiller_heap(self) -> None:
        self.prefiller_heap = []
        for idx, server in enumerate(self.prefillers):
            group = self.prefiller_group_by_server[server]
            priority = (
                TAINT_PRIORITY
                if group.tainted or group.pending_removal
                else server.active_tokens + server.active_kv_cache * 0.3
            )
            self.prefiller_heap.append((priority, idx, server))
        heapq.heapify(self.prefiller_heap)

    def _rebuild_decoder_heap(self) -> None:
        self.decoder_heap = []
        for idx, server in enumerate(self.decoders):
            group = self.decoder_group_by_server[server]
            priority = (
                TAINT_PRIORITY
                if group.tainted or group.pending_removal
                else server.active_tokens
            )
            self.decoder_heap.append((priority, idx, server))
        heapq.heapify(self.decoder_heap)

    def _rebuild_mixed_heap(self) -> None:
        self.mixed_heap = []
        for idx, server in enumerate(self.mixed_instances):
            group = self.mixed_group_by_server[server]
            priority = (
                TAINT_PRIORITY
                if group.tainted or group.pending_removal
                else server.active_tokens
            )
            self.mixed_heap.append((priority, idx, server))
        heapq.heapify(self.mixed_heap)

    def _get_group(self, instance_type: str, server: ServerState) -> ServerGroupState | None:
        if instance_type == InstanceType.PREFILL:
            return self.prefiller_group_by_server.get(server)
        if instance_type == InstanceType.DECODE:
            return self.decoder_group_by_server.get(server)
        if instance_type == InstanceType.MIXED:
            return self.mixed_group_by_server.get(server)
        raise ValueError(f"Unknown instance type: {instance_type}")

    def _get_groups_for_servers(
        self,
        instance_type: str,
        servers: list[ServerState],
    ) -> list[ServerGroupState]:
        groups: list[ServerGroupState] = []
        seen_group_ids: set[str] = set()
        for server in servers:
            group = self._get_group(instance_type, server)
            if group is not None and group.group_id not in seen_group_ids:
                groups.append(group)
                seen_group_ids.add(group.group_id)
        return groups

    def _update_prefiller_priority(self, server_idx: int):
        server = self.prefillers[server_idx]
        group = self.prefiller_group_by_server[server]
        if group.tainted or group.pending_removal:
            priority = TAINT_PRIORITY
        else:
            priority = server.active_tokens + server.active_kv_cache * 0.3
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        server = self.decoders[server_idx]
        group = self.decoder_group_by_server[server]
        if group.tainted or group.pending_removal:
            priority = TAINT_PRIORITY
        else:
            priority = server.active_tokens
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def _update_mixed_priority(self, server_idx: int):
        server = self.mixed_instances[server_idx]
        group = self.mixed_group_by_server[server]
        if group.tainted or group.pending_removal:
            priority = TAINT_PRIORITY
        else:
            priority = server.active_tokens
        self.mixed_heap = [(p, i, s) for p, i, s in self.mixed_heap if i != server_idx]
        heapq.heappush(self.mixed_heap, (priority, server_idx, server))

    def abort_prefiller_request(self, server_idx: int, request_id):
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):
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

    def has_available_mixed(self) -> bool:
        return any(priority < TAINT_PRIORITY for priority, _, _ in self.mixed_heap)

    async def wait_for_prefiller(self, timeout: float = 5.0) -> bool:
        if self.has_available_prefiller():
            return True
        self.prefiller_available_event.clear()
        if self.has_available_prefiller():
            self.prefiller_available_event.set()
            return True
        try:
            await asyncio.wait_for(self.prefiller_available_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        available = self.has_available_prefiller()
        if not available:
            self.prefiller_available_event.clear()
        return available

    async def wait_for_decoder(self, timeout: float = 5.0) -> bool:
        if self.has_available_decoder():
            return True
        self.decoder_available_event.clear()
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

    async def wait_for_mixed(self, timeout: float = 5.0) -> bool:
        if self.has_available_mixed():
            return True
        self.mixed_available_event.clear()
        if self.has_available_mixed():
            self.mixed_available_event.set()
            return True
        try:
            await asyncio.wait_for(self.mixed_available_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        available = self.has_available_mixed()
        if not available:
            self.mixed_available_event.clear()
        return available

    def _sync_availability_events(self) -> None:
        def apply_event_states() -> None:
            if self.has_available_prefiller():
                self.prefiller_available_event.set()
            else:
                self.prefiller_available_event.clear()
            if self.has_available_decoder():
                self.decoder_available_event.set()
            else:
                self.decoder_available_event.clear()
            if self.has_available_mixed():
                self.mixed_available_event.set()
            else:
                self.mixed_available_event.clear()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self.event_loop:
            apply_event_states()
        else:
            self.event_loop.call_soon_threadsafe(apply_event_states)

    def list_recovery_requests(self) -> list[RequestRecoveryState]:
        with self.recovery_requests_lock:
            return list(self.recovery_requests.values())

    def select_prefiller(self, token_count):
        """Select one rank endpoint from all currently healthy Prefill groups."""
        with self._prefiller_selection_lock:
            if not self.has_available_prefiller():
                raise RuntimeError("No prefiller groups available")
            priority, chosen, server = heapq.heappop(self.prefiller_heap)
            if priority >= TAINT_PRIORITY:
                heapq.heappush(self.prefiller_heap, (priority, chosen, server))
                raise RuntimeError("No prefiller groups available")
            self.prefillers[chosen].active_tokens += token_count
            self.prefillers[chosen].active_kv_cache += token_count
            self._update_prefiller_priority(chosen)
            return chosen

    def release_prefiller(self, idx, token_count):
        with self._prefiller_selection_lock:
            if idx >= len(self.prefillers):
                return
            self.prefillers[idx].active_tokens = max(
                0, self.prefillers[idx].active_tokens - token_count
            )
            self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        with self._prefiller_selection_lock:
            if idx >= len(self.prefillers):
                return
            self.prefillers[idx].active_kv_cache = max(
                0, self.prefillers[idx].active_kv_cache - token_count
            )
            self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):
        """Select one rank endpoint from all currently healthy Decode groups."""
        with self._decoder_selection_lock:
            if not self.has_available_decoder():
                raise RuntimeError("No decoder groups available")
            priority, chosen, server = heapq.heappop(self.decoder_heap)
            if priority >= TAINT_PRIORITY:
                heapq.heappush(self.decoder_heap, (priority, chosen, server))
                raise RuntimeError("No decoder groups available")
            self.decoders[chosen].active_tokens += token_count
            self._update_decoder_priority(chosen)
            return chosen

    def release_decoder(self, idx, token_count):
        with self._decoder_selection_lock:
            if idx >= len(self.decoders):
                return
            self.decoders[idx].active_tokens = max(
                0, self.decoders[idx].active_tokens - token_count
            )
            self._update_decoder_priority(idx)

    def select_mixed(self, token_count):
        """Select one rank endpoint from all currently healthy MIXED groups."""
        with self._mixed_selection_lock:
            if not self.has_available_mixed():
                raise RuntimeError("No mixed groups available")
            priority, chosen, server = heapq.heappop(self.mixed_heap)
            if priority >= TAINT_PRIORITY:
                heapq.heappush(self.mixed_heap, (priority, chosen, server))
                raise RuntimeError("No mixed groups available")
            self.mixed_instances[chosen].active_tokens += token_count
            self._update_mixed_priority(chosen)
            return chosen

    def release_mixed(self, idx, token_count):
        with self._mixed_selection_lock:
            if idx >= len(self.mixed_instances):
                return
            self.mixed_instances[idx].active_tokens = max(
                0, self.mixed_instances[idx].active_tokens - token_count
            )
            self._update_mixed_priority(idx)

    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        return length_score * 0.0345 + 120.0745

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    def calculate_mixed_scores(self, request_length: int) -> float:
        # MIXED 实例同时承担 Prefill + Decode，这里沿用请求长度作为负载近似值。
        return request_length

    def _append_group(self, group: ServerGroupState) -> None:
        if group.instance_type == InstanceType.PREFILL:
            with self._prefiller_selection_lock:
                self.prefiller_groups.append(group)
                self._rebuild_prefiller_topology()
                self._rebuild_prefiller_heap()
        elif group.instance_type == InstanceType.DECODE:
            with self._decoder_selection_lock:
                self.decoder_groups.append(group)
                self._rebuild_decoder_topology()
                self._rebuild_decoder_heap()
        elif group.instance_type == InstanceType.MIXED:
            with self._mixed_selection_lock:
                self.mixed_groups.append(group)
                self._rebuild_mixed_topology()
                self._rebuild_mixed_heap()
        else:
            raise ValueError(f"Unknown instance type: {group.instance_type}")
        self._sync_availability_events()

    async def add_instance_groups(
        self,
        instance_type: str,
        instance_groups: list[list[ServerState]],
    ) -> tuple[list[str], list[str]]:
        added_groups: list[str] = []
        waiting_groups: list[str] = []

        for servers in instance_groups:
            existing = [self._get_group(instance_type, server) for server in servers]
            existing = [group for group in existing if group is not None]
            if existing:
                existing_ids = {group.group_id for group in existing}
                if len(existing_ids) != 1 or set(existing[0].servers) != set(servers):
                    raise ValueError(
                        "An endpoint already belongs to another group; group membership cannot overlap"
                    )
                group = existing[0]
            else:
                group = self._new_group(instance_type, servers)
                # A newly discovered group starts isolated. It becomes selectable
                # only after every member passes the initial health check.
                group.tainted = True
                self._append_group(group)

            all_healthy = all(
                self.node_listener.check_instance_status(server) for server in group.servers
            )
            if all_healthy:
                self.restore_group(group)
                added_groups.append(group.group_id)
            else:
                self.taint_group(group)
                self.node_listener.watch_group(group, failed_server=None)
                waiting_groups.append(group.group_id)

        return added_groups, waiting_groups

    async def add_instances(
        self,
        instance_type: str,
        instances: list[ServerState],
    ) -> tuple[list[str], list[str]]:
        # Backward-compatible dynamic API: each item in "instances" is a
        # one-member group. Use the optional "groups" API field for a multi-rank group.
        return await self.add_instance_groups(instance_type, [[server] for server in instances])

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            group = self.prefiller_group_by_server.get(server)
            if group is None:
                group = self._new_group(InstanceType.PREFILL, [server])
                self._append_group(group)
            else:
                self.restore_group(group)
        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            group = self.decoder_group_by_server.get(server)
            if group is None:
                group = self._new_group(InstanceType.DECODE, [server])
                self._append_group(group)
            else:
                self.restore_group(group)
        self.print_status(f"Add decoder instances: {instances}.")

    def add_mixed_instances(self, instances: list[ServerState]) -> None:
        for server in instances:
            group = self.mixed_group_by_server.get(server)
            if group is None:
                group = self._new_group(InstanceType.MIXED, [server])
                self._append_group(group)
            else:
                self.restore_group(group)
        self.print_status(f"Add mixed instances: {instances}.")

    def taint_group(
        self,
        group: ServerGroupState,
        failed_server: ServerState | None = None,
    ) -> None:
        if group.pending_removal:
            return

        if group.instance_type == InstanceType.PREFILL:
            with self._prefiller_selection_lock:
                group.tainted = True
                if failed_server is not None:
                    group.failed_member = failed_server
                for server in group.servers:
                    if server not in self.tainted_prefillers:
                        self.tainted_prefillers.append(server)
                self._rebuild_prefiller_heap()
        elif group.instance_type == InstanceType.DECODE:
            with self._decoder_selection_lock:
                group.tainted = True
                if failed_server is not None:
                    group.failed_member = failed_server
                for server in group.servers:
                    if server not in self.tainted_decoders:
                        self.tainted_decoders.append(server)
                self._rebuild_decoder_heap()
        elif group.instance_type == InstanceType.MIXED:
            with self._mixed_selection_lock:
                group.tainted = True
                if failed_server is not None:
                    group.failed_member = failed_server
                for server in group.servers:
                    if server not in self.tainted_mixed:
                        self.tainted_mixed.append(server)
                self._rebuild_mixed_heap()
        else:
            raise ValueError(f"Unknown instance type: {group.instance_type}")
        self._sync_availability_events()

    def restore_group(self, group: ServerGroupState) -> None:
        if group.pending_removal:
            return

        if group.instance_type == InstanceType.PREFILL:
            with self._prefiller_selection_lock:
                group.tainted = False
                group.failed_member = None
                self.tainted_prefillers = [
                    server for server in self.tainted_prefillers if server not in group.servers
                ]
                self._rebuild_prefiller_heap()
        elif group.instance_type == InstanceType.DECODE:
            with self._decoder_selection_lock:
                group.tainted = False
                group.failed_member = None
                self.tainted_decoders = [
                    server for server in self.tainted_decoders if server not in group.servers
                ]
                self._rebuild_decoder_heap()
        elif group.instance_type == InstanceType.MIXED:
            with self._mixed_selection_lock:
                group.tainted = False
                group.failed_member = None
                self.tainted_mixed = [
                    server for server in self.tainted_mixed if server not in group.servers
                ]
                self._rebuild_mixed_heap()
        else:
            raise ValueError(f"Unknown instance type: {group.instance_type}")
        self._sync_availability_events()
        logger.info("[GROUP RECOVERED] %s", group)

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        for group in self._get_groups_for_servers(InstanceType.PREFILL, instances):
            self.taint_group(group)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        for group in self._get_groups_for_servers(InstanceType.DECODE, instances):
            self.taint_group(group)

    def _taint_mixed(self, instances: list[ServerState]) -> None:
        for group in self._get_groups_for_servers(InstanceType.MIXED, instances):
            self.taint_group(group)

    def mark_prefiller_unavailable(self, server: ServerState) -> None:
        group = self.prefiller_group_by_server.get(server)
        if group is None:
            logger.warning("[PREFILL UNAVAILABLE] unknown instance=%s", server)
            return
        self.taint_group(group, failed_server=server)
        self.node_listener.watch_group(group, failed_server=server)
        logger.warning(
            "[PREFILL GROUP UNAVAILABLE] group=%s failed_member=%s members=%s",
            group.group_id,
            server,
            [str(member) for member in group.servers],
        )

    def mark_decoder_unavailable(self, server: ServerState) -> None:
        group = self.decoder_group_by_server.get(server)
        if group is None:
            logger.warning("[DECODE UNAVAILABLE] unknown instance=%s", server)
            return
        self.taint_group(group, failed_server=server)
        self.node_listener.watch_group(group, failed_server=server)
        logger.warning(
            "[DECODE GROUP UNAVAILABLE] group=%s failed_member=%s members=%s",
            group.group_id,
            server,
            [str(member) for member in group.servers],
        )

    def mark_mixed_unavailable(self, server: ServerState) -> None:
        group = self.mixed_group_by_server.get(server)
        if group is None:
            logger.warning("[MIXED UNAVAILABLE] unknown instance=%s", server)
            return
        self.taint_group(group, failed_server=server)
        self.node_listener.watch_group(group, failed_server=server)
        logger.warning(
            "[MIXED GROUP UNAVAILABLE] group=%s failed_member=%s members=%s",
            group.group_id,
            server,
            [str(member) for member in group.servers],
        )

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        return self._remove_instance_groups(InstanceType.PREFILL, instances)

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        return self._remove_instance_groups(InstanceType.DECODE, instances)

    def remove_mixed(self, instances: list[ServerState]) -> bool:
        return self._remove_instance_groups(InstanceType.MIXED, instances)

    def _remove_instance_groups(
        self,
        instance_type: str,
        instances: list[ServerState],
    ) -> bool:
        groups = self._get_groups_for_servers(instance_type, instances)
        if not groups:
            return False

        if self.request_num > 0:
            for group in groups:
                group.pending_removal = True
                self.taint_group_for_removal(group)
                self.node_listener.cancel_group_watch(group)
            logger.warning(
                "Groups are isolated and will be removed after active requests finish: %s",
                [group.group_id for group in groups],
            )
            return True

        self._remove_groups_now(instance_type, groups)
        return False

    def taint_group_for_removal(self, group: ServerGroupState) -> None:
        if group.instance_type == InstanceType.PREFILL:
            with self._prefiller_selection_lock:
                group.tainted = True
                group.pending_removal = True
                for server in group.servers:
                    if server not in self.tainted_prefillers:
                        self.tainted_prefillers.append(server)
                self._rebuild_prefiller_heap()
        elif group.instance_type == InstanceType.DECODE:
            with self._decoder_selection_lock:
                group.tainted = True
                group.pending_removal = True
                for server in group.servers:
                    if server not in self.tainted_decoders:
                        self.tainted_decoders.append(server)
                self._rebuild_decoder_heap()
        elif group.instance_type == InstanceType.MIXED:
            with self._mixed_selection_lock:
                group.tainted = True
                group.pending_removal = True
                for server in group.servers:
                    if server not in self.tainted_mixed:
                        self.tainted_mixed.append(server)
                self._rebuild_mixed_heap()
        else:
            raise ValueError(f"Unknown instance type: {group.instance_type}")
        self._sync_availability_events()

    def _remove_groups_now(
        self,
        instance_type: str,
        groups: list[ServerGroupState],
    ) -> None:
        group_ids = {group.group_id for group in groups}
        for group in groups:
            self.node_listener.cancel_group_watch(group)

        if instance_type == InstanceType.PREFILL:
            with self._prefiller_selection_lock:
                self.prefiller_groups = [
                    group for group in self.prefiller_groups if group.group_id not in group_ids
                ]
                self._rebuild_prefiller_topology()
                self._rebuild_prefiller_heap()
                with self._session_lock:
                    self.session_prefill_map.clear()
        elif instance_type == InstanceType.DECODE:
            with self._decoder_selection_lock:
                self.decoder_groups = [
                    group for group in self.decoder_groups if group.group_id not in group_ids
                ]
                self._rebuild_decoder_topology()
                self._rebuild_decoder_heap()
                with self._session_lock:
                    self.session_decoder_map.clear()
        elif instance_type == InstanceType.MIXED:
            with self._mixed_selection_lock:
                self.mixed_groups = [
                    group for group in self.mixed_groups if group.group_id not in group_ids
                ]
                self._rebuild_mixed_topology()
                self._rebuild_mixed_heap()
                with self._session_lock:
                    self.session_mixed_map.clear()
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        self._sync_availability_events()
        self.print_status(f"Removed {instance_type} groups: {sorted(group_ids)}.")

    def finalize_pending_removals(self) -> None:
        if self.request_num > 0:
            return
        prefill_groups = [group for group in self.prefiller_groups if group.pending_removal]
        decode_groups = [group for group in self.decoder_groups if group.pending_removal]
        mixed_groups = [group for group in self.mixed_groups if group.pending_removal]
        if prefill_groups:
            self._remove_groups_now(InstanceType.PREFILL, prefill_groups)
        if decode_groups:
            self._remove_groups_now(InstanceType.DECODE, decode_groups)
        if mixed_groups:
            self._remove_groups_now(InstanceType.MIXED, mixed_groups)

    def group_status(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "prefill_groups": [group.to_dict() for group in self.prefiller_groups],
            "decode_groups": [group.to_dict() for group in self.decoder_groups],
            "mixed_groups": [group.to_dict() for group in self.mixed_groups],
        }

    def print_status(self, msg: str) -> None:
        status = self.group_status()
        status["deployment_mode"] = self.deployment_mode
        status["prefill_instances"] = [str(server) for server in self.prefillers]
        status["decode_instances"] = [str(server) for server in self.decoders]
        status["mixed_instances"] = [str(server) for server in self.mixed_instances]
        print(f"{msg} Status: {status}")


proxy_state = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        # key -> (group, check_times). Recovery is group-scoped: every member
        # must pass the same recovery cycle before the group is re-enabled.
        self.waiting_groups: dict[str, tuple[ServerGroupState, int]] = {}
        self._waiting_groups_lock = threading.RLock()
        self.check_timeout = 10.0
        self.max_failures = 3
        self.failure_counters: dict[str, int] = {}

        self.listening_thread = threading.Thread(
            target=self._node_listener,
            daemon=True,
        )
        self.listening_thread.start()
        logger.info("NodeListener background thread started.")

    @staticmethod
    def _group_key(group: ServerGroupState) -> str:
        return f"{group.instance_type}:{group.group_id}"

    @staticmethod
    def _failure_key(instance_type: str, server: ServerState) -> str:
        return f"{instance_type}:{server.host}:{server.port}"

    def watch_group(
        self,
        group: ServerGroupState,
        failed_server: ServerState | None,
    ) -> None:
        if group.pending_removal:
            return
        if failed_server is not None:
            group.failed_member = failed_server
        key = self._group_key(group)
        with self._waiting_groups_lock:
            previous = self.waiting_groups.get(key)
            check_times = previous[1] if previous else 0
            self.waiting_groups[key] = (group, check_times)
        for member in group.servers:
            self.failure_counters.pop(
                self._failure_key(group.instance_type, member),
                None,
            )

    def cancel_group_watch(self, group: ServerGroupState) -> None:
        with self._waiting_groups_lock:
            self.waiting_groups.pop(self._group_key(group), None)

    def _snapshot_waiting_groups(self) -> list[tuple[str, ServerGroupState, int]]:
        with self._waiting_groups_lock:
            return [
                (key, group, check_times)
                for key, (group, check_times) in self.waiting_groups.items()
            ]

    def _node_listener(self) -> None:
        while True:
            try:
                self._check_waiting_groups()
                self._check_active_servers(InstanceType.PREFILL)
                self._check_active_servers(InstanceType.DECODE)
                self._check_active_servers(InstanceType.MIXED)
                self.proxy_state.finalize_pending_removals()
            except Exception:
                logger.exception("Unexpected error in NodeListener loop")
            time.sleep(global_args.waiting_retry_interval)

    def _check_waiting_groups(self) -> None:
        waiting_snapshot = self._snapshot_waiting_groups()
        if waiting_snapshot:
            logger.debug("Checking %s groups waiting for recovery.", len(waiting_snapshot))

        for key, group, check_times in waiting_snapshot:
            if group.pending_removal:
                self.cancel_group_watch(group)
                continue

            unhealthy_members: list[str] = []
            for server in group.servers:
                if not self.check_instance_status(server, self.check_timeout):
                    unhealthy_members.append(str(server))

            check_times += 1
            if not unhealthy_members:
                logger.info(
                    "[GROUP RECOVERY] %s all members recovered; adding the whole group back.",
                    group,
                )
                self.proxy_state.restore_group(group)
                with self._waiting_groups_lock:
                    self.waiting_groups.pop(key, None)
                continue

            with self._waiting_groups_lock:
                if key in self.waiting_groups:
                    self.waiting_groups[key] = (group, check_times)
            if check_times == 1 or check_times % 10 == 0:
                logger.warning(
                    "[GROUP WAITING] group=%s checked=%s unhealthy_members=%s",
                    group.group_id,
                    check_times,
                    unhealthy_members,
                )

    def _check_active_servers(self, instance_type: str) -> None:
        if instance_type == InstanceType.PREFILL:
            servers = list(self.proxy_state.prefillers)
            tainted_servers = self.proxy_state.tainted_prefillers
        elif instance_type == InstanceType.DECODE:
            servers = list(self.proxy_state.decoders)
            tainted_servers = self.proxy_state.tainted_decoders
        elif instance_type == InstanceType.MIXED:
            servers = list(self.proxy_state.mixed_instances)
            tainted_servers = self.proxy_state.tainted_mixed
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        for server in servers:
            if server in tainted_servers:
                continue
            self._check_and_handle_active_node(server, instance_type)

    def _check_and_handle_active_node(
        self,
        server: ServerState,
        instance_type: str,
    ) -> None:
        server_key = self._failure_key(instance_type, server)
        is_valid = self.check_instance_status(server, self.check_timeout)

        if is_valid:
            self.failure_counters.pop(server_key, None)
            return

        current_failures = self.failure_counters.get(server_key, 0) + 1
        self.failure_counters[server_key] = current_failures

        if current_failures < self.max_failures:
            logger.warning(
                "[WARNING] %s instance %s check failed (%s/%s).",
                instance_type,
                server,
                current_failures,
                self.max_failures,
            )
            return

        logger.error(
            "[DOWN] %s instance %s failed %s times; isolating its entire group.",
            instance_type,
            server,
            current_failures,
        )
        if instance_type == InstanceType.PREFILL:
            self.proxy_state.mark_prefiller_unavailable(server)
        elif instance_type == InstanceType.DECODE:
            self.proxy_state.mark_decoder_unavailable(server)
        elif instance_type == InstanceType.MIXED:
            self.proxy_state.mark_mixed_unavailable(server)
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

    @staticmethod
    def check_instance_status(server: Any, timeout: float = 10) -> bool:
        """
        Request /metrics. A connection failure, non-2xx response, or the
        backend sentinel 999999.0 means the rank endpoint is unhealthy.
        """
        host = server.host
        try:
            ip = ipaddress.ip_address(host)
            if isinstance(ip, ipaddress.IPv6Address):
                host = f"[{host}]"
        except Exception:
            pass

        url = f"http://{host}:{server.port}/metrics"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if "999999.0" in response.text:
                logger.warning(
                    "[HEALTH CHECK] %s reported failure code '999999'.",
                    url,
                )
                return False
            return True
        except Exception as error:
            logger.warning(
                "[HEALTH CHECK] Failed to connect to %s. Error: %s - %s",
                url,
                type(error).__name__,
                error,
            )
            return False


def _normalize_endpoint_key(host: str, port: int) -> tuple[str, int]:
    normalized_host = host.replace("localhost", "0.0.0.0").replace(
        "127.0.0.1", "0.0.0.0"
    )
    return normalized_host, int(port)


def _parse_bracket_groups(values: list[str], option_name: str) -> list[list[str]]:
    if not values:
        raise ValueError(f"{option_name} cannot be empty")

    groups: list[list[str]] = []
    current_group: list[str] | None = None

    for token in values:
        starts_group = token.startswith("[")
        ends_group = token.endswith("]")

        if current_group is None:
            if starts_group:
                first_value = token[1:]
                current_group = []
                if first_value:
                    current_group.append(first_value)
                continue

            if ends_group:
                raise ValueError(f"{option_name}: unexpected closing bracket in {token!r}")
            if "[" in token or "]" in token:
                raise ValueError(f"{option_name}: invalid bracket placement in {token!r}")

            groups.append([token])
            continue

        # Inside an already-open bracket group.
        if starts_group or "[" in token:
            raise ValueError(f"{option_name}: nested bracket groups are not supported")

        if ends_group:
            value = token[:-1]
            if "]" in value:
                raise ValueError(f"{option_name}: invalid bracket placement in {token!r}")
            if value:
                current_group.append(value)
            if not current_group:
                raise ValueError(f"{option_name}: bracket group cannot be empty")
            groups.append(current_group)
            current_group = None
            continue

        if "]" in token:
            raise ValueError(f"{option_name}: invalid bracket placement in {token!r}")
        current_group.append(token)

    if current_group is not None:
        raise ValueError(f"{option_name}: missing closing bracket")

    return groups


def _build_instance_groups(
    hosts: list[str] | None,
    ports: list[str] | None,
    instance_type: str,
    default_host: str,
    default_port: int,
) -> list[list[tuple[str, int]]]:
    if hosts is None and ports is None:
        hosts = [default_host]
        ports = [str(default_port)]
    elif hosts is None or ports is None:
        raise ValueError(
            f"--{instance_type}-hosts and --{instance_type}-ports must be configured together"
        )

    host_groups = _parse_bracket_groups(hosts, f"--{instance_type}-hosts")
    port_groups_raw = _parse_bracket_groups(ports, f"--{instance_type}-ports")

    if len(host_groups) != len(port_groups_raw):
        raise ValueError(
            f"{instance_type}: host group count ({len(host_groups)}) must match "
            f"port group count ({len(port_groups_raw)})"
        )

    groups: list[list[tuple[str, int]]] = []
    seen_endpoints: dict[tuple[str, int], int] = {}

    for group_index, (host_group, port_group_raw) in enumerate(
        zip(host_groups, port_groups_raw)
    ):
        if len(host_group) != len(port_group_raw):
            raise ValueError(
                f"{instance_type} group {group_index}: number of hosts "
                f"({len(host_group)}) must match number of ports "
                f"({len(port_group_raw)})"
            )

        port_group: list[int] = []
        for raw_port in port_group_raw:
            try:
                port = int(raw_port)
            except ValueError as error:
                raise ValueError(
                    f"{instance_type} group {group_index}: invalid port {raw_port!r}"
                ) from error
            if not 1 <= port <= 65535:
                raise ValueError(
                    f"{instance_type} group {group_index}: port {port} must be in 1..65535"
                )
            port_group.append(port)

        group: list[tuple[str, int]] = []
        for host, port in zip(host_group, port_group):
            endpoint_key = _normalize_endpoint_key(host, port)
            if endpoint_key in seen_endpoints:
                previous_group = seen_endpoints[endpoint_key]
                raise ValueError(
                    f"Duplicate {instance_type} endpoint {host}:{port} appears in "
                    f"groups {previous_group} and {group_index}. One endpoint can only "
                    f"belong to one group."
                )
            seen_endpoints[endpoint_key] = group_index
            group.append((host, port))
        groups.append(group)

    return groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument(
        "--prefiller-hosts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Prefill hosts. Bare values are singleton groups; values inside "
            "an unquoted [ ... ] pair form one multi-rank group."
        ),
    )
    parser.add_argument(
        "--prefiller-ports",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Prefill ports using the same group boundaries as --prefiller-hosts."
        ),
    )
    parser.add_argument(
        "--decoder-hosts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Decode hosts. Bare values are singleton groups; values inside "
            "an unquoted [ ... ] pair form one multi-rank group."
        ),
    )
    parser.add_argument(
        "--decoder-ports",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Decode ports using the same group boundaries as --decoder-hosts."
        ),
    )
    parser.add_argument(
        "--mixed-hosts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "PD-mixed hosts. When configured, --mixed-ports must also be set and "
            "Prefill/Decode host/port options must not be configured. Bare values "
            "are singleton groups; [ ... ] forms one multi-rank failure group."
        ),
    )
    parser.add_argument(
        "--mixed-ports",
        type=str,
        nargs="+",
        default=None,
        help="PD-mixed ports using the same group boundaries as --mixed-hosts.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for HTTP requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.001,
        help="Base delay (seconds) for exponential backoff retries",
    )
    parser.add_argument(
        "--max-waiting-retries",
        type=int,
        default=3,
        help=(
            "Deprecated compatibility option. Recovery waiting is time-based; "
            "use --recovery-wait-timeout instead."
        ),
    )
    parser.add_argument(
        "--recovery-wait-timeout",
        type=float,
        default=1200.0,
        help=(
            "Maximum continuous time in seconds that one request may wait/retry "
            "for healthy Prefill/Decode instances during recovery. Default: 1200 (20 minutes). "
            "Set <= 0 to disable the timeout."
        ),
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting groups to recover",
    )
    parser.add_argument(
        "--router-method",
        type=str,
        choices=[RouterMethod.LEAST_LOAD, RouterMethod.SESSION_AFFINITY],
        default=RouterMethod.LEAST_LOAD,
        help="Router method for selecting backend rank endpoints",
    )
    parser.add_argument(
        "--custom-session-id-headers",
        type=str,
        nargs="+",
        default=[],
        help="Custom HTTP header names for session ID extraction (highest priority)",
    )
    args = parser.parse_args()

    mixed_configured = args.mixed_hosts is not None or args.mixed_ports is not None
    disaggregated_configured = any(
        value is not None
        for value in (
            args.prefiller_hosts,
            args.prefiller_ports,
            args.decoder_hosts,
            args.decoder_ports,
        )
    )

    if mixed_configured:
        if args.mixed_hosts is None or args.mixed_ports is None:
            raise ValueError("--mixed-hosts and --mixed-ports must be configured together")
        if disaggregated_configured:
            raise ValueError(
                "PD mixed deployment and PD disaggregated deployment cannot be configured "
                "at the same time. Use either --mixed-hosts/--mixed-ports or the "
                "--prefiller-*/--decoder-* options."
            )

        args.deployment_mode = DeploymentMode.MIXED
        args.prefiller_groups = []
        args.decoder_groups = []
        args.mixed_groups = _build_instance_groups(
            args.mixed_hosts,
            args.mixed_ports,
            "mixed",
            "localhost",
            8000,
        )
    else:
        args.deployment_mode = DeploymentMode.DISAGGREGATED
        args.mixed_groups = []
        args.prefiller_groups = _build_instance_groups(
            args.prefiller_hosts,
            args.prefiller_ports,
            "prefiller",
            "localhost",
            8001,
        )
        args.decoder_groups = _build_instance_groups(
            args.decoder_hosts,
            args.decoder_ports,
            "decoder",
            "localhost",
            8002,
        )

    args.prefiller_instances = [
        instance for group in args.prefiller_groups for instance in group
    ]
    args.decoder_instances = [
        instance for group in args.decoder_groups for instance in group
    ]
    args.mixed_instances = [
        instance for group in args.mixed_groups for instance in group
    ]
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(
        global_args.prefiller_groups,
        global_args.decoder_groups,
        global_args.mixed_groups,
        deployment_mode=global_args.deployment_mode,
    )
    print(
        f"Initialized deployment_mode={proxy_state.deployment_mode}; "
        f"{len(proxy_state.prefiller_groups)} prefill groups / "
        f"{len(proxy_state.prefillers)} prefill clients; "
        f"{len(proxy_state.decoder_groups)} decode groups / "
        f"{len(proxy_state.decoders)} decode clients; "
        f"{len(proxy_state.mixed_groups)} mixed groups / "
        f"{len(proxy_state.mixed_instances)} mixed clients."
    )
    proxy_state.print_status("Initial grouped topology.")
    yield
    closed_servers: set[ServerState] = set()
    for server in proxy_state.prefillers + proxy_state.decoders + proxy_state.mixed_instances:
        if server in closed_servers:
            continue
        closed_servers.add(server)
        await server.client.aclose()


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
    server: ServerState | None = None,
    instance_type: str = InstanceType.DECODE,
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

                            if server is not None and proxy_state is not None:
                                if instance_type == InstanceType.DECODE:
                                    tainted = server in proxy_state.tainted_decoders
                                elif instance_type == InstanceType.MIXED:
                                    tainted = server in proxy_state.tainted_mixed
                                else:
                                    tainted = False
                                if tainted:
                                    raise RuntimeError(
                                        f"{instance_type} instance {server} was marked unavailable "
                                        "while its response stream was still open"
                                    )
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
                    "[BACKEND STREAM INTERRUPTED] "
                    "request_id=%s attempt=%s type=%s server=%s error=%s",
                    request_id,
                    attempt,
                    instance_type,
                    server,
                    error,
                )
                raise DecodeStreamError(original_error=error, partial_response_sent=True) from error
            # 一个 chunk 都没返回，可以进行短暂重试。
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[BACKEND RETRY] request_id=%s "
                    "attempt=%s/%s delay=%s type=%s server=%s error=%s",
                    request_id,
                    attempt,
                    max_retries,
                    delay,
                    instance_type,
                    server,
                    error,
                )
                await asyncio.sleep(delay)
                continue
            # 一个 chunk 都没返回，但重试次数已经耗尽。
            logger.error(
                "[BACKEND REQUEST FAILED] "
                "request_id=%s attempts=%s type=%s server=%s error=%s",
                request_id,
                max_retries,
                instance_type,
                server,
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
        elif pd_state == InstanceType.MIXED:
            instance_score = proxy_state.calculate_mixed_scores(request_length)
            select_idx = proxy_state.select_mixed(instance_score)
        else:
            raise ValueError(f"Unknown pd_state: {pd_state}")
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
        elif pd_state == InstanceType.MIXED:
            instance_score = proxy_state.calculate_mixed_scores(request_length)
            session_map = proxy_state.session_mixed_map
            instance_list = proxy_state.mixed_instances
            tainted_list = proxy_state.tainted_mixed
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
        elif pd_state == InstanceType.DECODE:
            select_idx = proxy_state.select_decoder(instance_score)
        elif pd_state == InstanceType.MIXED:
            select_idx = proxy_state.select_mixed(instance_score)
        else:
            raise ValueError(f"Unknown pd_state: {pd_state}")

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
        if prefill_token_acct:
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise

    except httpx.HTTPStatusError as error:
        # 4xx 表示请求本身有问题，保持原错误语义，不把健康 Prefill 当成故障。
        # 5xx 则表示服务端当前不可用：立即把请求纳入故障恢复流程，
        # 不必再等 NodeListener 连续 3 次健康检查失败后才开始切流。
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        if prefill_token_acct:
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)

        status_code = error.response.status_code
        if 400 <= status_code < 500:
            raise

        proxy_state.mark_prefiller_unavailable(prefiller)
        raise PrefillUnavailableError(prefiller, error) from error

    except Exception as error:
        proxy_state.abort_prefiller_request(prefiller_idx, request_id)
        if prefill_token_acct:
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        # 请求链路已经直接证明该 Prefill 不可用时，立即隔离其 Group。
        # 后续请求会优先切到其他健康实例；若没有健康实例，则进入等待恢复。
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
        if prefill_token_acct:
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


async def _handle_select_mixed_instance(
    req_data: Any,
    request_length: int,
    request_id: str,
    req_header: Any,
):
    """选择一个 PD 混部实例。实际请求在后续流式执行阶段发送。"""
    try:
        mixed_idx, mixed_score, mixed_token_acct = await get_idx_router(
            InstanceType.MIXED,
            req_data,
            req_header,
            request_length,
        )
    except Exception as error:
        raise MixedSelectionError(error) from error

    mixed = proxy_state.mixed_instances[mixed_idx]
    logger.info("Using mixed instance %s", mixed.url)
    return MixedInstanceInfo(
        request_id=request_id,
        mixed_idx=mixed_idx,
        mixed_score=mixed_score,
        mixed=mixed,
        mixed_token_acct=mixed_token_acct,
    )


@dataclass
class MixedInstanceInfo:
    request_id: str
    mixed_idx: int
    mixed_score: float
    mixed: ServerState
    mixed_token_acct: bool


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


class MixedSelectionError(RuntimeError):
    """当前没有健康的 PD 混部实例可供选择。"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(f"No healthy mixed instance is available: {original_error}")


class RecoveryWaitTimeoutError(RuntimeError):
    """等待健康 Prefill/Decode/MIXED 超过请求级恢复等待上限。"""

    def __init__(self, phase: str, timeout_seconds: float):
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"No healthy inference instance became available within "
            f"{timeout_seconds:.1f}s while phase={phase}"
        )


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


def _build_resume_assistant_content(state: RequestRecoveryState, original_content: str,) -> str:
    """将本次已生成内容追加到原始 assistant 前缀。"""
    
    # 1. 启用 reasoning parser
    if state.reasoning_parser_detected:
        if state.content_text:
            original_answer = _extract_visible_answer_content(original_content)
            return original_answer + state.content_text

        if state.reasoning_text:
            if _has_unclosed_think_marker(original_content):
                return original_content + state.reasoning_text

            return (original_content + THINK_START_MARKER + state.reasoning_text)

        return original_content

    # 2. 未启用 reasoning parser
    raw_content = state.raw_content_text

    if not raw_content:
        return original_content

    # 3. enable_thinking=false
    if not state.thinking_enabled:
        return original_content + raw_content

    # 4. enable_thinking=true 已经出现</think>：
    if "</think>" in raw_content:
        generated_answer = raw_content.rsplit("</think>", 1)[1].lstrip("\n")
        original_answer = _extract_visible_answer_content(original_content)
        return original_answer + generated_answer
    
    # 5. 还没有 </think>
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
    instance_info: InstanceInfo | MixedInstanceInfo | None = None
    released_kv: bool = True
    completion_tokens: int = 0
    completion_tokens_before_attempt: int = 0
    done_received: bool = False
    normal_finish_received: bool = False
    recompute_requested: bool = False
    # 当前连续恢复/等待窗口的起点。None 表示当前没有处于恢复等待窗口。
    recovery_wait_started_at: float | None = None
    recovery_wait_phase: str | None = None

    def start_recovery_wait(self, phase: str) -> None:
        """开始或延续一次连续恢复等待窗口。"""
        if self.recovery_wait_started_at is None:
            self.recovery_wait_started_at = time.monotonic()
            logger.info(
                "[RECOVERY WAIT START] logical_request_id=%s phase=%s timeout=%ss",
                self.recovery_state.request_id,
                phase,
                getattr(global_args, "recovery_wait_timeout", 1200.0),
            )
        self.recovery_wait_phase = phase

    def clear_recovery_wait(self) -> None:
        """成功选到新的 P/D 后结束本次连续恢复等待窗口。"""
        if self.recovery_wait_started_at is not None:
            waited = time.monotonic() - self.recovery_wait_started_at
            logger.info(
                "[RECOVERY WAIT END] logical_request_id=%s waited=%.3fs",
                self.recovery_state.request_id,
                waited,
            )
        self.recovery_wait_started_at = None
        self.recovery_wait_phase = None

    def get_recovery_wait_timeout(self, phase: str, poll_interval: float = 5.0) -> float:
        """返回本轮 Event.wait 的 timeout，并保证总恢复等待不超过配置上限。"""
        self.start_recovery_wait(phase)
        max_wait = float(getattr(global_args, "recovery_wait_timeout", 1200.0))
        if max_wait <= 0:
            return poll_interval

        elapsed = time.monotonic() - self.recovery_wait_started_at
        remaining = max_wait - elapsed
        if remaining <= 0:
            raise RecoveryWaitTimeoutError(phase, max_wait)
        return min(poll_interval, remaining)

    def ensure_recovery_wait_not_timed_out(self) -> None:
        """在快速重选实例的循环中也检查恢复等待上限。"""
        if self.recovery_wait_started_at is None:
            return
        max_wait = float(getattr(global_args, "recovery_wait_timeout", 1200.0))
        if max_wait <= 0:
            return
        elapsed = time.monotonic() - self.recovery_wait_started_at
        if elapsed >= max_wait:
            raise RecoveryWaitTimeoutError(
                self.recovery_wait_phase or "waiting_instance",
                max_wait,
            )

    def is_mixed_mode(self) -> bool:
        return proxy_state.deployment_mode == DeploymentMode.MIXED

    def current_backend(self) -> ServerState | None:
        if self.instance_info is None:
            return None
        if isinstance(self.instance_info, MixedInstanceInfo):
            return self.instance_info.mixed
        return self.instance_info.decoder

    def current_stream_instance_type(self) -> str:
        return InstanceType.MIXED if self.is_mixed_mode() else InstanceType.DECODE

    def abort_current_prefill_if_needed(self) -> None:
        if self.instance_info is None or isinstance(self.instance_info, MixedInstanceInfo):
            return
        proxy_state.abort_prefiller_request(
            self.instance_info.prefiller_idx,
            self.recovery_state.backend_request_id,
        )

    def release_prefiller_kv_once(self) -> None:
        """当前轮次的 Prefill KV 只释放一次。"""
        if self.instance_info is None or self.released_kv:
            return
        if isinstance(self.instance_info, MixedInstanceInfo):
            return
        if not self.released_kv and global_args.router_method == RouterMethod.LEAST_LOAD:
            proxy_state.release_prefiller_kv(self.instance_info.prefiller_idx, self.instance_info.prefiller_score)
            self.released_kv = True

    def release_current_instance(self) -> None:
        """释放当前轮次的后端负载；分离模式还需要释放 Prefill KV。"""
        if self.instance_info is None:
            return

        if isinstance(self.instance_info, MixedInstanceInfo):
            if proxy_state and self.instance_info.mixed_token_acct:
                proxy_state.release_mixed(
                    self.instance_info.mixed_idx,
                    self.instance_info.mixed_score,
                )
        else:
            self.release_prefiller_kv_once()
            if proxy_state and self.instance_info.decoder_token_acct:
                proxy_state.release_decoder(
                    self.instance_info.decoder_idx,
                    self.instance_info.decoder_score,
                )

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


async def _wait_for_mixed_instance(
    context: CompletionExecutionContext,
) -> AsyncIterator[bytes]:
    """等待并选择一个可用 PD 混部实例。"""
    state = context.recovery_state

    while context.instance_info is None:
        while not proxy_state.has_available_mixed():
            state.phase = "waiting_mixed"
            wait_timeout = context.get_recovery_wait_timeout("waiting_mixed")
            available = await proxy_state.wait_for_mixed(timeout=wait_timeout)
            if not available:
                context.ensure_recovery_wait_not_timed_out()
                if context.stream_flag:
                    yield KEEP_ALIVE_CHUNK

        context.ensure_recovery_wait_not_timed_out()
        state.phase = "mixed"
        # MIXED 模式不使用 PD KV Transfer，恢复请求也不能携带旧 kv_transfer_params。
        context.req_data.pop("kv_transfer_params", None)
        request_length = len(
            json.dumps(context.req_data, ensure_ascii=False).encode("utf-8")
        )

        try:
            context.instance_info = await _handle_select_mixed_instance(
                context.req_data,
                request_length,
                state.backend_request_id,
                context.req_header,
            )
        except MixedSelectionError as error:
            state.phase = "waiting_mixed"
            context.start_recovery_wait("waiting_mixed")
            state.recovery_count += 1
            state.backend_request_id = await proxy_state.next_req_id()
            logger.warning(
                "[WAITING MIXED] logical_request_id=%s "
                "new_backend_request_id=%s error=%s",
                state.request_id,
                state.backend_request_id,
                error,
            )
            if context.stream_flag:
                yield KEEP_ALIVE_CHUNK
            continue

        context.clear_recovery_wait()
        context.released_kv = True
        state.phase = "mixed"
        state.current_request = copy.deepcopy(context.req_data)


async def _wait_for_completion_instance(
    context: CompletionExecutionContext,
) -> AsyncIterator[bytes]:
    """根据部署模式等待并选择当前轮次的后端实例。"""
    if proxy_state.deployment_mode == DeploymentMode.MIXED:
        async for heartbeat in _wait_for_mixed_instance(context):
            yield heartbeat
        return

    state = context.recovery_state

    while context.instance_info is None:
        while not proxy_state.has_available_prefiller():
            state.phase = "waiting_prefill"
            wait_timeout = context.get_recovery_wait_timeout("waiting_prefill")
            available = await proxy_state.wait_for_prefiller(timeout=wait_timeout)
            if not available:
                context.ensure_recovery_wait_not_timed_out()
                if context.stream_flag:
                    yield KEEP_ALIVE_CHUNK

        while not proxy_state.has_available_decoder():
            state.phase = "waiting_decode"
            wait_timeout = context.get_recovery_wait_timeout("waiting_decode")
            available = await proxy_state.wait_for_decoder(timeout=wait_timeout)
            if not available:
                context.ensure_recovery_wait_not_timed_out()
                if context.stream_flag:
                    yield KEEP_ALIVE_CHUNK

        # 即使当前 heap 中还有“可用”实例，也可能正处于 NodeListener 的 3 次失败确认窗口。
        # 如果请求实际打过去失败，下面的异常分支会立即隔离该组并继续重选。
        context.ensure_recovery_wait_not_timed_out()
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
            context.start_recovery_wait("waiting_prefill")
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
            context.start_recovery_wait("waiting_prefill")
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
            context.start_recovery_wait("waiting_decode")
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

        # Prefill + Decode 都成功选定，结束本次连续恢复等待窗口。
        context.clear_recovery_wait()
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
            decoder = context.current_backend()
            logger.warning(
                "[INFERENCE ERROR EVENT INTERCEPTED] "
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
            decoder = context.current_backend()
            logger.warning(
                "[INFERENCE ABNORMAL FINISH INTERCEPTED] "
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
        state.phase = (
            "mixed_generating"
            if context.is_mixed_mode()
            else "decode_generating"
        )

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

    backend = context.current_backend()
    if backend is None:
        raise RuntimeError("Inference backend is not selected")

    async for chunk in stream_service_response_with_retry(
        backend.client,
        context.api,
        context.req_data,
        request_id=context.recovery_state.backend_request_id,
        server=backend,
        instance_type=context.current_stream_instance_type(),
        max_retries=global_args.max_retries,
        base_delay=global_args.retry_delay,
    ):
        if not context.is_mixed_mode() and not context.released_kv and chunk:
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
    """处理 Decode/MIXED 生成阶段故障；返回 False 表示响应已完成。"""
    state = context.recovery_state

    # 某些连接可能在结束帧后再出现协议层关闭异常，此时不能续推。
    if context.done_received:
        logger.warning(
            "[POST-DONE ERROR IGNORED] logical_request_id=%s error=%s",
            state.request_id,
            error,
        )
        state.phase = "completed"
        return False

    failed_instance_info = context.instance_info
    if failed_instance_info is None:
        raise error

    failed_backend = context.current_backend()
    if failed_backend is None:
        raise error

    logger.error(
        "[INFERENCE RECOVERY START] logical_request_id=%s "
        "backend_request_id=%s mode=%s backend=%s "
        "partial_response_sent=%s generated_text_length=%s",
        state.request_id,
        state.backend_request_id,
        proxy_state.deployment_mode,
        failed_backend,
        error.partial_response_sent,
        len(state.generated_text),
    )

    if isinstance(failed_instance_info, MixedInstanceInfo):
        # MIXED 模式下当前实例同时承担 Prefill + Decode。请求侧已经确认它不可用时，
        # 立即隔离整个 MIXED Group，不必等待 NodeListener 的 3 次失败确认。
        proxy_state.mark_mixed_unavailable(failed_backend)
        context.release_current_instance()
        state.recovery_count += 1
        state.phase = "waiting_mixed"
        context.start_recovery_wait("waiting_mixed")
        state.backend_request_id = await proxy_state.next_req_id()
        context.apply_next_request()

        logger.info(
            "[MIXED WAITING] logical_request_id=%s "
            "new_backend_request_id=%s recovery_count=%s",
            state.request_id,
            state.backend_request_id,
            state.recovery_count,
        )
        return True

    # PD 分离模式保持原有恢复逻辑：隔离故障 Decode，重新 Prefill 后再选 Decode。
    proxy_state.abort_prefiller_request(
        failed_instance_info.prefiller_idx,
        state.backend_request_id,
    )
    proxy_state.mark_decoder_unavailable(failed_backend)
    context.release_current_instance()

    state.recovery_count += 1
    state.phase = "waiting_decode"
    context.start_recovery_wait("waiting_decode")
    state.backend_request_id = await proxy_state.next_req_id()
    context.apply_next_request()

    logger.info(
        "[DECODE WAITING] logical_request_id=%s "
        "new_backend_request_id=%s recovery_count=%s",
        state.request_id,
        state.backend_request_id,
        state.recovery_count,
    )
    return True

async def _prepare_recompute(context: CompletionExecutionContext,) -> None:
    """处理后端 recomputed 信号，不隔离 Decode，重新执行当前请求。"""
    state = context.recovery_state
    current_instance_info = context.instance_info
    if current_instance_info is not None and isinstance(current_instance_info, InstanceInfo):
        proxy_state.abort_prefiller_request(
            current_instance_info.prefiller_idx,
            state.backend_request_id,
        )

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
    """运行 PD 分离或 PD 混部模式下的选择、生成、故障恢复和续推循环。"""
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

    except RecoveryWaitTimeoutError as error:
        # 当前架构使用 StreamingResponse，响应头在等待结束前已经发出，
        # 因此超时后无法再把 HTTP 状态码改成 503/504；这里以 OpenAI 风格
        # error payload 结束流，避免由未捕获异常转成服务端 500。
        state.phase = "failed"
        logger.error(
            "[RECOVERY WAIT TIMEOUT] logical_request_id=%s phase=%s timeout=%ss recovery_count=%s",
            state.request_id,
            error.phase,
            error.timeout_seconds,
            state.recovery_count,
        )
        error_payload = {
            "error": {
                "message": str(error),
                "type": "service_unavailable",
                "code": "recovery_wait_timeout",
            }
        }
        serialized_error = json.dumps(
            error_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if context.stream_flag:
            yield b"data: " + serialized_error + b"\n\n"
            yield b"data: [DONE]\n\n"
        else:
            yield serialized_error

    except asyncio.CancelledError:
        # StreamingResponse 在客户端主动断开时会取消当前生成器。
        state.phase = "cancelled"
        logger.info(
            "[CLIENT DISCONNECTED] logical_request_id=%s "
            "backend_request_id=%s",
            state.request_id,
            state.backend_request_id,
        )
        context.abort_current_prefill_if_needed()
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
        context.abort_current_prefill_if_needed()
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
        supported_types = [InstanceType.PREFILL, InstanceType.DECODE, InstanceType.MIXED]
        if instance_type not in supported_types:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support {supported_types}."
            }

        if proxy_state.deployment_mode == DeploymentMode.MIXED and instance_type != InstanceType.MIXED:
            return {"error": "Current deployment mode is mixed; only mixed instances can be adjusted."}
        if proxy_state.deployment_mode == DeploymentMode.DISAGGREGATED and instance_type == InstanceType.MIXED:
            return {"error": "Current deployment mode is disaggregated; mixed instances cannot be adjusted."}

        groups_payload = req_data.get("groups")
        if groups_payload is not None:
            instance_groups = trans_instance_groups(groups_payload)
        else:
            instances = req_data.get("instances", [])
            if isinstance(instances, str):
                instances = [instances]
            # Backward-compatible behavior: each item is one singleton group.
            instance_groups = [[server] for server in trans_instances(instances)]

        flat_instances = [server for group in instance_groups for server in group]
        all_msg = (
            f"{adjust_mode} {instance_type} groups: "
            f"{[[str(server) for server in group] for group in instance_groups]}."
        )

        if adjust_mode == "add":
            added_groups, waiting_groups = await proxy_state.add_instance_groups(
                instance_type,
                instance_groups,
            )
            all_msg = (
                f"Added {instance_type} groups: {added_groups}. "
                f"Waiting for recovery: {waiting_groups}."
            )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(flat_instances)
            elif instance_type == InstanceType.DECODE:
                need_waiting = proxy_state.remove_decoders(flat_instances)
            else:
                need_waiting = proxy_state.remove_mixed(flat_instances)
            if need_waiting:
                all_msg = (
                    "The containing groups are isolated and will be removed "
                    "after active requests finish."
                )
            else:
                all_msg = "The containing groups were removed."

        result = {
            "message": all_msg,
            "current_prefill_instances": [
                str(prefiller) for prefiller in proxy_state.prefillers
            ],
            "current_decode_instances": [
                str(decoder) for decoder in proxy_state.decoders
            ],
            "current_mixed_instances": [
                str(server) for server in proxy_state.mixed_instances
            ],
            "deployment_mode": proxy_state.deployment_mode,
        }
        result.update(proxy_state.group_status())
        return result
    except Exception as error:
        logger.error("Failed to %s instances/groups: %s", adjust_mode, error)
        raise


def _split_instance_endpoint(instance: str) -> tuple[str, int]:
    instance = instance.strip()
    if instance.startswith("["):
        closing = instance.find("]")
        if closing == -1 or closing + 1 >= len(instance) or instance[closing + 1] != ":":
            raise ValueError(f"Invalid IPv6 endpoint: {instance}")
        return instance[1:closing], int(instance[closing + 2:])
    host, separator, port = instance.rpartition(":")
    if not separator or not host or not port:
        raise ValueError(f"Invalid endpoint: {instance}; expected host:port")
    return host, int(port)


def trans_instances(instances: list[str]) -> list[ServerState]:
    return [ServerState(*_split_instance_endpoint(instance)) for instance in instances]


def trans_instance_groups(groups: Any) -> list[list[ServerState]]:
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list of endpoint lists")
    result: list[list[ServerState]] = []
    for group_index, group in enumerate(groups):
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list) or not group:
            raise ValueError(f"groups[{group_index}] must be a non-empty endpoint list")
        result.append(trans_instances(group))
    return result



@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    status = {
        "status": "ok",
        "deployment_mode": proxy_state.deployment_mode,
        "prefill_instances": len(proxy_state.prefillers),
        "available_prefill_instances": sum(
            server not in proxy_state.tainted_prefillers for server in proxy_state.prefillers
        ),
        "decode_instances": len(proxy_state.decoders),
        "available_decode_instances": sum(
            server not in proxy_state.tainted_decoders for server in proxy_state.decoders
        ),
        "mixed_instances": len(proxy_state.mixed_instances),
        "available_mixed_instances": sum(
            server not in proxy_state.tainted_mixed for server in proxy_state.mixed_instances
        ),
    }
    status.update(proxy_state.group_status())
    return status


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
