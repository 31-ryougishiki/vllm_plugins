#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""ITSMultiprocExecutor for zero-interruption inference.

This module provides the ITS (Intelligent Transform Service) implementation
of MultiprocExecutor with fault keep and deployment strategy execution.
"""
from __future__ import annotations
import json
import os
import re
import socket
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict
from datetime import timedelta
from typing import Any, Callable
import pickle
import cloudpickle
from functools import partial
from collections.abc import Sequence

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.utils.network_utils import get_ip, get_loopback_ip, get_open_port, get_distributed_init_method
from vllm.utils.system_utils import get_mp_context
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config

from vllm.logger import logger
from vllm.v1.request import RequestStatus

from vllm_custom_plugins.plugins.zero_interrupt.common.constants import (
    VLLM_ITS_DECISION_CENTER_URL,
    VLLM_ITS_DECISION_CENTER_TOKEN,
    VLLM_ITS_ENABLE_FAULT_KEEP,
    VLLM_ITS_STRATEGY_TIMEOUT, VLLM_ITS_ENABLE_PD_REBUILD, VLLM_ITS_HTTP_SERVER_PORT_START, VLLM_SERVICE_ID,
    VLLM_ITS_HEALTH_CHECK_INTERVAL,
)
from vllm_custom_plugins.plugins.zero_interrupt.common.types import DeployState, DeployStrategy, DeployType, ExecutorState, EngineParallelConfig, \
    InitExecutorStateRequest, ModelInfo

from vllm_custom_plugins.plugins.zero_interrupt.common.communication.decision_center_client import DecisionCenterClient
from vllm_custom_plugins.plugins.zero_interrupt.vllm.v1.executor.utils import (
    BarrierPortPool,
    get_global_start_rank,
    get_heterogeneous_dp_config,
    get_surviving_dp_barrier_geometry,
    get_tp_asymmetric_shardings,
    is_heterogeneous_restart,
    recover_requires_full_restart,
)
from .health_monitor import ITSHealthMonitor, ITSFailureCallback
from .http_server import ITSHttpServer
from .strategy_sync import StrategySyncThread
from vllm_custom_plugins.plugins.zero_interrupt.common import StrategyHandler

from vllm_custom_plugins.plugins.zero_interrupt.vllm_ascend.utils import patch_direct_register_custom_op
patch_direct_register_custom_op()  # [h30014172] 防止 vllm_ascend.patch.worker 重复注册 unquantized_gemm operator

# Try to import AscendMultiprocExecutor, fallback to MultiprocExecutor
try:
    from vllm_ascend.patch.platform.patch_multiproc_executor import AscendMultiprocExecutor, AscendWorkerProc
except ImportError:
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor as AscendMultiprocExecutor
    from vllm.v1.executor.multiproc_executor import WorkerProc as AscendWorkerProc
from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle
from vllm.v1.executor.multiproc_executor import WorkerProc, FutureWrapper


def is_mm_scene() -> bool:
    """XDL_IP是ModelMate启动镜像时PredictManager镜像时独有的环境变量"""
    return True if os.getenv("XDL_IP") else False

def get_ip_mm() -> tuple(str, str):
    """
    适配ModelMate场景下的get_ip逻辑
    """
    if is_mm_scene():
        node_host_ip = os.getenv("XDL_IP", get_ip())
        pod_ip = os.getenv("THIS_POD_IP", get_ip())
    else:
        node_host_ip = get_ip()
        pod_ip = node_host_ip
    return node_host_ip, pod_ip

class ITSMultiprocExecutor(AscendMultiprocExecutor):
    """Intelligent Transform Service MultiprocExecutor.

    This executor provides:
    - Fault keep: Maintain service during worker failures
    - Strategy execution: Execute deployment strategies from decision center
    - State reporting: Report executor state to decision center
    - Smooth recovery: Recover service after deployment
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            monitor_workers: bool = True,
    ):
        """Initialize ITSMultiprocExecutor.

        Args:
            vllm_config: vLLM configuration
            monitor_workers: Enable worker monitoring
        """
        # Initialize signals/events before parent class init
        self.wait_new_deployment = threading.Event()
        self.recv_new_deployment = threading.Event()

        # Callback to trigger EngineCore busy_loop (set by engine_core_patch)
        self._trigger_busy_loop_callback: callable | None = None

        # Initialize state
        self.executor_state = ExecutorState.RUNNING
        self.current_strategy: DeployStrategy | None = None

        # Initialize ITS-specific components
        self._its_enabled = True

        # Executor id assigned by DecisionMakingCenter during
        # /init_executor_state. The center-generated id
        # (exe-<service>-<engine>-<n>) is used in every strategy payload and
        # deploy-status report; the local numeric data_parallel_rank remains
        # valid only for the old manual trigger scripts.
        self._decision_center_executor_id: str | None = None

        # Configuration
        self._fault_keep_enabled = VLLM_ITS_ENABLE_FAULT_KEEP
        self._http_port = VLLM_ITS_HTTP_SERVER_PORT_START

        # Full-restart barriers create a new stateless gloo group while the
        # current engine-core dp_group (and its TCPStore port) must stay
        # alive.  Reserve two ports from the shared DP port pool *before* the
        # parent class spawns workers: worker processes get a copy of
        # ``vllm_config`` and their ``get_next_dp_init_port()`` calls do not
        # propagate back, so popping the barrier ports here is the only point
        # where all DP executors can deterministically agree on the same
        # ports.  The pool rotates, so the port used by the previous barrier
        # is only reused after that group has been destroyed.
        self._its_barrier_port_pool = BarrierPortPool()
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            barrier_ports = [
                parallel_config.get_next_dp_init_port() for _ in range(2)
            ]
            self._its_barrier_port_pool = BarrierPortPool(barrier_ports)
            logger.info(
                "Reserved ITS full-restart barrier ports: %s",
                barrier_ports,
            )

        # Components (initialized in _init_executor)
        self._http_server: ITSHttpServer | None = None
        self._strategy_sync_thread: StrategySyncThread | None = None
        self._health_monitor: ITSHealthMonitor | None = None
        self._decision_center_client: DecisionCenterClient | None = None

        # Call parent class init
        super().__init__(vllm_config, monitor_workers)

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: "KVOutputAggregator" | None = None,
    ) -> Any:
        """
        对故障处理逻辑进行了需改，功能代码从vllm原生collective_rpc代码拷贝而来
        """
        assert self.rpc_broadcast_mq is not None, (
            f"collective_rpc should not be called on follower node, method={method}"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        if kv_output_aggregator is not None:
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                except Exception as exp:
                    # patched: 其它异常也不直接抛出，否则主进程会退出
                    # 目前写法，如果发生TimeoutError也会导致抛出异常导致程序退出
                    # 但目前开发自验过程没有出现类似问题，故目前暂不处理
                    logger.warning(f"[executor-collective_rpc] when calling method={method}, capture error but not raise. error={exp}")
                    status = WorkerProc.ResponseStatus.FAILURE

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    # patched: 不直接抛RuntimeError，否则主进程会退出
                    logger.warning("[executor-collective_rpc] request fail, return with None in collective_rpc()")
                    responses.append(None) # 发生故障时暂时返回None，交由上层处理故障
                else:
                    responses.append(result)
            return responses[0] if output_rank is not None else responses

        # v0.23 FutureWrapper semantics: __init__ takes (futures_queue,
        # get_response, aggregate), appends itself to the queue and drains
        # earlier futures inside result().  The v0.18 (future, get_response)
        # tuple queue no longer exists.
        future = FutureWrapper(
            self.futures_queue,
            get_response=get_response,
            aggregate=aggregate,
        )
        return future if non_block else future.result()


    def _init_executor(self) -> None:
        """Initialize the executor with ITS-specific components."""
        # Call parent class init first
        super()._init_executor()

        # Initialize decision center client (for reporting only)
        self._decision_center_client = DecisionCenterClient(
            base_url=VLLM_ITS_DECISION_CENTER_URL,
            token=VLLM_ITS_DECISION_CENTER_TOKEN,
        )

        # Initialize strategy sync thread FIRST (passive waiting mode)
        # This thread will passively wait for strategy events instead of polling
        self._strategy_sync_thread = StrategySyncThread(
            strategy_callback=self._on_strategy_sync,
            timeout=VLLM_ITS_STRATEGY_TIMEOUT,
        )
        self._strategy_sync_thread.start()

        # NOTE: AscendMultiprocExecutor._init_executor() already called
        # self.start_worker_monitor() through dynamic dispatch (the ITS
        # override).  Do NOT start a second ITSHealthMonitor thread here:
        # two monitors on the same worker list would both react to worker
        # death during restarts, and _cleanup_message_queues_and_workers
        # only stops the one stored in self._health_monitor.

        if is_mm_scene():
            # MM场景所有挂载卡visible
            _, mounted_npu_id_list = self.get_davinci_devices()
            ASCEND_RT_VISIBLE_DEVICES = os.getenv("ASCEND_RT_VISIBLE_DEVICES") 
            logger.info(f"ASCEND_RT_VISIBLE_DEVICES={ASCEND_RT_VISIBLE_DEVICES}")

            if ASCEND_RT_VISIBLE_DEVICES:
                sorted_mounted_npu_id_list = sorted(mounted_npu_id_list)
                # dp>1分支
                # 例子: MM+dp>1时，每个dp不一样: [0], [1], [2], [3]
                npu_id_list = []
                for rt_i in ASCEND_RT_VISIBLE_DEVICES.split(","):
                    npu_id_list.append(sorted_mounted_npu_id_list[int(rt_i)])
                logger.info(f"NPU IDs from environment: {npu_id_list}")
            else:
                # dp=1分支(e.g. tp4dp1) 保留原本逻辑
                npu_id_list = mounted_npu_id_list
                logger.info(f"NPU IDs from /dev/davinci*: {npu_id_list}")
        else:
            # 独立部署：--privilege + RT_VISIBLE
            # 从环境变量 ASCEND_RT_VISIBLE_DEVICES 或 /dev/davinci* 获取
            ASCEND_RT_VISIBLE_DEVICES = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
            if ASCEND_RT_VISIBLE_DEVICES:
                npu_id_list = ASCEND_RT_VISIBLE_DEVICES.split(",") # tp=1, dp>1
                logger.info(f"NPU IDs from environment: {npu_id_list}")
            else:
                _, npu_id_list = self.get_davinci_devices()  # tp>1, dp=1
                logger.info(f"NPU IDs from /dev/davinci*: {npu_id_list}")

        if npu_id_list:
            self._http_port = VLLM_ITS_HTTP_SERVER_PORT_START + int(npu_id_list[0])

        self._http_server = ITSHttpServer(
            port=self._http_port,
            strategy_sync_thread=self._strategy_sync_thread,
            expected_executor_id=str(self.parallel_config.data_parallel_rank),
        )
        self._http_server.start()
        logger.info(f"Waiting for deployment strategy via HTTP POST to port {self._http_port}")

        # Initialize HTTP server with strategy sync thread reference
        # The HTTP server will receive strategies via POST and notify the sync thread
        # Report initial state to decision center
        self._report_init_state(npu_id_list)

    def start_worker_monitor(self, inline=False) -> None:
        """Setup ITS-specific health monitor with fault keep."""
        failure_callback = ITSFailureCallback(self, fault_keep_enabled=self._fault_keep_enabled)

        self._health_monitor = ITSHealthMonitor(
            workers=self.workers,
            failure_callback=failure_callback,
            fault_keep_enabled=self._fault_keep_enabled,
        )
        self._health_monitor.start()


    def _get_node_type(self)->str:
        from vllm_ascend.utils import get_ascend_device_type, AscendDeviceType
        node_type = get_ascend_device_type()
        if node_type == AscendDeviceType.A2:
            return "A2"
        elif node_type == AscendDeviceType.A3:
            return "A3"
        else:
            return "Can't find device-type"

    def _full_mount(self) -> bool:
        """
        用于ASCEND_RT_VISIBLE_DEVICES设置计算
        """
        node_type = self._get_node_type()
        _, mounted_device_ids = self.get_davinci_devices()
        mounted_num_device = len(mounted_device_ids)
        return ((node_type == 'A3' and mounted_num_device == 16) or 
                (node_type == 'A2' and mounted_num_device == 8))

    # 上报给决策中心基础数据
    def _report_init_state(self, npu_id_list) -> None:
        """Report initial executor state to decision center.

        上报内容：
        - service_id: 服务实例唯一标识（从环境变量 VLLM_SERVICE_ID 获取）
        - engine_id: KV Cache Engine ID（用于 PD 分离场景）
        - engine_parallel_config: 并行配置（dp/tp/enable_expert_parallel）
        - engine_pd_role: PD 角色（从 kv_transfer_config.kv_role 获取）
        - executor_state: 执行器状态（RUNNING）
        - executor_ip_port: 执行器地址（data_parallel_address:data_parallel_rpc_port）
        - data_parallel_rank: DP 组内 rank
        - node_ip: 节点 IP
        - npu_id_list: NPU 物理 ID 列表
        - npu_rank_id: NPU 全局 rank 列表
        - npu_healthy: NPU 健康状态列表

        Returns:
            None
        """
        logger.info("Preparing to report init state to decision center")

        try:
            # 1. 构建并行配置
            # 从 vllm_config.parallel_config 获取当前并行配置
            engine_parallel_config = EngineParallelConfig(
                dp=self.parallel_config.data_parallel_size,
                tp=self.parallel_config.tensor_parallel_size,
                data_parallel_rank=self.parallel_config.data_parallel_rank,
                enable_expert_parallel=self.parallel_config.enable_expert_parallel
            )
            logger.debug(f"Engine parallel config: dp={engine_parallel_config.dp}, "
                         f"tp={engine_parallel_config.tp}, "
                         f"enable_expert_parallel={engine_parallel_config.enable_expert_parallel}")

            # 2. 获取 PD 角色
            # DecisionMakingCenter 的 StrategyOptimizer 只把
            # EXECUTOR_PD_ROLE_MAP == "P_ROLE" 的 executor 纳入 P-Engine 寻优，
            # 而 vLLM kv_transfer_config 里的值是 kv_producer/kv_consumer。
            # 这里转换为决策中心约定的 P_ROLE/D_ROLE，避免上报后寻优把
            # prefill executor 全部过滤掉。
            engine_pd_role = ""
            kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
            if kv_transfer_config and hasattr(kv_transfer_config, "kv_role"):
                kv_role = kv_transfer_config.kv_role
                if kv_role == "kv_producer":
                    engine_pd_role = "P_ROLE"
                elif kv_role == "kv_consumer":
                    engine_pd_role = "D_ROLE"
                else:
                    engine_pd_role = kv_role
            logger.debug(f"Instance PD role: {engine_pd_role}")

            # 3. 构建执行器地址
            # 从 vllm_config 获取 data_parallel_address 和 data_parallel_rpc_port
            master_addr = getattr(self.parallel_config, "data_parallel_master_ip", None)
            master_port = getattr(self.parallel_config, "data_parallel_rpc_port", None)
            if master_addr and master_port:
                data_parallel_ip_port = f"{master_addr}:{master_port}"
            else:
                data_parallel_ip_port = ""
            logger.debug(f"DATA PARALLEL IP:port: {data_parallel_ip_port}")

            # 4. 获取 engine_id
            engine_id = "0"
            if self.vllm_config.kv_transfer_config:
                engine_id = self.vllm_config.kv_transfer_config.engine_id
            if not self.parallel_config.is_moe_model:
                engine_id = f"{self.parallel_config.data_parallel_index}_{engine_id}"
            logger.debug(f"Engine ID: {engine_id}")
            # 5. 获取节点 IP
            node_ip, pod_ip = get_ip_mm()
            logger.debug(f"Node IP: {node_ip}")

            # 6. 获取 NPU 信息

            npu_rank_id = []
            npu_healthy = []

            if hasattr(self, "workers") and self.workers:
                # 默认所有 NPU 都是健康的
                npu_healthy = [True] * len(npu_id_list)

                tensor_parallel_size = self.parallel_config.tensor_parallel_size
                data_parallel_rank = self.parallel_config.data_parallel_rank
                npu_rank_id = [data_parallel_rank * tensor_parallel_size + i for i in range(tensor_parallel_size)]
                logger.debug(f"NPU rank IDs: {npu_rank_id}")

            # 6 获取 executor_ip_port
            executor_ip_port = f"{pod_ip}:{self._http_port}"
            # 7. 获取 service_id
            # 从环境变量 VLLM_SERVICE_ID 获取，生成 UUID 作为默认值
            service_id = VLLM_SERVICE_ID
            logger.info(f"Service ID: {service_id}")

            # 8. 获取模型信息
            model_info = self._get_model_info()

            node_hbm = self._get_npu_memory_size()
            # 9. 构建请求并上报
            self.init_executor_state_request = InitExecutorStateRequest(
                service_id=VLLM_SERVICE_ID,
                model_name=self.vllm_config.model_config.served_model_name,
                engine_id=engine_id,
                engine_parallel_config=engine_parallel_config,
                engine_pd_role=engine_pd_role,
                executor_state=ExecutorState.RUNNING.value,
                executor_ip_port=executor_ip_port,
                data_parallel_ip_port=data_parallel_ip_port,
                data_parallel_rank=self.parallel_config.data_parallel_rank,
                node_ip=node_ip,
                npu_id=npu_id_list,
                model_info=model_info,
                node_hbm=node_hbm,
                node_type=self._get_node_type(),
                npu_rank_id=npu_rank_id,
                npu_healthy=npu_healthy
            )
            logger.info(f"##########vllm_config: {self.vllm_config}")

            logger.info("Reporting init state to decision center...")
            assigned_executor_id = (
                self._decision_center_client.report_init_state_with_executor_id(
                    self.init_executor_state_request
                )
            )

            if assigned_executor_id:
                self._decision_center_executor_id = assigned_executor_id
                if self._http_server is not None:
                    self._http_server.add_expected_executor_id(
                        assigned_executor_id
                    )
                logger.info(
                    "Successfully reported init state to decision center: "
                    "assigned executor_id=%s; payload=%s",
                    assigned_executor_id,
                    asdict(self.init_executor_state_request),
                )
            else:
                logger.warning(
                    "Failed to report init state to decision center (no "
                    "executor_id returned): %s",
                    asdict(self.init_executor_state_request),
                )

        except Exception as e:
            logger.error(f"Error reporting init state: {e}", exc_info=True)

    def _load_model_info_from_config_json(self, model_path: str) -> dict:
        """从模型路径 config.json 加载模型配置。

        Args:
            model_path: 模型路径或目录

        Returns:
            dict: config.json 中的模型配置
        """
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.warning(f"config.json not found at {config_path}")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json from {config_path}: {e}")
            return {}

    def _get_model_info(self) -> ModelInfo:
        """从 vllm_config 或模型路径 config.json 获取模型信息。

        优先从 vllm_config.model_config.hf_config 获取字段，
        如果字段不存在或为 0/None，则从模型路径 config.json 读取。

        Returns:
            ModelInfo: 填充了实际数据的 ModelInfo 对象
        """
        model_info = ModelInfo()
        model_config = self.vllm_config.model_config
        hf_config = getattr(model_config, "hf_config", None)

        # 从 hf_config 获取字段
        # ========== 修改：增加对 text_config 的 fallback ==========
        def get_hf_attr(name: str, default: Any = None) -> Any:
            """从 hf_config 获取属性，如果不存在或为 0/None 则尝试 text_config。"""
            if hf_config is None:
                return default
            # 尝试从 hf_config 直接获取
            value = getattr(hf_config, name, None)
            if value is not None and value != 0:
                return value
            # 如果存在 text_config，尝试从 text_config 获取
            text_config = getattr(hf_config, "text_config", None)
            if text_config is not None:
                value = getattr(text_config, name, None)
                if value is not None and value != 0:
                    return value
            return default

        # ========================================================

        # 获取 model_path 用于 fallback
        model_path = getattr(model_config, "model", None)
        config_json = {}
        if model_path:
            config_json = self._load_model_info_from_config_json(model_path)

        def get_config_json_attr(path: list[str], default: Any = None) -> Any:
            """从 config_json 获取嵌套属性。"""
            value = config_json
            for key in path:
                if not isinstance(value, dict):
                    return default
                value = value.get(key, None)
                if value is None:
                    return default
            return value

        # 填充 ModelInfo 字段
        try:
            model_info.hidden_size = get_hf_attr("hidden_size", 0) or \
                                     get_config_json_attr(["hidden_size"], 0)
        except Exception:
            model_info.hidden_size = 0

        try:
            model_info.num_attention_heads = get_hf_attr("num_attention_heads", 0) or \
                                             get_config_json_attr(["num_attention_heads"], 0)
        except Exception:
            model_info.num_attention_heads = 0

        try:
            model_info.num_layers = get_hf_attr("num_hidden_layers", 0) or \
                                    get_config_json_attr(["num_hidden_layers"], 0)
        except Exception:
            model_info.num_layers = 0

        try:
            model_info.expert_num = get_hf_attr("n_routed_experts", 0) or \
                                    get_config_json_attr(["n_routed_experts"], 0)
            if model_info.expert_num == 0:
                model_info.expert_num = get_hf_attr("num_experts", 0) or \
                                        get_config_json_attr(["num_experts"], 0)
        except Exception:
            model_info.expert_num = 0

        try:
            model_info.moe_intermediate_size = get_hf_attr("moe_intermediate_size", 0) or \
                                               get_config_json_attr(["moe_intermediate_size"], 0)
        except Exception:
            model_info.moe_intermediate_size = 0

        try:
            model_info.intermediate_size = get_hf_attr("intermediate_size", 0) or \
                                           get_config_json_attr(["intermediate_size"], 0)
        except Exception:
            model_info.intermediate_size = 0

        try:
            architectures = get_hf_attr("architectures", None)
            if not architectures:
                architectures = get_config_json_attr(["architectures", 0], None)

            if isinstance(architectures, list):
                model_info.architectures = architectures[0]
        except Exception:
            model_info.architectures = None

        try:
            model_info.vocab_size = get_hf_attr("vocab_size", 0) or \
                                    get_config_json_attr(["vocab_size"], 0)
        except Exception:
            model_info.vocab_size = 0

        try:
            model_info.num_key_value_heads = get_hf_attr("num_key_value_heads", 0) or \
                                             get_config_json_attr(["num_key_value_heads"], 0)
        except Exception:
            model_info.num_key_value_heads = 0

        try:
            model_info.tie_word_embeddings = get_hf_attr("tie_word_embeddings", False) or \
                                             get_config_json_attr(["tie_word_embeddings"], False)
        except Exception:
            model_info.tie_word_embeddings = False

        model_info.max_model_len = getattr(model_config, "max_model_len", 0) or \
                                   get_config_json_attr(["max_position_embeddings"], 0)

        model_info.kv_quantize = getattr(model_config, "mla_quantize", None)
        if not model_info.kv_quantize:
            kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
            if kv_transfer_config and hasattr(kv_transfer_config, "kv_scale_dtype"):
                model_info.kv_quantize = str(kv_transfer_config.kv_scale_dtype)

        model_info.weight_quantize = getattr(model_config, "quantization", None)

        logger.debug(f"Model info retrieved: hidden_size={model_info.hidden_size}, "
                     f"num_layers={model_info.num_layers}, "
                     f"num_attention_heads={model_info.num_attention_heads}, "
                     f"expert_num={model_info.expert_num}, "
                     f"moe_intermediate_size={model_info.moe_intermediate_size}, "
                     f"intermediate_size={model_info.intermediate_size}, "
                     f"architectures={model_info.architectures}, "
                     f"vocab_size={model_info.vocab_size}, "
                     f"num_key_value_heads={model_info.num_key_value_heads}, "
                     f"tie_word_embeddings={model_info.tie_word_embeddings}, "
                     f"max_model_len={model_info.max_model_len}, "
                     f"kv_quantize={model_info.kv_quantize}, "
                     f"weight_quantize={model_info.weight_quantize}")

        return model_info

    def _on_deploy_strategy_received(self, strategy: DeployStrategy) -> None:
        """Callback when deployment strategy is received via HTTP.

        Args:
            strategy: Deployment strategy
        """
        logger.info(f"Received deployment strategy: {strategy.deploy_type.value}")
        self.current_strategy = strategy

        # Signal deployment event
        self.recv_new_deployment.set()

        # Early return if no callback to process
        if not self._trigger_busy_loop_callback:
            self.executor_state = ExecutorState.EXECUTING_STRATEGY
            return

        # Process busy_loop callback: pause and trigger
        try:
            # Pause busy_loop inference before triggering to prevent new inference
            if hasattr(self, '_engine_core_ref') and self._engine_core_ref:
                self._engine_core_ref._paused_for_restart = True
                logger.info("Paused busy_loop inference before strategy execution")

            # Trigger busy_loop to process the strategy
            self._trigger_busy_loop_callback()
            logger.info("Deployment strategy received: triggered busy_loop")
        except Exception as e:
            logger.error(f"Failed to process strategy: {e}")

        self.executor_state = ExecutorState.EXECUTING_STRATEGY

    def _on_strategy_sync(self, strategy: DeployStrategy) -> None:
        """Callback when strategy is received via polling.

        Args:
            strategy: Deployment strategy
        """
        logger.info(f"Strategy sync received: {strategy.deploy_type.value}")
        self._on_deploy_strategy_received(strategy)

    def execute_deploy_strategy(self, strategy: DeployStrategy) -> bool:
        """Execute deployment strategy.

        Strategy is passed to workers via VllmConfig.additional_config
        instead of RPC. Workers will read config at startup.

        Args:
            strategy: Deployment strategy to execute

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing strategy: {strategy.deploy_type.value}")

        try:

            if strategy.deploy_type == DeployType.STOP:
                self.shutdown()
                return True
            elif strategy.deploy_type == DeployType.DEGRADE:
                return self._execute_degrade_strategy(strategy)
            elif strategy.deploy_type == DeployType.RECOVER:
                return self._execute_recover_strategy(strategy)
            elif strategy.deploy_type == DeployType.PD_REBUILD:
                return self._execute_pd_rebuild_strategy(strategy)
            else:
                logger.error(f"Unknown deploy type: {strategy.deploy_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_FAIL)
            return False

    def handle_new_deployment(self) -> bool:
        """Handle new deployment strategy execution.

        Called from EngineCore patch when RecvNewDeployment signal is received.
        This method executes the current deployment strategy.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Handling new deployment")

        if not hasattr(self, "current_strategy") or not self.current_strategy:
            logger.warning("No deployment strategy available")
            return False

        try:
            # Execute the strategy
            success = self.execute_deploy_strategy(self.current_strategy)

            if success:
                # dp=0 场景：空转状态，不是 RUNNING
                if self.world_size == 0:
                    self.executor_state = ExecutorState.STOPPED
                    logger.info("Deployment strategy executed successfully, but executor is in idle mode (dp=0)")
                else:
                    self.executor_state = ExecutorState.RUNNING
                    logger.info("Deployment strategy executed successfully")
            else:
                logger.error("Deployment strategy execution failed")
                self.executor_state = ExecutorState.EXECUTING_STRATEGY_FAILED

            return success

        except Exception as e:
            logger.error(f"Error handling new deployment: {e}")
            return False

    def restart_workers_with_strategy(self) -> bool:
        """Restart workers with current strategy.

        Design (from DESIGN.md):
        - Timeout: restart workers with current strategy in VllmConfig
        - Use VllmConfig.additional_config["zero_interrupt_config"]

        Returns:
            True if successful, False otherwise
        """
        logger.info("Restarting workers with current strategy")

        try:
            self.executor_state = ExecutorState.RECOVERING
            self._cleanup_and_restart_workers()

            # dp=0 场景：空转状态，不是 RUNNING
            if self.world_size == 0:
                self.executor_state = ExecutorState.STOPPED
                logger.info("Workers restarted with dp=0, executor in idle mode")
            else:
                self.executor_state = ExecutorState.RUNNING
                logger.info("Workers restarted successfully with strategy")
            return True

        except Exception as e:
            logger.error(f"Error restarting workers: {e}")
            return False

    def _destroy_stateless_dp_group(self, dp_group) -> None:
        """Best-effort destroy of a stateless gloo DP group."""
        if dp_group is None:
            return
        try:
            stateless_destroy_torch_distributed_process_group(dp_group)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to destroy stateless DP group: %s", exc
            )

    def _next_barrier_master_port(self) -> int:
        """Advance the barrier-port generation and return the port to use.

        All DP executors (including scale-to-zero executors that skip the
        barrier) must call this exactly once per full-restart generation so
        that a later RECOVER strategy picks the same port on every executor.
        """
        port = self._its_barrier_port_pool.next_port()
        logger.info(
            "Full-restart barrier: using reserved barrier master port %d.",
            port,
        )
        return port

    def _build_surviving_dp_group(
        self,
        surviving_dp_size: int,
        surviving_dp_rank: int,
        barrier_master_port: int,
    ):
        """Create a stateless gloo group over the surviving DP executors.

        The barrier port is taken from a pool reserved before the first
        worker was spawned.  The pre-restart ``parallel_config`` is not
        mutated, so it is still available to
        ``_try_backup_origin_parallel_config_when_degrade`` for a later
        RECOVER strategy.

        ``stateless_init_dp_group()`` pops a port from the shared DP port
        list itself, but that list is stale in the EngineCore process:
        workers pop ports on their private ``VllmConfig`` copies, so the
        parent's next entry is the port still bound by the live worker
        distributed group.  On top of that only rank 0 sees EADDRINUSE and
        retries, while ranks > 0 stay on the old port.  Using an explicit
        reserved port with a pre-bound listen socket avoids both problems.
        """
        host = self.parallel_config.data_parallel_master_ip
        listen_socket: socket.socket | None = None
        if int(surviving_dp_rank) == 0:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listen_socket.bind((host, int(barrier_master_port)))
                listen_socket.listen()
            except Exception:
                listen_socket.close()
                raise
        logger.info(
            "Full-restart barrier: building target dp_group "
            "world_size=%d rank=%d port=%d.",
            surviving_dp_size,
            surviving_dp_rank,
            barrier_master_port,
        )
        return stateless_init_torch_distributed_process_group(
            host,
            int(barrier_master_port),
            int(surviving_dp_rank),
            int(surviving_dp_size),
            backend="gloo",
            return_store=True,
            listen_socket=listen_socket,
        )

    def _mark_scale_to_zero_dp_excluded(self) -> None:
        """Stop the idle (scale-to-zero) EngineCore from using the old dp_group.

        Once the surviving executors have replaced the pre-restart dp_group
        with a renumbered group, the old group has no peers left for this
        executor. Returning from ``_has_global_unfinished_reqs`` without an
        all_reduce prevents the idle loop from blocking on a 10s timeout at
        every DP sync.
        """
        engine_core = getattr(self, "_engine_core_ref", None)
        if engine_core is None:
            return
        setattr(engine_core, "_its_dp_sync_excluded", True)
        logger.info(
            "Executor marked as scale-to-zero; EngineCore will skip the "
            "cross-executor DP state sync."
        )

    def _adopt_surviving_dp_group(
        self,
        engine_core,
        barrier_group,
        barrier_store,
        surviving_dp_size: int,
        surviving_dp_rank: int,
    ) -> None:
        """Swap the EngineCore dp_group to the surviving-rank group.

        The barrier guarantees that every surviving executor has left the old
        dp_group collectives, so the old 16-rank group can be destroyed
        locally without waiting for the faulty executor. Future
        ``sync_dp_state`` calls then run on the new 15-rank group and no
        longer depend on the scale-to-zero executor at all.
        """
        old_group = getattr(engine_core, "dp_group", None)
        old_size = None
        if old_group is not None and old_group is not barrier_group:
            try:
                old_size = int(old_group.size())
            except Exception:  # noqa: BLE001
                old_size = "?"
        engine_core.dp_group = barrier_group
        engine_core.dp_store = barrier_store
        engine_core.dp_size = int(surviving_dp_size)
        engine_core.dp_rank = int(surviving_dp_rank)
        setattr(engine_core, "_its_dp_sync_excluded", False)
        if old_group is not None and old_group is not barrier_group:
            self._destroy_stateless_dp_group(old_group)
            logger.info(
                "Adopted surviving dp_group world_size=%d rank=%d and "
                "destroyed the pre-restart dp_group (old world_size=%s).",
                surviving_dp_size,
                surviving_dp_rank,
                old_size,
            )

    def _barrier_for_full_restart(
        self, timeout_seconds: int = VLLM_ITS_STRATEGY_TIMEOUT
    ) -> None:
        """Barrier across DP executor processes before a full restart.

        DP4TP4 -> DP4TP(3,4,4,4) changes the global worker world from 16 to
        15 ranks and rebuilds every MoE/HCCL communication group.  If one
        executor killed its workers while the others were still serving from
        the old 16-rank world, the old collectives on the healthy executors
        can fail in an uncontrolled way and the new 15-rank
        ``init_process_group`` can hang forever waiting for ranks that never
        join.

        For a decode DP16TP1 -> DP15TP1 restart, the executor whose only NPU
        failed can be stuck in a long NPU task timeout.  Instead of waiting
        for it to reach the rendezvous, the surviving executors build a new
        stateless gloo group with the renumbered topology (world_size=15,
        ranks 0..14) and barrier among themselves. The scale-to-zero
        executor skips the barrier and cleans up independently; the new group
        then replaces ``engine_core.dp_group`` so post-restart DP state sync
        no longer depends on the faulty executor either.
        """
        strategy = getattr(self, "current_strategy", None)
        surviving_dp_size = None
        surviving_dp_rank = None
        scale_to_zero_ranks: set[int] = set()
        if strategy is not None:
            try:
                strategy_dict = self._convert_enums_to_values(
                    asdict(strategy)
                )
                (
                    surviving_dp_size,
                    surviving_dp_rank,
                    scale_to_zero_ranks,
                ) = get_surviving_dp_barrier_geometry(strategy_dict)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to compute surviving-DP barrier geometry for "
                    "full restart: %s",
                    exc,
                )
                raise

            if surviving_dp_size == 0:
                # Advance the barrier port generation even though this
                # executor does not join the rendezvous.  Healthy executors
                # advance once when they build the survivor group; if this
                # executor skipped that advance, a later RECOVER would pick
                # a different barrier port and deadlock.
                self._next_barrier_master_port()
                logger.info(
                    "Full-restart barrier skipped: executor_id=%s is "
                    "scale-to-zero and cleans up independently.",
                    strategy.executor_id,
                )
                self._mark_scale_to_zero_dp_excluded()
                return

        engine_core = getattr(self, "_engine_core_ref", None)
        barrier_group = (
            getattr(engine_core, "dp_group", None) if engine_core else None
        )
        barrier_store = (
            getattr(engine_core, "dp_store", None) if engine_core else None
        )
        temporary_group = False

        # The engine-core dp_group must cover exactly the target topology.
        # - DP16 -> DP15: the old group still contains the faulty executor.
        # - DP15 -> DP16 (RECOVER): healthy executors only have a 15-rank
        #   group, while the previously scale-to-zero executor has a stale
        #   16-rank group (and is marked _its_dp_sync_excluded). A new group
        #   including all 16 target ranks must be created before any executor
        #   kills its workers.
        # - Heterogeneous TP recovery (DP4TP(3,4,4,4) -> DP4TP4) keeps the
        #   same four engine-core processes, so the existing 4-rank group is
        #   reused.
        current_dp_size = getattr(
            self.parallel_config, "data_parallel_size", None
        )
        old_dp_group_size = None
        if barrier_group is not None:
            try:
                old_dp_group_size = int(barrier_group.size())
            except Exception:  # noqa: BLE001
                old_dp_group_size = None
        dp_sync_excluded = bool(
            getattr(engine_core, "_its_dp_sync_excluded", False)
            if engine_core is not None else False
        )
        needs_target_dp_group = bool(scale_to_zero_ranks)
        if surviving_dp_size is not None and (
            dp_sync_excluded
            or int(current_dp_size or 0) != int(surviving_dp_size)
            or (
                old_dp_group_size is not None
                and int(old_dp_group_size) != int(surviving_dp_size)
            )
        ):
            needs_target_dp_group = True

        if needs_target_dp_group:
            if scale_to_zero_ranks:
                logger.info(
                    "Full-restart barrier: excluding scale-to-zero executor "
                    "ranks %s.",
                    sorted(scale_to_zero_ranks),
                )
            else:
                logger.info(
                    "Full-restart barrier: current dp_group topology "
                    "(current_dp=%s, group_size=%s) differs from target "
                    "dp=%s, rebuilding.",
                    current_dp_size,
                    old_dp_group_size,
                    surviving_dp_size,
                )
            # This must happen before `_get_engine_parallel_config` mutates
            # parallel_config.  The reserved port rotates once per barrier
            # generation, so the previous barrier's still-live dp_group never
            # occupies the port selected here.
            barrier_master_port = self._next_barrier_master_port()
            barrier_group, barrier_store = self._build_surviving_dp_group(
                surviving_dp_size,
                surviving_dp_rank,
                barrier_master_port,
            )
            temporary_group = True

        if barrier_group is None:
            logger.info(
                "Full-restart barrier skipped: no cross-executor DP group."
            )
            return

        def _discard_temporary_group() -> None:
            if temporary_group and barrier_group is not None:
                self._destroy_stateless_dp_group(barrier_group)

        try:
            world_size = int(barrier_group.size())
            rank_in_dp = int(barrier_group.rank())
        except Exception as exc:  # defensive: non-torch/stateless group
            logger.warning(
                "Full-restart barrier skipped: cannot query barrier group "
                "size/rank: %s",
                exc,
            )
            _discard_temporary_group()
            return

        if world_size < 2:
            # A single surviving executor has no rendezvous to wait for, but
            # its EngineCore dp_group still has to be replaced so the idle
            # scale-to-zero executors are no longer referenced.
            if temporary_group and engine_core is not None:
                self._adopt_surviving_dp_group(
                    engine_core,
                    barrier_group,
                    barrier_store,
                    surviving_dp_size,
                    surviving_dp_rank,
                )
            else:
                _discard_temporary_group()
            logger.info(
                "Full-restart barrier skipped: cross-executor DP group "
                "world_size=%d.",
                world_size,
            )
            return

        logger.info(
            "Full-restart barrier: waiting for %d/%d DP executors to receive "
            "the strategy before worker cleanup.",
            rank_in_dp + 1,
            world_size,
        )
        # Use the exact same collective shape/op as
        # ParallelConfig.sync_dp_state (two int32 SUM). On the old group this
        # makes the barrier interchangeable with peers still inside
        # `_has_global_unfinished_reqs`; on the temporary survivor group it
        # is a plain deterministic rendezvous.
        tensor = torch.tensor([0, 0], dtype=torch.int32)
        work = torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=barrier_group,
            async_op=True,
        )
        completed = False
        try:
            completed = bool(
                work.wait(timeout=timedelta(seconds=timeout_seconds))
            )
        except Exception as exc:  # defensive: older torch raises on timeout
            logger.warning(
                "Full-restart barrier wait raised: %s", exc
            )
        if not completed:
            try:
                work.abort()
            except Exception as abort_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to abort timed-out barrier all_reduce: %s",
                    abort_exc,
                )
            _discard_temporary_group()
            raise RuntimeError(
                "Heterogeneous full restart barrier timed out after "
                f"{timeout_seconds}s waiting for all {world_size} DP "
                "executors. The decision center must push the same "
                "engine_parallel_config to EVERY surviving DP executor "
                "(topology changes rebuild MoE weights and global "
                "communication groups, so restarting only the faulty DP is "
                "invalid)."
            )

        # Every surviving executor has now left the old dp_group. Adopt the
        # temporary survivor group as the new EngineCore dp_group and retire
        # the old one, so neither the restart nor post-restart DP sync needs
        # the scale-to-zero executor.
        if temporary_group and engine_core is not None:
            self._adopt_surviving_dp_group(
                engine_core,
                barrier_group,
                barrier_store,
                surviving_dp_size,
                surviving_dp_rank,
            )
        logger.info(
            "Full-restart barrier passed: all %d DP executors will now "
            "restart their workers together.",
            world_size,
        )

    def _cleanup_and_restart_workers(self) -> None:
        """Restart all workers with current VllmConfig.

        The VllmConfig already contains the strategy in
        additional_config["zero_interrupt_config"].
        """
        logger.info("Restarting all workers with current VllmConfig")

        try:
            # Heterogeneous TP (e.g. DP4TP4 -> DP4TP(3,4,4,4)) rebuilds the
            # global worker world and all MoE communication groups.  Before
            # killing any worker, rendezvous with the other DP executors:
            # restarting only the faulty DP would leave the healthy DPs in the
            # old 16-rank world and the new 15-rank init_process_group would
            # wait forever.
            if (
                self.current_strategy is not None
                and not getattr(self, "shutting_down", False)
                and self._strategy_requires_full_restart(self.current_strategy)
            ):
                self._barrier_for_full_restart()
                logger.info(
                    "Heterogeneous TP restart: restarting workers of EVERY "
                    "DP instance."
                )

            # Step 1: Clean up all requests in scheduler before worker restart
            self._cleanup_scheduler_requests()

            # Step 2: Clear old message queues and worker references
            self._cleanup_message_queues_and_workers()


            # Step 3: Update VllmConfig with strategy
            #if self._is_asymmetric():
            self._update_vllm_config_for_restart()

            # Step 5: Re-initialize workers with new config
            self._init_workers()

            # dp=0/tp=0 场景：空转模式，跳过后续 worker 相关步骤
            if self.world_size == 0:
                time.sleep(VLLM_ITS_HEALTH_CHECK_INTERVAL)
                self.wait_new_deployment.clear()
                logger.info("Idle mode (dp=0), skipping worker-dependent steps")
                return

            # Step 6: Reset health monitor failure state, update workers reference, and restart
            # Note: health monitor was stopped in Step 2, need to restart it
            if hasattr(self, '_health_monitor') and self._health_monitor:
                self._health_monitor.reset_failure_state()
                self._health_monitor.update_workers(self.workers)
                self._health_monitor.start()
                self.wait_new_deployment.clear()

            # Step 8: Re-initialize KV cache after worker restart
            # This fixes the kv_cache_config issue in vllm_ascend after worker restart
            self._reinitialize_kv_cache()

            logger.info("All workers restarted successfully")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in _restart_all_workers: {e}")
            raise

    def _is_asymmetric(self):
        model_info = self._get_model_info()
        from dataclasses import asdict
        strategy_dict = asdict(self.current_strategy)
        # Convert enum values to strings
        strategy_dict = self._convert_enums_to_values(strategy_dict)
        logger.info(f"model_info: {model_info}, strategy_dict: {strategy_dict}")
        executor_id = strategy_dict.get("executor_id")
        engine_parallel_config = strategy_dict.get("engine_parallel_config", [])

        new_tp = next(
            (item.get("new_tp") for item in engine_parallel_config if item.get("executor_id") == executor_id),
            None
        )
        return new_tp>1 and model_info.num_key_value_heads % new_tp != 0

    def _reinitialize_kv_cache(self) -> None:
        """Re-initialize KV cache after worker restart.

        This fixes the kv_cache_config issue where NPUModelRunner
        is missing the kv_cache_config attribute after worker restart.
        The issue occurs because initialize_from_config is not called
        when workers are restarted.
        """
        logger.info("Re-initializing KV cache after worker restart")

        try:
            # Step 1: Get KV cache specs from workers
            kv_cache_specs = self.get_kv_cache_specs()
            logger.info(f"Got kv_cache_specs: {len(kv_cache_specs)} workers")
            # [mzm] Add detailed logging for debugging KV cache spec issues
            for i, spec in enumerate(kv_cache_specs):
                logger.debug(f"[mzm] kv_cache_specs[{i}]: type={type(spec).__name__}, "
                           f"is_None={spec is None}, "
                           f"len={len(spec) if spec is not None else 'N/A'}")
                if spec is not None:
                    for layer_name, layer_spec in spec.items():
                        logger.debug(f"[mzm]   layer: {layer_name}, spec_type: {type(layer_spec).__name__}")

            # Step 2: Determine available memory (same logic as original _initialize_kv_caches)
            available_memory = self.determine_available_memory()
            logger.info(f"Determined available_memory: {available_memory}")

            # collective_rpc is fault-tolerant and returns None for failed
            # workers (typically profile_run raised on the worker side).
            # Passing None into get_kv_cache_configs produces a misleading
            # ``NoneType <= int`` TypeError deep inside kv_cache_utils, so
            # fail here with the worker logs as the actionable error source.
            if available_memory is None or (
                isinstance(available_memory, (list, tuple))
                and any(memory is None for memory in available_memory)
            ):
                raise RuntimeError(
                    "determine_available_memory returned "
                    f"{available_memory!r}; one or more workers failed "
                    "profile_run. Check the worker tracebacks above."
                )

            # Step 3: Get KV cache configs
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config,
                kv_cache_specs,
                available_memory
            )
            logger.info(f"Generated kv_cache_configs: {len(kv_cache_configs)} configs")

            # Step 4: Generate scheduler KV cache config and update vllm_config
            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
            self.vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
            kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
            if kv_cache_groups:
                self.vllm_config.cache_config.block_size = min(
                    g.kv_cache_spec.block_size for g in kv_cache_groups
                )
            self.vllm_config.validate_block_size()

            # Step 4.5: Rebuild the scheduler-side KVCacheManager from the
            # fresh config.  The scheduler was created before the restart and
            # still holds the old block pool / prefix-cache mappings; reusing
            # it after a worker restart can allocate stale block ids whose
            # contents belong to earlier requests ("coherent but unrelated"
            # completions).
            self._rebuild_scheduler_kv_cache_manager(scheduler_kv_cache_config)

            # Step 5: Initialize workers with KV cache config
            self.initialize_from_config(kv_cache_configs)
            logger.info("KV cache re-initialized successfully")

        except Exception as e:
            logger.error(f"Failed to re-initialize KV cache: {e}")
            raise Exception(f"KV cache re-initialization failed: {e}")

    def _rebuild_scheduler_kv_cache_manager(self, kv_cache_config) -> None:
        """Replace the scheduler KVCacheManager after a worker restart.

        The scheduler process survives a worker restart, so its
        ``kv_cache_manager`` still references the pre-restart block pool.
        Rebuild it from the freshly profiled ``KVCacheConfig`` and rebind the
        scheduler-side KV connector to the new block pool.  Without this step,
        a request scheduled after restart can receive block ids whose memory
        was last written by a different request.
        """
        engine_core = getattr(self, "_engine_core_ref", None)
        scheduler = getattr(engine_core, "scheduler", None)
        if scheduler is None:
            logger.warning(
                "Scheduler KVCacheManager not rebuilt: engine_core scheduler "
                "is unavailable."
            )
            return

        from vllm.v1.core.kv_cache_manager import KVCacheManager

        old_manager = getattr(scheduler, "kv_cache_manager", None)
        hash_block_size = getattr(scheduler, "block_size", None)
        new_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=scheduler.max_model_len,
            scheduler_block_size=scheduler.block_size,
            hash_block_size=hash_block_size or scheduler.block_size,
            max_num_batched_tokens=(
                scheduler.scheduler_config.max_num_batched_tokens
            ),
            enable_caching=scheduler.cache_config.enable_prefix_caching,
            use_eagle=scheduler.use_eagle,
            log_stats=scheduler.log_stats,
            enable_kv_cache_events=scheduler.enable_kv_cache_events,
            dcp_world_size=scheduler.dcp_world_size,
            pcp_world_size=scheduler.pcp_world_size,
            metrics_collector=scheduler.kv_metrics_collector,
        )
        scheduler.kv_cache_config = kv_cache_config
        scheduler.kv_cache_manager = new_manager
        scheduler.has_mamba_layers = bool(
            getattr(kv_cache_config, "has_mamba_layers", False)
        )
        scheduler.needs_kv_cache_zeroing = bool(
            getattr(kv_cache_config, "needs_kv_cache_zeroing", False)
        )

        connector = getattr(scheduler, "connector", None)
        if connector is not None:
            try:
                connector.bind_gpu_block_pool(new_manager.block_pool)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to rebind scheduler KV connector block pool: %s",
                    exc,
                )

        logger.info(
            "Scheduler KVCacheManager rebuilt: num_blocks=%s groups=%s "
            "(old manager was %s)",
            getattr(kv_cache_config, "num_blocks", None),
            len(getattr(kv_cache_config, "kv_cache_groups", []) or []),
            type(old_manager).__name__ if old_manager is not None else None,
        )

    def update_kv_connector_metadata(self, engine_core) -> None:
        """Update KV connector handshake metadata after worker restart.

        This is called after PD_REBUILD strategy to refresh the KV connector
        metadata in the scheduler with the new worker information.

        Args:
            engine_core: The EngineCore instance
        """
        logger.info("Updating KV connector metadata after worker restart")

        try:
            # Get KV connector from scheduler
            kv_connector = engine_core.scheduler.get_kv_connector()
            if kv_connector is None:
                logger.debug("No KV connector found in scheduler")
                return

            # Get metadata from workers
            xfer_handshake_metadata = self.get_kv_connector_handshake_metadata()

            if not xfer_handshake_metadata:
                logger.warning("No KV connector handshake metadata from workers")
                return

            # Merge all worker dicts into a single dict
            content: dict = {}
            for worker_dict in xfer_handshake_metadata:
                if worker_dict is not None:
                    content.update(worker_dict)

            # 重启后旧 worker 的端口/engine_id 全部失效，先清掉 scheduler
            # connector 里缓存的旧条目，避免脏 mapping 残留。
            connector_scheduler = getattr(
                kv_connector, "connector_scheduler", None
            )
            if connector_scheduler is not None and hasattr(
                connector_scheduler, "multi_nodes_meta_mapping"
            ):
                connector_scheduler.multi_nodes_meta_mapping.clear()

            # Set metadata to KV connector
            kv_connector.set_xfer_handshake_metadata(content)
            logger.info("KV connector metadata updated successfully")

        except Exception as e:
            logger.error(f"Failed to update KV connector metadata: {e}")
            # Don't raise - this is not critical, log and continue

    def _cleanup_scheduler_requests(self) -> None:
        """Clean up all requests in scheduler before worker restart.

        This ensures that when workers are restarted during DEGRADE (scale down),
        there are no pending requests that could cause state inconsistency.
        """
        logger.info("Cleaning up scheduler requests before worker restart")

        if not hasattr(self, '_engine_core_ref') or not self._engine_core_ref:
            logger.debug("No engine_core_ref available, skipping scheduler cleanup")
            return

        engine_core = self._engine_core_ref

        # First, drain the input_queue to prevent new requests from being processed
        if hasattr(engine_core, 'input_queue'):
            try:
                input_queue = engine_core.input_queue
                if hasattr(input_queue, 'qsize'):
                    qsize = input_queue.qsize()
                    if qsize > 0:
                        logger.info(f"Draining {qsize} items from input_queue")
                        drained = 0
                        while not input_queue.empty() and drained < qsize + 10:
                            try:
                                input_queue.get_nowait()
                                drained += 1
                            except Exception:
                                break
                        logger.info(f"Drained {drained} items from input_queue")
            except Exception as e:
                logger.warning(f"Failed to drain input_queue: {e}")

        if not hasattr(engine_core, 'scheduler'):
            logger.debug("Engine core has no scheduler, skipping cleanup")
            return

        scheduler = engine_core.scheduler

        if not scheduler.has_unfinished_requests():
            logger.debug("Scheduler has no unfinished requests")
            return

        # 1) finish_requests: 将scheduler中的所有请求设为FINISHED_ABORTED(None会遍历所有请求)
        # 2)  _send_abort_outputs: 每个相关客户端都能收到对应请求的 ABORT 完成通知，而不是让客户端一直等待
        logger.info("Aborting all unfinished requests in scheduler")
        aborted_reqs = scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        engine_core._send_abort_outputs(aborted_reqs)

        # Clear waiting queue if it exists
        if hasattr(scheduler, 'wait_queue'):
            wait_queue = getattr(scheduler, 'wait_queue', None)
            if wait_queue and hasattr(wait_queue, 'clear'):
                wait_queue.clear()
                logger.debug("Wait queue cleared")

        # Clear running queue if it exists
        if hasattr(scheduler, 'running_queue'):
            running_queue = getattr(scheduler, 'running_queue', None)
            if running_queue and hasattr(running_queue, 'clear'):
                running_queue.clear()
                logger.debug("Running queue cleared")

        # Reset sequence ID if it exists
        if hasattr(scheduler, 'next_seq_id'):
            scheduler.next_seq_id = 0
            logger.debug("Scheduler sequence ID reset to 0")

    def _cleanup_message_queues_and_workers(self) -> None:
        """清理消息队列和 Worker 引用。

        处理流程：
        0. 停止健康监控，避免误报 Worker 死亡
        1. 关闭 death_writer 通知 Worker 退出
        2. 等待 Worker 进程终止
        3. 关闭响应队列
        4. 清除 Worker 引用，防止残留数据
        """
        logger.info("Cleaning up message queues and worker references")

        # 0. 先停止健康监控，避免误报 Worker 死亡
        # 当我们主动清理 workers 时，健康监控器可能会检测到已死亡的 workers
        # 并触发 ITSFailureCallback 设置 wait_new_deployment，导致状态混乱
        # 注意：stop() 已经设置了 _failure_handled=True，不要在此调用 reset_failure_state()
        # 否则会撤销 stop() 的保护，导致误触发回调
        if hasattr(self, '_health_monitor') and self._health_monitor:
            self._health_monitor.stop()
            logger.info("Health monitor stopped before worker cleanup")

        # 1. 关闭 death_writer 并等待 workers 终止
        if workers := getattr(self, "workers", None):
            for w in workers:
                # 关闭 death_writer 以通知子进程退出
                if w.death_writer is not None:
                    w.death_writer.close()
                    w.death_writer = None
            self._ensure_worker_termination([w.proc for w in workers])

            # 2. 关闭响应队列并清除引用
            for w in workers:
                try:
                    # 关闭响应队列
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

                    # 清除对端 Worker 响应队列引用
                    if hasattr(w, 'peer_worker_response_mqs'):
                        w.peer_worker_response_mqs = None

                    # 清除进程引用
                    if hasattr(w, 'proc'):
                        w.proc = None

                except Exception as e:
                    logger.warning(f"Error cleaning up worker reference: {e}")
                    # 即使某个 worker 清理失败，也继续处理其他 worker

            # 清除 workers 列表
            self.workers = []

        # 3. 清理 Executor 级别的消息队列
        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []
        time.sleep(VLLM_ITS_HEALTH_CHECK_INTERVAL)
        logger.info("Message queues and worker references cleaned up")

    def _init_workers(self) -> None:
        """Initialize workers with current VllmConfig."""
        logger.info("Initializing workers with updated VllmConfig")

        # Reset distributed environment if needed
        if model_parallel_is_initialized():
            destroy_model_parallel()
        destroy_distributed_environment()

        # Get distributed init method
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )

        # Calculate parallel config based on strategy
        if self.current_strategy:
            # 从策略获取健康的 NPU 列表，设置 ASCEND_RT_VISIBLE_DEVICES
            # 这样新启动的 worker 进程只会使用健康的卡
            if self.current_strategy.deploy_type == DeployType.RECOVER:
                if is_mm_scene():
                    # 此处假设使用所有挂载卡(即所有挂载卡是visible的)
                    # 查看挂载的所有卡
                    _, mounted_npu_id_list_int = self.get_davinci_devices()
                    mounted_npu_id_list_int = sorted(mounted_npu_id_list_int)
                    mounted_npu_id_list_str = [str(i) for i in mounted_npu_id_list_int]

                    # 当前executor的所有好卡集合(所有卡均为好卡)
                    # self.init_executor_state_request.npu_id: 启动时executor的npu-id绑定信息
                    cur_healthy_npu_ids = set([str(i) for i in self.init_executor_state_request.npu_id])

                    # 从物理npu-id映射到从0开始的连续逻辑id
                    sorted_mounted_npu_id_list_str = sorted(
                        mounted_npu_id_list_str, key=int
                    )
                    new_npu_id_list = []
                    for npu_id in cur_healthy_npu_ids:
                        new_npu_id_list.append(str(sorted_mounted_npu_id_list_str.index(npu_id)))
                else:
                    # 独立部署：--privilege + RT_VISIBLE
                    new_npu_id_list = [str(i) for i in self.init_executor_state_request.npu_id]
                healthy_npu_str = ",".join(sorted(new_npu_id_list, key=int))
                logger.info(f"Setting ASCEND_RT_VISIBLE_DEVICES to healthy NPUs: {healthy_npu_str}")
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = healthy_npu_str
            else:
                # 决策中心下发健康卡数据时, 包含当前节点内的所有卡信息, e.g. [0,1,2,3,4(x),5,6,7] 
                healthy_npu_list = self._get_healthy_npu_ids_from_strategy(self.current_strategy)
                if is_mm_scene():
                    # 查看挂载的所有卡
                    _, mounted_npu_id_list_int = self.get_davinci_devices()
                    mounted_npu_id_list_int = sorted(mounted_npu_id_list_int)
                    mounted_npu_id_list_str = [str(i) for i in mounted_npu_id_list_int]
                    
                    # 挂载卡的所有好卡集合
                    mounted_health_npu_ids = set(mounted_npu_id_list_str) & set(healthy_npu_list) # [4(x),5,6,7]

                    # 可用卡的所有好卡集合
                    # 健康且visible卡 = 健康且挂载卡
                    visible_health_npu_ids = mounted_health_npu_ids

                    # 当前executor的所有好卡集合
                    # self.init_executor_state_request.npu_id: 启动时executor的npu-id绑定信息
                    cur_healthy_npu_ids = set([str(i) for i in self.init_executor_state_request.npu_id]) & set(visible_health_npu_ids)

                    # 从物理npu-id映射到从0开始的连续逻辑id
                    sorted_mounted_npu_id_list_str = sorted(
                        mounted_npu_id_list_str, key=int
                    )
                    new_npu_id_list = []
                    for npu_id in cur_healthy_npu_ids:
                        new_npu_id_list.append(str(sorted_mounted_npu_id_list_str.index(npu_id)))
                    healthy_npu_str = ",".join(
                        sorted(new_npu_id_list, key=int)
                    )
                    logger.info(f"Setting ASCEND_RT_VISIBLE_DEVICES to healthy NPUs: {healthy_npu_str}")
                    logger.debug(f"self.init_executor_state_request.npu_id={self.init_executor_state_request.npu_id}, {type(self.init_executor_state_request.npu_id[0])},healthy_npu_list={healthy_npu_list}, cur_healthy_npu_ids: {cur_healthy_npu_ids}, visible_health_npu_ids={visible_health_npu_ids}, sorted_mounted_npu_id_list_str={sorted_mounted_npu_id_list_str}, mounted_health_npu_ids={mounted_health_npu_ids}, ")
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = healthy_npu_str
                else:
                    # 独立部署：--privilege + RT_VISIBLE
                    npu_id_list = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
                    new_npu_id_list = []
                    if healthy_npu_list:
                        for npu_id in npu_id_list:
                            if npu_id in healthy_npu_list:
                                new_npu_id_list.append(npu_id)
                        healthy_npu_str = ",".join(
                            sorted(new_npu_id_list, key=int)
                        )
                        logger.info(f"Setting ASCEND_RT_VISIBLE_DEVICES to healthy NPUs: {healthy_npu_str}")
                        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = healthy_npu_str

            self._get_engine_parallel_config(self.current_strategy)
        # WorkerProc.rank/local_rank 仍是节点内（DP 内）的本地 rank，全局
        # torch.distributed rank 偏移由 v0.23 hetero
        # init_distributed_environment 按 get_rank_offset_for_dp 计算
        # （DP4TP(3,4,4,4) 时 0/3/7/11）。不要把全局偏移传给
        # WorkerProc，否则 _is_driver_worker(rank % local_tp_size == 0)
        # 会在 DP1..3 失效。
        global_start_rank = (
            self.parallel_config.local_world_size
            * self.parallel_config.node_rank_within_dp
        )
        # 从 parallel_config 获取并行配置（已在 _get_engine_parallel_config 中更新）
        new_tp = self.parallel_config.tensor_parallel_size
        new_dp = self.parallel_config.data_parallel_size
        world_size = self.parallel_config.world_size
        local_world_size = self.parallel_config.local_world_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        logger.info(f"Starting workers: parallel_config：{self.vllm_config.parallel_config}")
        logger.info(f"Parallel config check: tp={new_tp}, dp={new_dp}, pp={pp_size}, pcp={pcp_size}, "
                    f"world_size={world_size}, local_world_size={local_world_size}, "
                    f"nnodes={self.parallel_config.nnodes}, "
                    f"data_parallel_size_local={self.parallel_config.data_parallel_size_local}, "
                    f"global_start_rank={global_start_rank}")

        # dp=0/tp=0 场景：此实例需要空转，不启动 workers
        # 保持进程不退出，等待后续恢复策略
        if new_dp == 0 or new_tp == 0:
            logger.info(f"Scale-to-zero detected: new_dp={new_dp}, new_tp={new_tp}, skipping worker initialization")
            self.workers = []
            self.world_size = 0
            self.local_world_size = 0
            self.response_mqs = []
            self.futures_queue = deque[FutureWrapper]()
            # 停止健康监控，避免检测到 worker 死亡后重复触发
            # stop() 设置 _running=False 和 _failure_handled=True
            # update_workers([]) 清空 workers 列表
            # 最后再次设置 _failure_handled=True 防止竞态
            if hasattr(self, '_health_monitor') and self._health_monitor:
                self._health_monitor.stop()
                self._health_monitor.update_workers([])  # 清空 workers 列表
                self._health_monitor._failure_handled = True  # 再次设置确保阻止 callback
            logger.info("Executor idle mode: no workers running, waiting for recovery strategy")
            return
        # Check if this is the leader node of the DP group.
        # 与 vllm 0.23 / vllm-ascend 一致使用 node_rank_within_dp：
        # 多节点 DP 时每个 DP 组都需要一个 leader 创建 rpc_broadcast_mq，
        # 只有全局 node_rank==0 建队列会让其它节点 worker 收不到调度。
        is_leader = self.parallel_config.node_rank_within_dp == 0
        logger.info(f"Starting workers: world_size={world_size}, local_world_size={local_world_size}, "
                    f"global_start_rank={global_start_rank}, tp={new_tp}, dp={new_dp}，is_leader：{is_leader}")
        # Create MessageQueue for leader node
        scheduler_output_handle = None
        if is_leader:
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                world_size,
                local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=self.parallel_config.master_addr,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Recreate workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        inherited_fds: list[int] | None = [] if context.get_start_method() == "fork" else None

        # Create worker processes
        for local_rank in range(local_world_size):
            global_rank = global_start_rank + local_rank
            is_driver_worker = self._is_driver_worker(global_rank)
            logger.info(
                f"******##########global_rank:{global_rank},-----global_start_rank:{global_start_rank}-----self.word_size:{self.world_size}"
                f"----local_world_size:{self.local_world_size}----node_rank_within_dp:{self.parallel_config.node_rank_within_dp}")
            unready_worker_handle = AscendWorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=global_rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
                shared_worker_lock=shared_worker_lock,
                is_driver_worker=is_driver_worker,
                inherited_fds=inherited_fds,
            )
            unready_workers.append(unready_worker_handle)

            if inherited_fds is not None:
                inherited_fds.append(unready_worker_handle.death_writer.fileno())
                inherited_fds.append(unready_worker_handle.ready_pipe.fileno())
        logger.info(
            f"Creating worker unready_workers：{len(unready_workers)}")
        # Wait for workers to be ready
        self.workers = AscendWorkerProc.wait_for_ready(unready_workers)
        self.world_size = world_size
        self.local_world_size = local_world_size

        if not self.workers:
            logger.error("No workers initialized, restart failed")
            raise RuntimeError("Worker restart failed: no workers available")

        logger.info(
            f"Initialized {len(self.workers)} workers with world_size={world_size}, local_world_size={local_world_size}")

        # Setup response message queues
        self.response_mqs = []
        if is_leader:
            for rank in range(self.world_size):
                if rank < self.local_world_size:
                    local_message_queue = self.workers[rank].worker_response_mq
                    assert local_message_queue is not None
                    self.response_mqs.append(local_message_queue)
                else:
                    remote_message_queue = self.workers[0].peer_worker_response_mqs[rank]
                    assert remote_message_queue is not None
                    self.response_mqs.append(remote_message_queue)

        # Ensure message queues are ready
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        # Reset futures queue
        self.futures_queue = deque[FutureWrapper]()
        logger.info("Message queues setup complete")

    def _update_vllm_config_for_restart(self) -> None:
        """Update VllmConfig with current strategy before worker restart.

        Note: We convert DeployStrategy to a JSON-serializable dict to avoid
        JSON serialization errors in vLLM's compute_hash() during profile_run.
        """
        if self.current_strategy is None:
            logger.warning("No current_strategy available, skipping VllmConfig update")
            return

        # Use additional_config to store zero_interrupt_config
        if self.vllm_config.additional_config is None:
            self.vllm_config.additional_config = {}

        # Convert DeployStrategy (dataclass) to JSON-serializable dict using asdict
        # This is needed because vLLM's compute_hash() tries to serialize additional_config
        from dataclasses import asdict
        strategy_dict = asdict(self.current_strategy)
        # Convert enum values to strings
        strategy_dict = self._convert_enums_to_values(strategy_dict)

        # 注入异构重启所需的完整拓扑信息。所有 DP executor 都使用同一份
        # heterogeneous_dp_config，worker 启动时据此建立 15-rank 的
        # torch.distributed 世界和各 DP 的 TP/DP/EP 通信组。
        engine_parallel_config_list = strategy_dict.get(
            "engine_parallel_config", []
        )
        if engine_parallel_config_list and is_heterogeneous_restart(
            strategy_dict
        ):
            heterogeneous_dp_config = get_heterogeneous_dp_config(strategy_dict)
            strategy_dict["heterogeneous_dp_config"] = heterogeneous_dp_config
            strategy_dict["global_world_size"] = sum(
                cfg["tp_size"] for cfg in heterogeneous_dp_config
            )
            strategy_dict["global_start_rank"] = get_global_start_rank(
                strategy_dict
            )
            # 保留 vllm_plugins 原生配置名，worker/model patch 直接读取。
            current_config = next(
                (
                    cfg
                    for cfg in engine_parallel_config_list
                    if str(cfg.get("executor_id", None))
                    == str(strategy_dict.get("executor_id", "0"))
                ),
                None,
            )
            if current_config is not None:
                shardings = current_config.get("tp_asymmetric_shardings")
                if shardings is None and (
                    current_config.get("new_tp") is not None
                    and current_config.get("new_tp") != current_config.get("tp")
                ):
                    shardings = get_tp_asymmetric_shardings(strategy_dict)
                strategy_dict["tp_asymmetric_shardings"] = shardings

        self.vllm_config.additional_config["zero_interrupt_config"] = strategy_dict
        logger.info(f"Updated VllmConfig.additional_config with strategy: {self.current_strategy.deploy_type.value}")

    def _convert_enums_to_values(self, obj: Any) -> Any:
        """Recursively convert enum values to their string representation."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            return [self._convert_enums_to_values(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._convert_enums_to_values(value) for key, value in obj.items()}
        # Handle enum
        if hasattr(obj, "value"):
            return obj.value
        return obj

    # 缩容
    def _execute_degrade_strategy(self, strategy: DeployStrategy) -> bool:
        """Execute DEGRADE (scale down) strategy.

        Args:
            strategy: DEGRADE deployment strategy

        Returns:
            True if successful
        """
        logger.info("Executing DEGRADE strategy")

        try:
            self.executor_state = ExecutorState.EXECUTING_STRATEGY

            # Scheduler cleanup is now handled inside _cleanup_and_restart_workers
            # via _cleanup_scheduler_requests() for all DEGRADE scenarios
            self._cleanup_and_restart_workers()

            # dp=0 场景：空转状态
            if self.world_size == 0:
                self.executor_state = ExecutorState.STOPPED
            else:
                self.executor_state = ExecutorState.RUNNING
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_SUCCESS)
            return True

        except Exception as e:
            logger.error(f"Error during DEGRADE strategy execution: {e}")
            self.executor_state = ExecutorState.EXECUTING_STRATEGY_FAILED
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_FAIL)
            return False

    # 恢复
    def _execute_recover_strategy(self, strategy: DeployStrategy) -> bool:
        """Execute RECOVER (scale up) strategy.

        Args:
            strategy: RECOVER deployment strategy

        Returns:
            True if successful
        """
        logger.info("Executing RECOVER strategy")

        try:
            self.executor_state = ExecutorState.RECOVERING

            # 决策中心可能重放同一份 RECOVER（strategy_sync 会转发重复策略）。
            # DP>1 且当前拓扑已经等于目标拓扑时不需要重启：各 executor 的
            # dp_group / worker world 都是目标布局，重复全量重启反而会在没有
            # barrier 保护的情况下异步杀 worker。DP=1 没有跨 executor 通信域，
            # 保留原来的无条件重启语义。
            if self.parallel_config.data_parallel_size > 1 and (
                not self._strategy_requires_full_restart(strategy)
            ):
                logger.info(
                    "RECOVER strategy matches current topology "
                    "(dp=%s, tp=%s); skipping redundant worker restart.",
                    self.parallel_config.data_parallel_size,
                    self.parallel_config.tensor_parallel_size,
                )
                self.executor_state = ExecutorState.RUNNING
                self._report_deploy_status(
                    strategy, DeployState.EXECUTOR_DEPLOY_SUCCESS
                )
                return True

            self._cleanup_and_restart_workers()

            # dp=0 场景：空转状态
            if self.world_size == 0:
                self.executor_state = ExecutorState.STOPPED
            else:
                self.executor_state = ExecutorState.RUNNING
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_SUCCESS)
            return True

        except Exception as e:
            logger.error(f"Error during RECOVER strategy execution: {e}")
            self.executor_state = ExecutorState.EXECUTING_STRATEGY_FAILED
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_FAIL)
            return False

    # pd
    def _execute_pd_rebuild_strategy(self, strategy: DeployStrategy) -> bool:
        """Execute PD_REBUILD strategy for P/D分离 scenarios.

        区分处理：
        - 故障实例（有不健康NPU）：使用 new_tp/new_dp 重启 Workers
        - 健康实例（全部NPU健康）：通过 RPC 更新 KVConnector

        Args:
            strategy: PD_REBUILD deployment strategy

        Returns:
            True if successful
        """
        logger.info("Executing PD_REBUILD strategy")

        try:
            # DeepSeek-V4 DP4TP4 -> DP4TP(3,4,4,4) 属于异构 TP 切换：
            # MoE 权重、EP 通信组和全局 rank 布局全部发生变化，因此
            # **所有 DP**（包括没有故障卡的 DP）都必须重启 worker，
            # 不能只重启故障卡所在的 DP。
            if self._strategy_requires_full_restart(strategy):
                current_config = self._get_current_engine_config(strategy)
                new_tp = (
                    current_config.new_tp
                    if current_config.new_tp is not None
                    else current_config.tp
                )
                new_dp = (
                    current_config.new_dp
                    if current_config.new_dp is not None
                    else current_config.dp
                )
                logger.info(
                    "Heterogeneous TP PD_REBUILD: restarting workers of EVERY "
                    "DP instance with new config: TP=%s, DP=%s, shardings=%s",
                    new_tp, new_dp,
                    getattr(current_config, "tp_asymmetric_shardings", None),
                )
                # _init_workers 会从 strategy 中提取并应用完整异构配置。
                self._cleanup_and_restart_workers()
            elif self._is_fault_instance_for_pd(strategy):
                # 兼容原有非异构故障恢复：只有故障实例重启。
                current_config = self._get_current_engine_config(strategy)
                new_tp = (
                    current_config.new_tp
                    if current_config.new_tp is not None
                    else current_config.tp
                )
                new_dp = (
                    current_config.new_dp
                    if current_config.new_dp is not None
                    else current_config.dp
                )
                logger.info(
                    f"This instance is fault (has unhealthy NPUs), "
                    f"restarting workers with new config: TP={new_tp}, DP={new_dp}"
                )
                self._cleanup_and_restart_workers()
            else:
                logger.info("This instance is healthy, updating KVConnector via RPC")
                # 健康实例通过 RPC 更新 KVConnector
                self.collective_rpc("update_kv_connector_for_pd",
                                    args=(strategy,),
                                    timeout=60)

            if self.world_size == 0:
                self.executor_state = ExecutorState.STOPPED
            else:
                self.executor_state = ExecutorState.RUNNING
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_SUCCESS)
            return True

        except Exception as e:
            logger.error(f"Error during PD_REBUILD strategy execution: {e}")
            self.executor_state = ExecutorState.EXECUTING_STRATEGY_FAILED
            self._report_deploy_status(strategy, DeployState.EXECUTOR_DEPLOY_FAIL)
            return False

    def _get_current_engine_config(self, strategy: DeployStrategy) -> EngineParallelConfig:
        """获取当前实例的 EngineParallelConfig

        Args:
            strategy: PD_REBUILD deployment strategy

        Returns:
            当前实例的 EngineParallelConfig
        """
        executor_id = strategy.executor_id
        for config in strategy.engine_parallel_config:
            if str(config.executor_id) == str(executor_id):
                return config

        # 如果找不到匹配的 config，返回默认配置
        logger.warning(f"Cannot find matching engine_parallel_config for executor_id={executor_id}, using defaults")
        return EngineParallelConfig(tp=1, dp=1)

    def _is_fault_instance_for_pd(self, strategy: DeployStrategy) -> bool:
        """判断当前实例是否是故障实例（通过 NPU 健康状态）

        Args:
            strategy: PD_REBUILD deployment strategy

        Returns:
            True if this instance is a fault instance (has unhealthy NPUs)
        """
        npu_state = strategy.engine_npu_healthy_state
        if not npu_state:
            # 没有状态信息，默认为故障需要重启
            logger.warning("No NPU state in strategy, treating as fault instance")
            return True

        # 获取当前节点 IP
        current_host_ip, pod_ip = get_ip_mm()
        logger.debug(f"Current host IP: {current_host_ip}")

        # 遍历 NPU 状态，找到当前节点对应的状态
        for state in npu_state:
            if not hasattr(state, 'server_list'):
                continue
            for server in state.server_list:
                server_host_ip = getattr(server, 'host_ip', None)
                if server_host_ip == current_host_ip:
                    # 检查该 server 的设备是否有不健康的
                    if hasattr(server, 'device'):
                        for device in server.device:
                            if not getattr(device, 'healthy', True):
                                logger.info(f"Found unhealthy device on {current_host_ip}: {device}")
                                return True  # 有不健康设备，是故障实例
        return False  # 全部健康，是健康实例

    def _strategy_requires_full_restart(self, strategy: DeployStrategy) -> bool:
        """判断策略是否需要所有 DP executor 重启 worker。

        DeepSeek-V4 异构场景（DP4TP4 -> DP4TP(3,4,4,4)）改变了全局
        world_size、MoE EP 通信组和非对称权重切分。vllm_plugins 的
        ``tp_asymmetric_shardings`` / ``new_tp`` 一旦出现，所有 DP rank
        都必须重建通信组，健康 DP 仅做 KV connector RPC 更新是不够的。

        DEGRADE / PD_REBUILD / RECOVER 三种会改变拓扑的策略都必须把
        完整的 engine_parallel_config 下发给每一个 DP executor。
        """
        if strategy.deploy_type == DeployType.RECOVER:
            # 从异构/缩容拓扑恢复同样会重建 world_size 和通信组。只要当前
            # 拓扑是异构的，或备份的对称 tp/dp 与当前不一致，或目标 dp 与
            # 当前不一致，就必须让全部 DP 同时重启；否则会复现与 DEGRADE
            # 相同的旧通信域问题。
            # 纯 DP 恢复（DP15TP1 -> DP16TP1）时 tp 没有变化，旧逻辑只看
            # tp 会误判为“无需全量重启”：健康 executor 先杀旧 worker，而
            # 恢复中的 executor 还在等 16-rank barrier，新 worker 的
            # init_process_group 会永久等待。
            backup = getattr(self, "backup_parallel_config", {}) or {}
            current_tp = self.parallel_config.tensor_parallel_size
            current_dp = self.parallel_config.data_parallel_size
            current_is_hetero = bool(
                getattr(self.parallel_config, "is_heterogeneous_tp", False)
            )
            target_dp = max(
                (
                    conf.new_dp
                    if conf.new_dp is not None
                    else conf.dp
                    for conf in strategy.engine_parallel_config
                ),
                default=current_dp,
            )
            requires_full_restart = recover_requires_full_restart(
                backup=backup,
                current_tp=current_tp,
                current_dp=current_dp,
                current_is_heterogeneous=current_is_hetero,
                target_dp=target_dp,
            )
            if not requires_full_restart:
                return False
            # fall through to the coverage validation below.
        elif strategy.deploy_type in (DeployType.DEGRADE, DeployType.PD_REBUILD):
            requires_full_restart = False
            for conf in strategy.engine_parallel_config:
                new_tp = conf.new_tp
                if new_tp is not None and new_tp != conf.tp:
                    requires_full_restart = True
                if conf.tp_asymmetric_shardings:
                    requires_full_restart = True
                if new_tp == 0 and conf.new_dp == 0:
                    # 有 executor 被缩到零，通信拓扑也发生变化。
                    requires_full_restart = True
            if not requires_full_restart:
                return False
        else:
            return False

        # Fail closed instead of deadlocking: every active DP rank must be
        # present in the strategy.  A missing rank would make the new global
        # init_process_group wait for ranks that will never join.
        expected_dp = max(
            (
                conf.new_dp
                if conf.new_dp is not None
                else conf.dp
                for conf in strategy.engine_parallel_config
            ),
            default=0,
        )
        present = {
            int(getattr(conf, "data_parallel_rank", -1))
            for conf in strategy.engine_parallel_config
            if getattr(conf, "data_parallel_rank", None) is not None
            and int(getattr(conf, "data_parallel_rank", -1)) >= 0
        }
        missing = set(range(expected_dp)) - present
        if missing:
            raise ValueError(
                "Heterogeneous full-restart strategy does not cover DP ranks "
                f"{sorted(missing)}. The decision center must send the same "
                "engine_parallel_config to every DP executor because the "
                "global worker world size and MoE communication groups are "
                "being rebuilt."
            )
        return True

    def _get_healthy_npu_ids_from_strategy(self, strategy: DeployStrategy) -> list[str] | None:
        """从策略中获取当前节点健康的 NPU ID 列表。

        Args:
            strategy: PD_REBUILD deployment strategy

        Returns:
            健康 NPU ID 列表（如 "4,5"），如果无法获取则返回 None
        """
        npu_state = strategy.engine_npu_healthy_state
        if not npu_state:
            return None

        current_host_ip, pod_ip = get_ip_mm()
        healthy_npu_ids = []

        for state in npu_state:
            if not hasattr(state, 'server_list'):
                continue
            for server in state.server_list:
                server_host_ip = getattr(server, 'host_ip', None)
                if server_host_ip == current_host_ip:
                    if hasattr(server, 'device'):
                        for device in server.device:
                            if getattr(device, 'healthy', True):
                                npu_id = getattr(device, 'npu_id', None)
                                if npu_id is not None:
                                    healthy_npu_ids.append(str(npu_id))

        logger.info(f"Healthy NPUs on {current_host_ip}: {healthy_npu_ids}")
        return healthy_npu_ids

    # 上报执行状态
    def _report_deploy_status(self, strategy: DeployStrategy, state: DeployState) -> None:
        """Report deployment status to decision center.

        DecisionMakingCenter waits for ``report_deploy_status`` keyed by the
        executor id it generated during ``/init_executor_state``
        (``exe-<service>-<engine>-<n>``), not by the local numeric
        data_parallel_rank. Prefer the registered id; fall back to the
        strategy's top-level executor_id for manual trigger testing.

        Args:
            state: Deployment state
        """
        try:
            if self._decision_center_client:
                executor_id = (
                    self._decision_center_executor_id
                    or str(strategy.executor_id)
                )
                self._decision_center_client.report_deploy_status(
                    executor_id=executor_id,
                    deploy_state=state,
                    update_engine_info=strategy.update_engine_info
                )
        except Exception as e:
            logger.error(f"Error reporting deploy status: {e}")

    def shutdown(self) -> None:
        """Shutdown the executor and cleanup resources."""
        # Idempotent check: already shutting down, skip
        if getattr(self, "shutting_down", False):
            logger.info("ITSMultiprocExecutor already shutting down, skipping")
            return

        logger.info("Shutting down ITSMultiprocExecutor")

        # Set shutdown flag FIRST to prevent health monitor from reacting to worker termination
        self.shutting_down = True

        # Stop health monitor BEFORE terminating workers
        # This prevents health monitor from detecting worker death and triggering callback
        if self._health_monitor:
            self._health_monitor.stop()

        # Stop other components
        if self._http_server:
            self._http_server.stop()

        if self._strategy_sync_thread:
            self._strategy_sync_thread.stop()

        # Report stopped state
        self.executor_state = ExecutorState.STOPPED

        # Only tear down workers/message queues here.  Calling
        # _cleanup_and_restart_workers() would run _init_workers() and
        # spawn a fresh set of workers during shutdown; the parent
        # MultiprocExecutor.shutdown() skips worker termination once
        # shutting_down=True, so those workers would leak.
        self._cleanup_message_queues_and_workers()

        # Use current_strategy if available, otherwise create a dummy strategy for shutdown reporting
        if self.current_strategy:
            self._report_deploy_status(self.current_strategy, DeployState.EXECUTOR_STOP)

        # Call parent shutdown
        super().shutdown()

    @property
    def executor_id(self) -> str:
        """Get executor ID."""
        if self.current_strategy:
            return self.current_strategy.executor_id
        return "0"

    @property
    def state(self) -> ExecutorState:
        """Get current executor state."""
        return self.executor_state

    def _try_backup_origin_parallel_config_when_degrade(self):
        """
            第一次缩容时备份原始对称并行策略，在恢复时使用
            当发生多次缩容时，上一次的缩容场景的并行策略不会覆盖原始并行策略
        """
        backup_parallel_config = getattr(self, "backup_parallel_config", {})
        logger.debug(f"+++++[mzm]++++++++init_backup parallel config:{backup_parallel_config}+++++[mzm]++++++++")
        if len(backup_parallel_config) == 0:
            backup_parallel_config["data_parallel_rank_local"] = self.parallel_config.data_parallel_rank_local
            backup_parallel_config["data_parallel_rank"] = self.parallel_config.data_parallel_rank
            backup_parallel_config["data_parallel_size"] = self.parallel_config.data_parallel_size
            backup_parallel_config["tensor_parallel_size"] = self.parallel_config.tensor_parallel_size
            backup_parallel_config["world_size"] = self.parallel_config.world_size
            logger.debug(f"+++++[mzm]++++++++backup parallel config:{backup_parallel_config}+++++[mzm]++++++++")
            self.backup_parallel_config = backup_parallel_config

    def _get_engine_parallel_config(self, strategy: DeployStrategy):
        """从策略获取新的 TP/DP 配置。

        除了更新当前 executor 的 TP/DP 外，还把完整的
        ``heterogeneous_dp_config``（每个 DP rank 的 tp_size 与
        tp_asymmetric_shardings）写入 parallel_config。这样：
        - worker 可以据此建立 15-rank 的全局通信域；
        - DeepSeek-V4 权重加载可以读取 [2,1,1] 这类非对称切分；
        - DP4TP4 -> DP4TP(3,4,4,4) 时所有 DP 都会用同一份拓扑重启。
        """
        logger.info("Getting new TP/DP config from strategy")
        engine_parallel_config_list = sorted(
            strategy.engine_parallel_config,
            key=lambda x: getattr(x, 'data_parallel_rank', 0) or 0
        )

        executor_id = strategy.executor_id
        logger.info(
            "executor_id=%s, engine_parallel_config_list=%s",
            executor_id, engine_parallel_config_list,
        )
        matched = False
        rm_data_parallel_rank = 0

        # 构建完整异构拓扑（跳过 new_tp/new_dp 均为 0 的空转 executor）。
        hetero_configs = []
        for conf in engine_parallel_config_list:
            new_tp = conf.new_tp if conf.new_tp is not None else conf.tp
            new_dp = conf.new_dp if conf.new_dp is not None else conf.dp
            if new_tp == 0 and new_dp == 0:
                continue
            ratios = conf.tp_asymmetric_shardings
            if ratios is None and new_tp != conf.tp:
                # 与 get_tp_asymmetric_shardings 的 legacy 逻辑保持一致。
                tmp_strategy = {
                    "executor_id": conf.executor_id,
                    "engine_parallel_config": [asdict(conf)],
                }
                ratios = get_tp_asymmetric_shardings(tmp_strategy)
            hetero_configs.append(
                {
                    "executor_id": str(conf.executor_id),
                    "dp_rank": int(
                        getattr(conf, "data_parallel_rank", 0) or 0
                    ),
                    "tp_size": int(new_tp),
                    "tp_sharding_ratios": (
                        [int(r) for r in ratios] if ratios else None
                    ),
                }
            )
        hetero_configs.sort(key=lambda c: c["dp_rank"])
        # 跳过被缩到零的 executor 后重新连续编号，保证
        # heterogeneous_dp_config 覆盖 0..N-1。
        for new_rank, cfg in enumerate(hetero_configs):
            cfg["dp_rank"] = new_rank

        # 纯 DP 扩缩容（各 rank tp 不变且无显式非对称配比）继续走
        # 原有对称流程，避免 legacy 多机/EPLB 场景被 hetero 校验拒绝。
        hetero_restart = any(
            (
                c.new_tp is not None and c.new_tp != c.tp
            ) or bool(c.tp_asymmetric_shardings)
            for c in engine_parallel_config_list
        )
        hetero_configs_for_pc = hetero_configs if hetero_restart else None
        own_dp_rank = (
            next(
                (
                    cfg["dp_rank"]
                    for cfg in hetero_configs
                    if cfg["executor_id"] == str(executor_id)
                ),
                None,
            )
            if hetero_restart else None
        )

        for idx, engine_parallel_config in enumerate(engine_parallel_config_list):
            engine_executor_id = engine_parallel_config.executor_id
            logger.info(
                "Checking executor_id match: strategy.executor_id=%s (%s) vs "
                "config.executor_id=%s (%s), config: new_tp=%s, new_dp=%s, "
                "tp=%s, dp=%s, tp_asymmetric_shardings=%s",
                executor_id, type(executor_id).__name__,
                engine_executor_id,
                type(engine_executor_id).__name__
                if engine_executor_id is not None else "None",
                engine_parallel_config.new_tp,
                engine_parallel_config.new_dp,
                engine_parallel_config.tp,
                engine_parallel_config.dp,
                engine_parallel_config.tp_asymmetric_shardings,
            )
            if engine_executor_id is None or executor_id is None:
                continue
            if str(executor_id) != str(engine_executor_id):
                continue

            matched = True
            if strategy.deploy_type == DeployType.RECOVER:
                # 恢复策略的参数和备份的并行配置应一致。没有备份时（例如
                # 实例启动后直接收到 RECOVER）退化为按当前配置恢复。
                backup = getattr(self, "backup_parallel_config", {})
                expected_tp = backup.get(
                    "tensor_parallel_size",
                    self.parallel_config.tensor_parallel_size,
                )
                expected_dp = backup.get(
                    "data_parallel_size",
                    self.parallel_config.data_parallel_size,
                )
                expected_rank = backup.get(
                    "data_parallel_rank",
                    self.parallel_config.data_parallel_rank,
                )
                assert engine_parallel_config.tp == expected_tp, (
                    f'{engine_parallel_config.tp} == {expected_tp}'
                )
                assert engine_parallel_config.dp == expected_dp, (
                    f'{engine_parallel_config.dp} == {expected_dp}'
                )
                assert (
                    engine_parallel_config.data_parallel_rank == expected_rank
                ), (
                    f'{engine_parallel_config.data_parallel_rank}'
                    f' == {expected_rank}'
                )

                self.parallel_config.tensor_parallel_size = engine_parallel_config.tp
                self.parallel_config.data_parallel_size = engine_parallel_config.dp
                self.parallel_config.data_parallel_rank = (
                    own_dp_rank
                    if own_dp_rank is not None
                    else engine_parallel_config.data_parallel_rank
                )
                self.parallel_config.data_parallel_rank_local = backup.get(
                    "data_parallel_rank_local",
                    self.parallel_config.data_parallel_rank_local,
                )
                self.parallel_config.heterogeneous_dp_config = self._to_heterogeneous_dp_config(hetero_configs_for_pc)
                self.parallel_config.__post_init__()
                self.parallel_config.data_parallel_rank = (
                    own_dp_rank
                    if own_dp_rank is not None
                    else engine_parallel_config.data_parallel_rank
                )
                self.parallel_config.data_parallel_rank_local = backup.get(
                    "data_parallel_rank_local",
                    self.parallel_config.data_parallel_rank_local,
                )
                self.parallel_config.heterogeneous_dp_config = self._to_heterogeneous_dp_config(hetero_configs_for_pc)
                # __post_init__/rank 回写之后，index 与单节点 local 字段必须
                # 与最终 rank 保持一致。D 端 DP15 -> DP16 恢复时
                # data_parallel_size_local 之前仍可能是 15（或 scale-to-zero
                # executor 的 0），会让 worker 的 Mooncake 端口偏移和
                # node_rank 计算指向旧拓扑。
                self.parallel_config.data_parallel_index = (
                    self.parallel_config.data_parallel_rank
                )
                if self.parallel_config.nnodes == 1:
                    self.parallel_config.data_parallel_rank_local = (
                        self.parallel_config.data_parallel_rank
                    )
                    self.parallel_config.data_parallel_size_local = (
                        self.parallel_config.data_parallel_size
                    )
            elif strategy.deploy_type in (DeployType.DEGRADE, DeployType.PD_REBUILD):
                # 连续多次缩容/异构重建只备份第一次的原始对称配置，
                # 供后续 RECOVER 恢复。
                self._try_backup_origin_parallel_config_when_degrade()

                # 汇总当前 executor 之前被置空（scale-to-zero）的 DP rank。
                rm_data_parallel_rank = len([
                    c for c in engine_parallel_config_list[:idx]
                    if c.new_tp == 0 and c.new_dp == 0 and c.dp > 1
                ])

                new_tp = engine_parallel_config.new_tp if engine_parallel_config.new_tp is not None else engine_parallel_config.tp
                new_dp = engine_parallel_config.new_dp if engine_parallel_config.new_dp is not None else engine_parallel_config.dp
                backup = getattr(self, "backup_parallel_config", {})
                logger.info(
                    "Matched executor_id=%s, backup_parallel_config=%s, "
                    "New TP=%s, new DP=%s, rm_data_parallel_rank=%s",
                    executor_id, backup, new_tp, new_dp,
                    rm_data_parallel_rank,
                )

                self.parallel_config.tensor_parallel_size = new_tp
                self.parallel_config.data_parallel_size = new_dp
                if own_dp_rank is not None:
                    # 异构拓扑里按 active executor 重新连续编号。
                    self.parallel_config.data_parallel_rank = own_dp_rank
                else:
                    explicit_dp_rank = getattr(
                        engine_parallel_config, "data_parallel_rank", None
                    )
                    if explicit_dp_rank is not None:
                        self.parallel_config.data_parallel_rank = (
                            explicit_dp_rank
                        )
                    else:
                        self.parallel_config.data_parallel_rank = (
                            backup.get("data_parallel_rank", 0)
                            - rm_data_parallel_rank
                        )
                self.parallel_config.heterogeneous_dp_config = self._to_heterogeneous_dp_config(hetero_configs_for_pc)
                if new_tp > 0 and new_dp > 0:
                    self.parallel_config.world_size = (
                        new_tp
                        * self.parallel_config.pipeline_parallel_size
                        * self.parallel_config.prefill_context_parallel_size
                    )
                else:
                    self.parallel_config.world_size = 0
                # data_parallel_index 是异构全局 rank 计算的输入
                # （get_global_rank），必须和重新编号后的 data_parallel_rank
                # 保持一致。
                self.parallel_config.data_parallel_index = (
                    self.parallel_config.data_parallel_rank
                )
                # 单节点 DP 下 local == global；scale-to-zero 后如果
                # active executor 被重新编号，local rank 也必须跟着变，
                # 否则 Mooncake worker 侧按 data_parallel_rank_local 计算
                # 端口偏移会指向旧 DP。
                if self.parallel_config.nnodes == 1:
                    self.parallel_config.data_parallel_rank_local = (
                        self.parallel_config.data_parallel_rank
                    )
                    self.parallel_config.data_parallel_size_local = (
                        self.parallel_config.data_parallel_size
                    )
            elif strategy.deploy_type == DeployType.STOP:
                self._try_backup_origin_parallel_config_when_degrade()
                self.parallel_config.tensor_parallel_size = 0
                self.parallel_config.data_parallel_size = 0
                self.parallel_config.world_size = 0
            else:
                raise ValueError(f"receive unknown strategy: {strategy}")

            logger.info(
                "executor_id=%s, TP=%s, DP=%s, world_size=%s, "
                "world_size_across_dp=%s, data_parallel_rank=%s, "
                "heterogeneous_dp_config=%s",
                executor_id,
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.data_parallel_size,
                self.parallel_config.world_size,
                self.parallel_config.world_size_across_dp,
                self.parallel_config.data_parallel_rank,
                self.parallel_config.heterogeneous_dp_config,
            )
            break

        if not matched:
            logger.warning(
                f"No matching executor config found for executor_id={executor_id} (type={type(executor_id).__name__}), "
                f"available configs: {[(ec.executor_id, type(ec.executor_id).__name__, ec.new_tp, ec.new_dp) for ec in engine_parallel_config_list]}"
            )

    @staticmethod
    def _to_heterogeneous_dp_config(configs: list[dict]):
        """Convert strategy dicts to ParallelConfig.HeterogeneousDPConfig."""
        if not configs:
            return None
        try:
            from vllm.config.parallel import HeterogeneousDPConfig
        except ImportError:
            from vllm_custom_plugins.plugins.zero_interrupt.vllm.config.parallel import (
                HeterogeneousDPConfig,
            )
        return [
            HeterogeneousDPConfig(
                **{
                    k: v
                    for k, v in cfg.items()
                    if k in ("dp_rank", "tp_size", "tp_sharding_ratios")
                }
            )
            for cfg in configs
        ]

    @staticmethod
    def get_davinci_devices():
        dev_path = '/dev/'
        davinci_devices = []
        davinci_numbers = []

        # 正则表达式匹配以 'davinci' 开头且后面是数字的文件名
        pattern = re.compile(r'^davinci(\d+)$')

        # 列出 /dev/ 目录下的所有文件
        for filename in os.listdir(dev_path):
            # 检查文件名是否符合正则表达式
            match = pattern.match(filename)
            if match:
                davinci_devices.append(os.path.join(dev_path, filename))
                davinci_numbers.append(int(match.group(1)))  # 提取数字部分

        return davinci_devices, davinci_numbers

    @staticmethod
    def _get_npu_memory_size(device_id=0):
        """获取指定NPU的显存总大小（单位：MB）。"""
        try:
            # 查询指定NPU的显存信息，-i指定设备ID，-t指定memory类型
            output = subprocess.check_output(
                ['npu-smi', 'info', '-i', str(device_id), '-t', 'memory'],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            for line in output.split('\n'):
                line = line.strip()
                # 寻找包含 "HBM Capacity(MB)" 的行
                if line.startswith('HBM Capacity(MB)'):
                    size_mb = int(line.split(':')[1].strip())
                    logger.info(f"NPU-{device_id} memory size: {size_mb} MB")
                    return size_mb
            return 32768
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute the npu-smi command.: {e.output}")
            return 32768



from vllm_custom_plugins.plugins.zero_interrupt.vllm_ascend.utils import patch_direct_register_custom_op
patch_direct_register_custom_op()  # [h30014172] 防止子进程重复注册 operator


class ITSNPUWorker(AscendWorkerProc):
    """Intelligent Transform Service NPU Worker.

    Main functionality:
    - DEGRADE/RECOVER: Apply new tp/dp from zero_interrupt_config
    - PD_REBUILD: Online KV-Cache transfer chain rebuild via RPC

    NOTE: worker_busy_loop dispatches string RPC methods on the wrapped
    worker implementation (self.worker -> WorkerWrapperBase -> NPUWorker),
    NOT on this WorkerProc subclass.  Therefore the PD_REBUILD RPC method
    must be installed on the NPUWorker class (see patch.py) and the
    StrategyHandler is attached to the wrapped worker below.
    """

    def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize ITSNPUWorker."""
        # Save vllm_config for later use
        self._vllm_config = vllm_config

        # Initialize fault keep before parent
        self._fault_keep_enabled = VLLM_ITS_ENABLE_FAULT_KEEP

        # Call parent init (will call _apply_zero_interrupt_config after worker ready)
        super().__init__(vllm_config, *args, **kwargs)

        # Patch worker's execute_dummy_batch to handle all_reduce errors during strategy execution
        self._patch_worker_execute_dummy_batch()

        # Initialize strategy handler after worker is ready
        self._strategy_handler = StrategyHandler(
            worker=self.worker,
            pd_rebuild_enabled=VLLM_ITS_ENABLE_PD_REBUILD,
        )

        # worker_busy_loop dispatches `update_kv_connector_for_pd` on the
        # wrapped worker implementation.  Attach the handler there as well so
        # the NPUWorker method installed by patch.py can find it.
        try:
            wrapped_worker = self.worker.worker
        except Exception:  # noqa: BLE001
            wrapped_worker = None
        if wrapped_worker is not None:
            wrapped_worker._its_strategy_handler = self._strategy_handler

        # Apply zero interrupt config (update tp/dp or execute PD rebuild)
        self._apply_zero_interrupt_config()

    def _patch_worker_execute_dummy_batch(self) -> None:
        """Patch worker's execute_dummy_batch to handle all_reduce errors.

        During strategy execution, some workers may be restarted while others
        are still running execute_dummy_batch. This causes all_reduce to fail
        with "Connection closed by peer". This patch catches that error and
        allows the worker to gracefully exit without triggering FAILURE response.
        """
        worker = getattr(self, 'worker', None)
        if not worker or not hasattr(worker, 'execute_dummy_batch'):
            return

        original_execute_dummy_batch = worker.execute_dummy_batch

        def patched_execute_dummy_batch():
            try:
                original_execute_dummy_batch()
            except RuntimeError as e:
                # Catch all_reduce connection errors during strategy execution
                # Don't re-raise - just log and return to avoid FAILURE response
                error_str = str(e)
                if "Connection closed by peer" in error_str or "all_reduce" in error_str:
                    logger.warning(f"Worker {self.rank}: execute_dummy_batch caught all_reduce error "
                                   f"during strategy execution, skipping: {e}")
                    return
                raise

        worker.execute_dummy_batch = patched_execute_dummy_batch
        logger.debug(f"Worker {self.rank}: patched execute_dummy_batch for all_reduce error handling")

        # Also patch execute_model to handle the same error case
        if hasattr(worker, 'execute_model'):
            original_execute_model = worker.execute_model

            def patched_execute_model(*args, **kwargs):
                try:
                    return original_execute_model(*args, **kwargs)
                except RuntimeError as e:
                    error_str = str(e)
                    if "Connection closed by peer" in error_str or "all_reduce" in error_str:
                        logger.warning(f"Worker {self.rank}: execute_model caught all_reduce error "
                                       f"during strategy execution, returning None: {e}")
                        return None
                    raise

            worker.execute_model = patched_execute_model
            logger.debug(f"Worker {self.rank}: patched execute_model for all_reduce error handling")

    def _apply_zero_interrupt_config(self) -> None:
        """Apply zero interrupt config from VllmConfig.

        This reads the configuration from VllmConfig.additional_config["zero_interrupt_config"].

        For DEGRADE/RECOVER:
            - Read new_tp, new_dp from config
            - Update vllm_config.parallel_config to use new tp/dp

        For PD_REBUILD:
            - Execute strategy to rebuild KV chain (worker not restarted)
        """
        try:
            vllm_config = self._vllm_config
            if vllm_config is None:
                logger.debug("No VllmConfig available")
                return

            # Read from additional_config
            additional_config = getattr(vllm_config, "additional_config", {})
            zi_config = additional_config.get("zero_interrupt_config", {})
            if not zi_config:
                logger.debug("No zero_interrupt_config in vllm_config.additional_config")
                return

            # Parse config
            deploy_type = zi_config.get("deploy_type", "DEGRADE")
            new_tp = zi_config.get("new_tp")
            new_dp = zi_config.get("new_dp")

            logger.info(f"Worker {self.rank} applying zero interrupt config: "
                        f"deploy_type={deploy_type}, new_tp={new_tp}, new_dp={new_dp}")

            # DEGRADE/RECOVER: Update parallel config with new tp/dp
            if deploy_type in (DeployType.DEGRADE.value, DeployType.RECOVER.value):
                self._apply_parallel_config(vllm_config, new_tp, new_dp)
                return

            # Note: PD_REBUILD is handled via RPC call to update_kv_connector_for_pd
            # instead of at worker startup, to ensure all instances coordinate properly

            logger.warning(f"Worker {self.rank}: Unknown deploy_type={deploy_type}")

        except Exception as e:
            logger.warning(f"Error applying zero interrupt config: {e}")

    def _apply_parallel_config(self, vllm_config, new_tp: int | None, new_dp: int | None) -> None:
        """Apply new tp/dp to parallel config.

        For DEGRADE/RECOVER scenarios, the worker restarts with new parallel config.
        This updates the parallel_config to use the new tp/dp values.

        Args:
            vllm_config: VllmConfig instance
            new_tp: New tensor parallel size
            new_dp: New data parallel size
        """
        if new_tp is None and new_dp is None:
            logger.debug("No new tp/dp to apply")
            return

        parallel_config = vllm_config.parallel_config

        if new_tp is not None:
            old_tp = parallel_config.tensor_parallel_size
            parallel_config.tensor_parallel_size = new_tp
            logger.info(f"Worker {self.rank} updated tp: {old_tp} -> {new_tp}")

        if new_dp is not None:
            old_dp = parallel_config.data_parallel_size
            parallel_config.data_parallel_size = new_dp
            logger.info(f"Worker {self.rank} updated dp: {old_dp} -> {new_dp}")

    def update_kv_connector_for_pd(self, strategy: DeployStrategy) -> None:
        """Update KV connector for healthy instances in PD rebuild scenario.

        This is called via RPC on healthy instances to update their
        KV connector state without restarting workers.

        Args:
            strategy: PD_REBUILD deployment strategy
        """
        logger.info(f"Worker {self.rank} updating KV connector for PD rebuild")

        try:
            # Execute PD rebuild through strategy handler
            success = self._strategy_handler.execute_strategy(strategy)

            if success:
                logger.info(f"Worker {self.rank} KV connector update successful")
            else:
                logger.error(f"Worker {self.rank} KV connector update failed")

        except Exception as e:
            logger.error(f"Worker {self.rank} KV connector update error: {e}")

    def _get_zero_interrupt_config(self) -> dict | None:
        """Get zero interrupt config from VllmConfig.

        Returns:
            Zero interrupt config dict or None
        """
        try:
            vllm_config = getattr(self.worker, "vllm_config", None)
            if vllm_config is None:
                return None

            additional_config = getattr(vllm_config, "additional_config", {})
            return additional_config.get("zero_interrupt_config")
        except Exception:
            return None

    def handle_output(self, output: Any) -> None:
        """Handle output from worker method."""
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)
