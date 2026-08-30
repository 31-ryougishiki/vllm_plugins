#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Type definitions for ITS plugin."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutorState(Enum):
    """Executor state enumeration."""

    RUNNING = "RUNNING"
    WAITING_STRATEGY = "WAITING_STRATEGY"
    EXECUTING_STRATEGY = "EXECUTING_STRATEGY"
    RECOVERING = "RECOVERING"
    STOPPED = "STOPPED"
    EXECUTING_STRATEGY_FAILED = "EXECUTING_STRATEGY_FAILED"


class DeployType(Enum):
    """Deployment strategy type enumeration."""

    STOP = "STOP"
    DEGRADE = "DEGRADE"
    RECOVER = "RECOVER"
    PD_REBUILD = "PD_REBUILD"


class DeployState(Enum):
    """Deployment result state enumeration."""

    EXECUTOR_DEPLOY_SUCCESS = "EXECUTOR_DEPLOY_SUCCESS"
    EXECUTOR_STOP = "EXECUTOR_STOP"
    EXECUTOR_DEPLOY_FAIL = "EXECUTOR_DEPLOY_FAIL"


@dataclass
class UpdateEngineInfo:
    """ update engine id when pd rebuild strategy is executed, used for updating executor_id """
    
    orig_engine_id: str
    new_engine_id: str


@dataclass
class NPUInfo:
    """NPU device information."""

    npu_id: int
    device_ip: str
    rank_id: str
    healthy: bool


@dataclass
class ServerInfo:
    """Server information for deployment."""

    server_id: str
    host_ip: str
    device: list[NPUInfo]


@dataclass
class EngineNPUHealthyState:
    """Instance NPU healthy state information."""

    server_count: str
    status: str
    version: str
    server_list: list[ServerInfo] = field(default_factory=list)


@dataclass
class EngineParallelConfig:
    """Instance parallel configuration."""

    dp: int
    tp: int
    executor_id: str = None
    data_parallel_rank: int | None = None
    enable_expert_parallel: bool = False
    new_dp: int | None = None
    new_tp: int | None = None
    tp_asymmetric_shardings: list[int] | None = None
    """Per-TP-rank sharding ratios for the new_tp ranks, e.g. [2, 1, 1].
    None means uniform sharding. This is the vllm_plugins-native config."""


@dataclass
class ModelInfo:
    """
    模型信息 - 用于HBM估算
    """
    hidden_size: int = 0  # 隐藏层大小
    num_attention_heads: int = 0  # Attention头数
    num_layers: int = 0  # 层数
    expert_num: int = 0  # 路由专家总数。  取config的n_routed_experts
    moe_intermediate_size: int = 0  # MoE中间层大小
    intermediate_size: int = 0  # FFN中间层维度
    architectures: str = None  # 取config的architectures
    vocab_size: int = 0  # 词表大小
    num_key_value_heads: int = 0  # GQA时的KV头数，默认 n_q // 4
    tie_word_embeddings: bool = False  # embedding输出共享系数    config的tie_word_embeddings  false是1 true是2
    max_model_len: int = 0
    kv_quantize: str = None  # 取config的mla_quantize
    weight_quantize: str = None  # 取config的quantize


@dataclass
class DeployStrategy:
    """Deployment strategy data class."""

    deploy_type: DeployType
    executor_id: str
    engine_parallel_config: list[EngineParallelConfig]
    engine_npu_healthy_state: list[EngineNPUHealthyState]
    update_engine_info: UpdateEngineInfo | None = None
    strategy_generation: str | None = None
    """Idempotency/rendezvous generation set by the triggering client.

    Every executor in one trigger wave receives the same value.  A different
    value forces executors that already reached the target topology to join a
    fresh full-restart barrier instead of short-circuiting, which lets an
    executor that missed the previous partial trigger wave recover together
    with its peers.  ``None`` preserves the legacy topology-only idempotency.
    """
    barrier_master_port: int | None = None
    """Fresh TCPStore rendezvous port selected by the triggering client.

    Present when ``strategy_generation`` is present: every executor in the
    trigger wave receives the same pre-checked free port, so a full-restart
    barrier no longer depends on each executor's local barrier-port pool
    state (which can drift after a partially delivered wave).  ``None``
    preserves the legacy local-pool selection for decision-center payloads.
    """


@dataclass
class InitExecutorStateRequest:
    """初始化执行器状态请求数据。

    用于上报执行器初始状态到决策中心，包含服务标识、并行配置、NPU 信息等。

    Attributes:
        service_id: 服务实例唯一标识，从环境变量 VLLM_SERVICE_ID 获取
        model_name: 模型名称
        engine_id: KV Cache Engine ID，用于 PD 分离场景，默认为 0
        engine_parallel_config: Engine 并行配置（dp/tp/enable_expert_parallel）
        model_info: 模型信息
        engine_pd_role: PD 角色，从 kv_transfer_config 获取
        executor_state: 执行器状态（RUNNING/WAITING_STRATEGY 等）
        executor_ip_port: 执行器地址（data_parallel_address:data_parallel_rpc_port）
        data_parallel_ip_port: DP 地址
        data_parallel_rank: DP 组内 rank
        node_ip: 节点 IP 地址
        node_hbm: 节点 HBM 总容量（单位：Byte），用于决策中心估算 HBM 使用率
        npu_id: NPU 物理 ID 列表，从 ASCEND_RT_VISIBLE_DEVICES 环境变量获取
        npu_rank_id: NPU 全局 rank 列表，从 0 开始编号
        npu_healthy: NPU 健康状态列表，True 表示健康
    """
    service_id: str
    model_name: str
    engine_id: str
    engine_parallel_config: EngineParallelConfig
    model_info: ModelInfo
    engine_pd_role: str | None
    executor_state: str
    executor_ip_port: str
    data_parallel_ip_port: str
    data_parallel_rank: int
    node_ip: str
    node_hbm: int
    node_type: str
    npu_id: list[str]
    npu_rank_id: list[str]
    npu_healthy: list[bool]


@dataclass
class ZeroInterruptConfig:
    """Configuration for zero-interruption inference.

    This config is passed to workers via VllmConfig to enable
    zero-interruption deployment without RPC calls.
    """

    enabled: bool = False
    """Whether zero-interruption is enabled."""

    executor_id: str = "0"
    """Executor ID for this instance."""

    deploy_type: str = "DEGRADE"
    """Current deployment strategy type."""

    tp: int = 1
    """Tensor parallelism degree."""

    dp: int = 1
    """Data parallelism degree."""

    new_tp: int | None = None
    """New TP after scale down (if applicable)."""

    new_dp: int | None = None
    """New DP after scale down (if applicable)."""

    healthy_npu_ids: list[int] = field(default_factory=list)
    """List of healthy NPU device IDs."""

    worker_ranks: list[int] = field(default_factory=list)
    """Ranks of workers managed by this executor."""

    is_healthy_worker: bool = True
    """Whether this worker should stay healthy."""
