# ITSMultiprocExecutor 插件开发详细设计文档


## 文档信息

| 属性 | 内容 |
|------|------|
| 版本 | 11.0 |
| 状态 | 已完善 |
| 创建日期 | 2025-05-13 |
| 更新日期 | 2026-06-08 |

---

## 1 概述

### 1.1 背景与目的

本文档描述基于vLLM 的插件实现方案，源码地址：(VLLM的代码地址：D:\27.0\推理引擎\vllm-releases-v0.18.0\vllm-releases-v0.18.0，vllm-ascend的代码地址：D:\27.0\推理引擎\vllm-releases-v0.18.0\vllm-ascend-releases-v0.18.0)
，该插件是零中断推理功能的核心组件，负责实现：
- Worker多进程管理增强
- 扩缩容策略的接收与执行
- 与决策中心的通信机制
- 故障场景下的服务保持

### 1.2 设计目标

1. **故障自愈**：当Worker进程发生故障时，保持进程不退出，维持有限的接口响应能力
2. **策略执行**：接收决策中心下发的扩缩容策略，控制Worker执行相应策略
3. **状态同步**：定期上报Executor状态至决策中心
4. **平滑恢复**：支持扩缩容后Worker的平滑恢复

### 1.3 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         整体架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌─────────────────────────────────────┐ │
│  │ DevicePlugin │      │           Decision Center            │ │
│  │  (硬件故障   │─────▶│  - 故障过滤                          │ │
│  │   上报)      │      │  - 状态监控                          │ │
│  └──────────────┘      │  - 策略寻优                          │ │
│                        │  - 策略下发                          │ │
│                        └─────────────────────────────────────┘ │
│                                      │                           │
│                                      ▼ (HTTP)                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              MultiprocExecutor@ITS                        │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │ │
│  │  │ Health     │ │ Strategy   │ │ HTTP       │            │ │
│  │  │ Monitor    │ │ Sync       │ │ Server     │            │ │
│  │  └────────────┘ └────────────┘ └────────────┘            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                      │                           │
│                              ┌───────┴───────┐                  │
│                              ▼               ▼                  │
│  ┌────────────────┐   ┌────────────────┐                        │
│  │ WorkerProc@ITS │   │ WorkerProc@ITS │   ... (多Worker)      │
│  │  (主Worker)    │   │  (备Worker)    │                        │
│  └────────────────┘   └────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2 核心组件设计

### 2.1 组件继承关系

```
Executor (ABC, vllm.v1.executor.abstract)
    │
    └── MultiprocExecutor (vllm.v1.executor.multiproc_executor)
            │
            └── AscendMultiprocExecutor (vllm_ascend.patch)
                    │
                    └── ITSMultiprocExecutor (本插件实现)


WorkerProc (vllm.v1.executor.multiproc_executor)
    │
    └── AscendWorkerProc (vllm_ascend.patch)
            │
            └── ITSNPUWorker (本插件实现)
```

### 2.2 类图

```
┌─────────────────────────────────────────────────────────────────┐
│                     ITSMultiprocExecutor                        │
├─────────────────────────────────────────────────────────────────┤
│ - wait_new_deployment: threading.Event                         │
│ - recv_new_deployment: threading.Event                         │
│ - current_strategy: DeployStrategy                             │
│ - executor_state: ExecutorState                                │
│ - _health_monitor: ITSHealthMonitor                            │
│ - _http_server: ITSHttpServer                                  │
│ - _strategy_sync_thread: StrategySyncThread                    │
├─────────────────────────────────────────────────────────────────┤
│ + execute_deploy_strategy(strategy): bool                      │
│ + handle_new_deployment(): bool                                │
│ + restart_workers_with_strategy(): bool                        │
│ + _cleanup_and_restart_workers(): None                         │
│ + _restart_all_workers(): None                                 │
│ + _init_workers(): None                                        │
│ + _cleanup_message_queues(): None                              │
│ + _setup_message_queues(): None                                │
│ + _get_engine_parallel_config(...): tuple                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ creates
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ITSNPUWorker                               │
├─────────────────────────────────────────────────────────────────┤
│ - _current_strategy: DeployStrategy                            │
│ - _strategy_handler: StrategyHandler                           │
├─────────────────────────────────────────────────────────────────┤
│ + update_kv_connector_for_pd(strategy): None  # PD重建链入口   │
└─────────────────────────────────────────────────────────────────┘
```

> 注意：Worker 主要功能是 **PD 重建链**（在线更新 KV-Cache 传输链），其他策略（DEGRADE/RECOVER/STOP）由 Executor 重启 Worker 处理。

### 2.3 策略执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                 execute_deploy_strategy()                       │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: _update_vllm_config_with_strategy(strategy)           │
│    - 将 strategy 存入 VllmConfig.additional_config             │
│  Step 2: 根据 deploy_type 调用对应方法                          │
│    - STOP: self.shutdown() → 关闭 Executor                     │
│    - DEGRADE: _execute_degrade_strategy()                      │
│    - RECOVER: _execute_recover_strategy()                      │
│    - PD_REBUILD: _execute_pd_rebuild_strategy()                │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│   STOP           │ │ DEGRADE/RECOVER  │ │   PD_REBUILD         │
├──────────────────┤ ├──────────────────┤ ├──────────────────────┤
│ shutdown()       │ │ 1. 设置状态      │ │ 1. 判断实例健康状态   │
│                  │ │ 2. 重启Workers  │ │    _is_fault_instance│
│                  │ │ 3. 设置RUNNING  │ │ 2a.故障→重启Workers  │
│                  │ │ 4. 上报状态     │ │ 2b.健康→RPC更新KV    │
└──────────────────┘ └──────────────────┘ └──────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              _cleanup_and_restart_workers()                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: _cleanup_message_queues_and_workers()                 │
│    - 关闭 death_writer                                          │
│    - 等待 worker 进程退出                                        │
│    - 清理消息队列                                               │
│    - 清理 worker 引用                                           │
│  Step 2: _update_vllm_config_for_restart()                     │
│  Step 3: _init_workers()                                        │
│    - 重建分布式环境                                             │
│    - 计算新的并行配置                                           │
│    - 创建并启动 workers                                         │
│  Step 5: _setup_message_queues()                               │
│  Step 6: 重置健康监控状态                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 非对称并行配置计算

```
_get_engine_parallel_config(world_size, local_world_size, node_rank_within_dp, strategy)
│
├─ 遍历 engine_parallel_config_list
│   └─ 匹配 executor_id，找到对应的 new_tp/new_dp
│
├─ 计算新的 world_size:
│   new_world_size = new_tp * original_world_size / original_tp
│
├─ 计算新的 local_world_size:
│   new_local_world_size = new_world_size / (nnodes * new_dp * data_parallel_size_local)
│
└─ 返回: (world_size, local_world_size, global_start_rank, node_rank_within_dp)
```

---

## 3 接口设计

### 3.1 决策中心 -> Executor 接口

#### 3.1.1 初始化状态上报

```http
POST /api/v1/decision_center/init_executor_state
Authorization: Bearer <token>
Content-Type: application/json

{
    "service_id": "550e8400-e29b-41d4-a716-446655440000",  // 服务实例唯一标识
    "engine_id": "0",                                         // Engine ID (KV-Cache engine)，字符串类型
    "model_name": "Qwen/Qwen2-7B",                           // 模型名称
    "engine_parallel_config": {                             // Engine 并行配置
        "dp": 2,
        "tp": 4,
        "enable_expert_parallel": true
    },
    "model_info": {                                         // 模型信息（用于HBM估算和策略寻优）
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_layers": 28,
        "expert_num": 8,
        "moe_intermediate_size": 18944,
        "intermediate_size": 18944,
        "architectures": "Qwen2MoeForCausalLM",
        "vocab_size": 151936,
        "num_key_value_heads": 2,
        "tie_word_embeddings": false,
        "max_model_len": 32768,
        "kv_quantize": "fp8_e4m3",
        "weight_quantize": "fp8"
    },
    "engine_pd_role": "kv_producer",     // PD角色: 从 kv_transfer_config 获取 (可空)
    "executor_state": "RUNNING",      // Executor状态
    "executor_ip_port": "192.168.1.10:29500",  // Executor IP:端口
    "data_parallel_ip_port": "192.168.1.10:29501",  // DP地址
    "data_parallel_rank": 0,          // DP组内 rank
    "node_ip": "192.168.1.10",        // 节点IP
    "node_hbm": 68828198400,         // 节点HBM总容量（Byte），用于HBM使用率估算
    "npu_id": ["0", "1", "2", "3"],   // NPU 物理ID列表
    "npu_rank_id": ["0", "1", "2", "3"], // NPU Rank ID列表
    "npu_healthy": [true, true, true, true]  // NPU健康状态
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 | 数据来源 |
|------|------|------|------|----------|
| service_id | string | 是 | 服务实例唯一标识 | 环境变量 VLLM_SERVICE_ID 或 UUID |
| engine_id | str | 是 | KV-Cache Engine ID | 见下方 engine_id 格式说明 |
| model_name | string | 是 | 模型名称 | vllm_config.model_config.served_model_name |
| engine_parallel_config | object | 是 | Engine 并行配置 | vllm_config.parallel_config |
| model_info | object | 是 | 模型信息 | 见下方 model_info 字段说明 |
| engine_pd_role | string | 否 | PD 角色 | kv_transfer_config.kv_role |
| executor_state | string | 是 | Executor 状态 | ExecutorState.RUNNING |
| executor_ip_port | string | 是 | Executor 地址 | data_parallel_address:data_parallel_rpc_port |
| data_parallel_ip_port | string | 是 | DP 地址 | data_parallel_address:data_parallel_rpc_port |
| data_parallel_rank | int | 是 | DP 组内 rank | parallel_config.node_rank_within_dp |
| node_ip | string | 是 | 节点 IP | get_ip() |
| node_hbm | int | 是 | 节点 HBM 总容量（Byte） | 从 NPU 驱动获取，用于 HBM 使用率估算 |
| npu_id | list[str] | 是 | NPU 物理 ID 列表 | ASCEND_RT_VISIBLE_DEVICES 环境变量 |
| npu_rank_id | list[str] | 是 | NPU Rank ID 列表 | 本地索引 0~n-1 |
| npu_healthy | list[bool] | 是 | NPU 健康状态 | 默认全为 true |

**model_info 字段说明：**

| 字段 | 类型 | 说明 | 数据来源 |
|------|------|------|----------|
| hidden_size | int | 隐藏层大小 | hf_config.hidden_size / config.json |
| num_attention_heads | int | Attention 头数 | hf_config.num_attention_heads / config.json |
| num_layers | int | 层数 | hf_config.num_hidden_layers / config.json |
| expert_num | int | 路由专家总数 | hf_config.n_routed_experts / config.json |
| moe_intermediate_size | int | MoE 中间层大小 | hf_config.moe_intermediate_size / config.json |
| intermediate_size | int | FFN 中间层维度 | hf_config.intermediate_size / config.json |
| architectures | string | 模型架构 | hf_config.architectures[0] / config.json |
| vocab_size | int | 词表大小 | hf_config.vocab_size / config.json |
| num_key_value_heads | int | GQA 时的 KV 头数 | hf_config.num_key_value_heads / config.json |
| tie_word_embeddings | bool | embedding 输出共享系数 | hf_config.tie_word_embeddings / config.json |
| max_model_len | int | 最大模型长度 | model_config.max_model_len / config.json.max_position_embeddings |
| kv_quantize | string | KV 量化方式 | model_config.mla_quantize / kv_transfer_config.kv_scale_dtype |
| weight_quantize | string | 权重量化方式 | model_config.quantize |

**engine_id 格式说明：**

| 模型类型 | engine_id 格式 | 说明 |
|----------|---------------|------|
| MoE 模型 | `{kv_transfer_config.engine_id}` | 直接使用 kv_transfer_config 中的 engine_id |
| 非 MoE 模型 | `{data_parallel_index}_{engine_id}` | 拼接 data_parallel_index 前缀，用于区分不同 DP 实例 |

**示例：**
- MoE 模型（TP=4, DP=2, engine_id=0）：`engine_id = "0"`
- 非 MoE 模型（TP=4, DP=2, dp_index=1, engine_id=0）：`engine_id = "1_0"`

#### 3.1.2 策略下发

```http
POST /api/v1/executor/deploy
Content-Type: application/json

{
    "deploy_type": "DEGRADE",         // 策略类型: STOP/DEGRADE/RECOVER/PD_REBUILD
    "executor_id": 0,                 // Executor ID
    "engine_parallel_config": [     // 实例非对称并行策略
        {
            "executor_id": 0,
            "dp": 2,
            "tp": 4,
            "new_dp": 2,
            "new_tp": 2,              // 缩容后TP
            "enable_expert_parallel": true
        }
    ],
    "engine_npu_healthy_state": [
       {   // 实例硬件健康分布
           "server_count": "1",
           "status": "completed",
           "version": "1.2",
           "server_list": [
               {
                   "device": [
                       {"npu_id": 0, "device_ip": "192.168.1.10", "rank_id": "0", "healthy": true},
                       {"npu_id": 1, "device_ip": "192.168.1.10", "rank_id": "1", "healthy": false}
                   ],
                   "host_ip": "192.168.1.20",
                   "server_id": "1"
               }
           ]
       }
    ]
}
```

#### 3.1.3 策略执行结果上报

```http
POST /api/v1/decision_center/report_deploy_status
Authorization: Bearer <token>
Content-Type: application/json

{
    "executor_id": 2,
    "deploy_state": "EXECUTOR_DEPLOY_SUCCESS"  // EXECUTOR_DEPLOY_SUCCESS/EXECUTOR_STOP/EXECUTOR_DEPLOY_FAIL
}
```

### 3.2 策略类型定义

| 策略类型 | 说明 | 场景 |
|----------|------|------|
| STOP | 停止实例 | 策略执行失败，超时 |
| DEGRADE | 缩容 | NPU故障，缩减TP/DP |
| RECOVER | 恢复 | 故障NPU恢复，扩展TP/PD |
| PD_REBUILD | P/D重建链 | P/D分离场景下的链重建 |

### 3.3 并行策略结构

```json
{
    "tp": 4,                    // 当前TP大小
    "dp": 2,                    // 当前DP大小
    "new_tp": 2,                // 缩容后TP (可选)
    "new_dp": 2,                // 缩容后DP (可选)
    "enable_expert_parallel": true  // 是否开启EP
}
```

### 3.4 非对称并行配置

在多实例部署场景下，不同实例可以有不同的 TP/DP 配置：

```json
[
    {
        "executor_id": 0,
        "dp": 2,
        "tp": 4,
        "data_parallel_rank": 0,
        "new_dp": 2,
        "new_tp": 2
    },
    {
        "executor_id": 1,
        "dp": 2,
        "tp": 4,
        "data_parallel_rank": 1,
        "new_dp": 1,
        "new_tp": 4
    }
]
```

**非对称并行计算公式：**

```
# 1. 计算新的 world_size
world_size = new_tp * original_world_size / original_tp

# 2. 计算新的 local_world_size
# local_world_size = TP × PP × PCP / nnodes_within_dp
# nnodes_within_dp = nnodes * data_parallel_size_local / data_parallel_size
local_world_size = world_size / (nnodes * new_dp * data_parallel_size_local)

# 3. 计算 global_start_rank（当前实例的全局起始 rank）
# = 累加所有 data_parallel_rank 小于当前实例的 local_world_size
global_start_rank = sum(previous_instances_local_world_size)
```

**计算示例（节点上有 2 个实例）：**

```
┌─────────────────────────────────────────────────────────────┐
│  实例 0 (executor_id=0, data_parallel_rank=0)               │
│  - new_tp=2, new_dp=2, local_world_size=2                   │
│  - global_start_rank = 0                                    │
├─────────────────────────────────────────────────────────────┤
│  实例 1 (executor_id=1, data_parallel_rank=1)               │
│  - new_tp=4, new_dp=1, local_world_size=2                   │
│  - global_start_rank = 0 + 2 = 2                            │
│    (累加 data_parallel_rank < 1 的实例 local_world_size)   │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 PD建链信息

```json
// 每个元素是rank-id: 是待建链P实例的Worker rank-id
// 假设原本4个卡，rank=2故障了
[0, 1, 3]
```

---

## 4 核心流程设计

### 4.1 EngineCore Patch 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    EngineCore.run_busy_loop                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌─────────────────────────────────────────┐
         │  1. 检查 RecvNewDeployment 信号量        │
         └─────────────────────────────────────────┘
                    │                 │
                   是                否
                    │                 │
                    ▼                 ▼
         ┌────────────────┐  ┌─────────────────────────┐
         │  执行策略        │  │ 2. 检查 WaitNewDeployment│
         │  handle_new_   │  └─────────────────────────┘
         │   deployment   │        │                │
         └────────────────┘       是                否
                                  │                │
                                  ▼                ▼
                          ┌────────────┐    ┌──────────┐
                          │ 3. 等待    │    │ 继续正常   │
                          │ RecvNew    │    │ 提供服务  │
                          │ (超时30s)  │    └──────────┘
                          └────────────┘
                              │
                    ┌─────────┴─────────┐
                   超时               未超时
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐  ┌────────────────┐
          │ 以当前策略重启     │  │ 执行策略       │
          │ worker进程       │  │ handle_new_    │
          └─────────────────┘  │ deployment     │
                     │
                                    │
                     ▼              ▼
                    ┌──────────────────┐
                    │ 恢复 Worker      │
                    │ Monitor 线程     │
                    └──────────────────┘
```

#### 4.1.1 两层空转过滤机制

为了彻底排除空转 Executor（dp=0/tp=0），实现了两层过滤机制：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           两层空转过滤架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  第一层：Engine侧过滤 (engine_core_patch.py)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  patched_handle_client_request()                                     │   │
│  │  - 当 world_size==0 时丢弃 ADD 请求                                   │   │
│  │  - 防止请求堆积在空转 scheduler                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                        │
│                                      ▼                                        │
│  第二层：Client侧过滤 (core_client_patch.py)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  patched_get_core_engine_for_request()                               │   │
│  │  - 跳过被标记为空转的 engine                                          │   │
│  │  - 从候选列表中排除 idle_engines                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                        │
│                                      ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  patched_process_engine_outputs()                                    │   │
│  │  - 检测 engine 负载分数 (999999, 999999)                              │   │
│  │  - 标记空转 engine                                                    │   │
│  │  - 检测恢复信号 (0, 0)                                                │   │
│  │  - 移除恢复 engine 的 idle 标记                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键设计点：**

1. **信号机制**：
   - `(999999, 999999)` - 空转标记，通过 `output_queue` 发送
   - `(0, 0)` - 恢复标记，告知客户端移除 idle 标记

2. **空转通知发送**（`_send_idle_notification`）：
   - 当 `world_size==0` 时，在 `patched_handle_shutdown` 中发送
   - 通过 `output_queue.put_nowait((-1, outputs))` 发送
   - 发送后设置 `_idle_notification_sent = True` 避免重复发送

3. **Client 端检测**（`patched_process_engine_outputs`）：
   - 当 `num_waiting >= 999999` 且 `num_running >= 999999` 时标记为空转
   - 当 `num_waiting == 0` 且 `num_running == 0` 时移除 idle 标记

4. **策略执行后引擎不退出**：
   - `patched_handle_shutdown` 中增加 `EngineShutdownState.RUNNING` 检查
   - 当 `shutdown_state == RUNNING` 时返回 `True`，保持引擎运行

#### 4.1.2 busy_loop 触发机制

为了在策略下发时立即触发 EngineCore 的 `patched_handle_shutdown` 执行，采用了以下机制：

```
决策中心下发策略
        │
        ▼
executor._trigger_busy_loop_callback()
        │
        ├─▶ process_input_queue_block = False  (跳过阻塞等待)
        │
        └─▶ input_queue.put_nowait(WAKEUP)     (唤醒阻塞的get())
        │
        ▼
busy_loop 退出 _process_input_queue
        │
        ▼
patched_handle_shutdown() 被调用
        │
        ▼
检测到信号，执行部署策略
```

#### 4.1.2 部署期间暂停推理机制

在 Worker 重启期间，需要阻止新的推理请求执行，避免 `RuntimeError: cancelled` 错误：

1. **设置暂停标志**：`engine_core._paused_for_restart = True`
2. **patched_has_work**：`has_work()` 检查该标志，返回 `False` 跳过推理
3. **执行策略**：直接执行部署策略，不等待进行中的请求（因为故障时请求已在失败中）
4. **重启 Worker**：执行 `_cleanup_and_restart_workers()`
5. **恢复**：`engine_core._paused_for_restart = False`

#### 4.1.3 process_input_queue_block 状态管理

- **部署期间**：`process_input_queue_block = False`，`_process_input_queue` 非阻塞执行
- **恢复后**：`process_input_queue_block = True`（恢复到阻塞模式，只有请求来时才执行推理）
- **部署信号触发**：通过 WAKEUP 信号唤醒阻塞循环，检测到部署信号后执行策略

#### 4.1.4 JSON 序列化问题处理

在 Worker 重启过程中，vLLM 的 `compute_hash()` 方法会对 `VllmConfig.additional_config` 进行 JSON 序列化。由于 `DeployStrategy` 是包含枚举和嵌套对象的 dataclass，直接存储会导致序列化失败。

**解决方案**：
1. 使用 `dataclasses.asdict()` 将 `DeployStrategy` 转换为字典
2. 递归将枚举值转换为字符串（通过 `.value` 属性）

```python
from dataclasses import asdict
strategy_dict = asdict(self.current_strategy)
strategy_dict = self._convert_enums_to_values(strategy_dict)
self.vllm_config.additional_config["zero_interrupt_config"] = strategy_dict
```

#### 4.1.5 KV Cache 重新初始化

Worker 重启后，新的 Worker 进程缺少 `kv_cache_config` 属性（该属性在 `initialize_from_config` 中设置）。这会导致首次推理请求时出现 `AttributeError`。

**解决方案**：
在 Worker 重启后调用 `_reinitialize_kv_cache()`，该方法：
1. 调用 `get_kv_cache_specs()` 获取每个 worker 的 KV cache 规格
2. 调用 `determine_available_memory()` 获取可用内存
3. 调用 `get_kv_cache_configs()` 生成 KV cache 配置
4. 调用 `initialize_from_config()` 重新初始化 Worker

### 4.2 Worker 主要功能

Worker 端主要处理 **PD 重建链**（在线 P/D KV-Cache 传输链重建），不涉及 Worker 故障保持。

> 注意：故障保持（Fault Keep）由 Executor 层的 HealthMonitor 处理，通过重启 Worker 实现。

### 4.3 故障场景时序图

```
时间 ─────────────────────────────────────────────────────────────▶

决策中心                    Executor                       Worker
  │                            │                              │
  │                            │                              │
  │◀─── 健康检测 ─────────────│                              │
  │                            │                              │
  │                      Worker 进程退出                       │
  │                            │                              │
  │                            │─── HealthMonitor 检测 ──────▶│
  │                            │     (proc.is_alive() = False)│
  │                            │                              │
  │                            │─── ITSFailureCallback ──────▶│
  │                            │     (设置 executor_state =    │
  │                            │      WAITING_STRATEGY)        │
  │                            │                              │
  │                            │─── wait_new_deployment.set()─▶│
  │                            │                              │
  │─── 故障检测 ──────────────▶│                              │
  │                            │                              │
  │◀─── 策略寻优 ─────────────│                              │
  │                            │                              │
  │─── 下发策略 ──────────────▶│                              │
  │     (POST /deploy)         │                              │
  │                            │◀─── 执行策略 ──────────────│
  │                            │     (_cleanup_and_restart_   │
  │                            │      workers)                │
  │                            │                              │
  │◀─── 上报结果 ─────────────│                              │
  │     (EXECUTOR_DEPLOY_     │                              │
  │      SUCCESS)             │                              │
  │                            │                              │
```

### 4.4 超时场景时序图

**场景：Monitor 检测到 Worker 故障，等待决策中心下发策略，但策略超时未到**

```
时间 ─────────────────────────────────────────────────────────────▶

决策中心                    Executor                       Worker
  │                            │                              │
  │                            │◀─── HealthMonitor 检测 ──────▶│
  │                            │     (Worker 进程退出)         │
  │                            │                              │
  │                            │─── 设置 wait_new_deployment ──▶│
  │                            │     唤醒 busy_loop             │
  │                            │                              │
  │─── 健康检测 ──────────────▶│                              │
  │                            │                              │
  │◀─── 状态: WAITING_STRATEGY │                              │
  │                            │                              │
  │                       超时 (无策略下发)                    │
  │                            │                              │
  │                            │─── 关闭 death_writer ────────▶│
  │                            │     (信号 worker 退出)        │
  │                            │                              │
  │                            │◀─── Worker 退出 ──────────────│
  │                            │                              │
  │                            │─── 重启 workers (重试3次) ───▶│
  │                            │     (_restart_workers_with_  │
  │                            │      strategy)               │
  │                            │                              │
  │                            │◀─── 重启成功 ────────────────│
  │                            │                              │
  │◀─── 上报 DEPLOY_SUCCESS ──│                              │
  │     (EXECUTOR_DEPLOY_     │                              │
  │      SUCCESS, RUNNING)    │                              │
  │                            │                              │
  │                            │─── 回到正常服务 ────────────▶│
  │                            │                              │
```

**超时场景处理流程：**

```
Monitor 检测到 Worker 故障
        ↓
设置 wait_new_deployment，唤醒 busy_loop
        ↓
busy_loop 等待 recv_new_deployment（timeout=30s）
        ↓
├── 收到策略 → 执行策略 → 回到正常服务
│
└── 超时未收到策略
        ↓
    重启 Workers（最多重试3次）
        ↓
    ├── 重启成功 + 有策略 → 执行策略 → 回到正常服务
    ├── 重启成功 + 无策略 → 直接回到正常服务
    └── 重启失败（3次） → shutdown → 退出
```

### 4.5 Worker 重启流程

```
┌─────────────────────────────────────────────────────────────────┐
│              _cleanup_and_restart_workers()                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 0: _init_workers() 中从策略获取健康 NPU 列表              │
│    - 从 engine_npu_healthy_state 解析当前节点健康 NPU           │
│    - 设置 ASCEND_RT_VISIBLE_DEVICES 环境变量                    │
│  Step 1: _cleanup_message_queues_and_workers()                  │
│    - 关闭 death_writer 信号                                     │
│    - 等待 worker 进程退出                                        │
│    - 清理消息队列                                               │
│    - 清理 worker 引用                                           │
│  Step 2: _update_vllm_config_for_restart()                      │
│    - 将策略写入 additional_config                               │
│  Step 3: _init_workers()                                        │
│    - 重建分布式环境                                             │
│    - 计算新的并行配置 (TP/DP)                                   │
│    - 启动新 worker 进程（仅使用健康的 NPU）                     │
│  Step 4: 重置 health monitor 状态                               │
│  Step 5: _reinitialize_kv_cache()                               │
│    - 重新初始化 KV cache 配置                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 缓存清理详细设计

#### 4.6.1 Worker 重启前的缓存清理

在 Worker 重启前，通过 RPC 调用清理缓存：

```python
# Executor 端调用
self.collective_rpc("reset_encoder_cache", timeout=30)  # 清理 encoder cache
self.collective_rpc("reset_mm_cache", timeout=30)       # 清理 multimodal cache
```

**清理内容：**

| 缓存类型 | 调用方法 | 清理内容 |
|----------|----------|----------|
| Encoder Cache | `reset_encoder_cache()` | 清空 `encoder_outputs` 字典（GPU 端 vision embeddings） |
| Multimodal Cache | `reset_mm_cache()` | 清空多模态缓存，确保旧模型权重计算的 embeddings 不会被复用 |

**注意：** KV Cache 的清理通过 Worker 进程退出自然完成，Worker 进程终止后所有 GPU 内存（包括 KV Cache）会被自动释放。

#### 4.6.2 Worker 重启后的 KV Cache 重新初始化

Worker 重启后，需要重新初始化 KV Cache 配置：

```python
# Executor 端调用 _reinitialize_kv_cache()
def _reinitialize_kv_cache(self) -> None:
    # Step 1: 从 workers 获取 KV cache specs
    kv_cache_specs = self.get_kv_cache_specs()

    # Step 2: 计算可用内存
    available_memory = self.determine_available_memory()

    # Step 3: 生成 KV cache configs
    kv_cache_configs = get_kv_cache_configs(
        self.vllm_config, kv_cache_specs, available_memory
    )

    # Step 4: 生成 scheduler KV cache config
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)

    # Step 5: 更新 vllm_config 并初始化 workers
    self.vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    self.initialize_from_config(kv_cache_configs)
```

#### 4.6.3 清理失败处理

- **Encoder/MM Cache 清理失败**：抛出 `RuntimeError`，终止重启流程
- **KV Cache 重新初始化失败**：抛出 `Exception`，终止重启流程

---

## 5 故障处理设计

### 5.1 故障类型

| 故障事件名称 | 故障码 | 说明 |
|-------------|--------|------|
| 网口Link状态变化 | 0x81078603 | 硬件链路故障 |

### 5.2 异常处理

1. **策略执行失败**：
   - 使用决策中心最新下发的策略重试
   - 若连续失败，Executor 设置状态为 `EXECUTING_STRATEGY_FAILED`
   - 上报 `EXECUTOR_DEPLOY_FAIL` 至决策中心
   - Executor 不主动退出，等待进一步处理

2. **策略等待超时**（默认30秒，由 `VLLM_ITS_STRATEGY_TIMEOUT` 配置）：
   - 关闭 `death_writer` 信号通知 Worker 退出（而非 kill 命令）
   - 等待 Worker 进程终止
   - 使用当前策略重启 Workers（执行 DEGRADE/RECOVER 策略）
   - **重试逻辑**：若 Worker 重启失败，最多重试 3 次；3 次都失败则关闭 Executor 并退出
   - 上报 `EXECUTOR_DEPLOY_SUCCESS`（状态为 RUNNING）至决策中心

### 5.3 Worker 故障保持

1. **故障检测**：HealthMonitor 定期检测 worker 进程状态（`proc.is_alive()`）
2. **标志设置**：ITSFailureCallback 设置 `executor_state = WAITING_STRATEGY` 并调用 `wait_new_deployment.set()`
3. **状态设置**：设置 `executor_state = WAITING_STRATEGY` 等待决策中心下发策略
4. **策略执行**：收到策略后执行 `_cleanup_and_restart_workers()`，然后恢复正常

**注意**：当前实现检测的是 Worker 进程退出，而非 NPU 卡级别故障。NPU 卡故障的检测由决策中心负责。

---

## 6 数据结构设计

### 6.1 ExecutorState

```python
class ExecutorState(Enum):
    RUNNING = "RUNNING"                    # 正常运行
    WAITING_STRATEGY = "WAITING_STRATEGY"  # 等待策略
    EXECUTING_STRATEGY = "EXECUTING_STRATEGY"  # 执行策略中
    RECOVERING = "RECOVERING"              # 恢复中
    STOPPED = "STOPPED"                    # 已停止
    EXECUTING_STRATEGY_FAILED = "EXECUTING_STRATEGY_FAILED"  # 策略执行失败
```

### 6.2 DeployType

```python
class DeployType(Enum):
    STOP = "STOP"          # 停止实例
    DEGRADE = "DEGRADE"    # 缩容
    RECOVER = "RECOVER"    # 恢复
    PD_REBUILD = "PD_REBUILD"  # P/D重建链
```

### 6.3 DeployState

```python
class DeployState(Enum):
    EXECUTOR_DEPLOY_SUCCESS = "EXECUTOR_DEPLOY_SUCCESS"  # 部署成功
    EXECUTOR_STOP = "EXECUTOR_STOP"  # 实例停止
    EXECUTOR_DEPLOY_FAIL = "EXECUTOR_DEPLOY_FAIL"  # 部署失败
```

### 6.4 DeployStrategy

```python
@dataclass
class DeployStrategy:
    deploy_type: DeployType                         # 策略类型
    executor_id: str                                # Executor ID (字符串类型)
    engine_parallel_config: list[EngineParallelConfig]    # 并行配置
    engine_npu_healthy_state: list[EngineNPUHealthyState] # NPU健康状态
```

### 6.5 EngineParallelConfig

```python
@dataclass
class EngineParallelConfig:
    dp: int                      # DP大小
    tp: int                      # TP大小
    executor_id: str = None      # Executor ID
    data_parallel_rank: int | None = None  # DP组内 rank
    enable_expert_parallel: bool = False   # EP使能
    new_dp: int | None = None    # 缩容后DP
    new_tp: int | None = None    # 缩容后TP
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| dp | int | 当前 DP 大小 |
| tp | int | 当前 TP 大小 |
| executor_id | str | Executor ID (字符串类型) |
| data_parallel_rank | int \| None | DP组内的 rank (用于非对称并行) |
| enable_expert_parallel | bool | 是否开启 Expert Parallel |
| new_dp | int \| None | 缩容后 DP (可选) |
| new_tp | int \| None | 缩容后 TP (可选) |

---

## 7 目录结构

```
plugins/zero_interrup/
├── __init__.py                    # 插件入口，register() 注册插件
├── patch.py                       # 插件 Patch 类 (ZeroInterruptPluginPatch)
├── engine_core_patch.py           # EngineCore Monkey-Patch
├── common/
│   ├── __init__.py
│   ├── types.py                   # 类型定义
│   │   │                            # ExecutorState, DeployType, DeployStrategy,
│   │   │                            # EngineParallelConfig, ModelInfo,
│   │   │                            # InitExecutorStateRequest
│   └── constants.py               # 常量定义 (环境变量配置)
├── executor/
│   ├── __init__.py
│   ├── its_multiproc_executor.py  # ITSMultiprocExecutor 实现
│   ├── health_monitor.py          # 健康检测增强
│   │   │                            # ITSHealthMonitor, ITSFailureCallback
│   ├── strategy_sync.py           # 策略同步线程
│   ├── http_server.py             # HTTP 服务 (FastAPI + Uvicorn)
│   └── its_npu_worker.py          # Worker 增强实现 (通过 patch.py 注入)
├── worker/
│   ├── __init__.py
│   └── strategy_handler.py        # 策略处理 (_execute_pd_rebuild)
├── communication/
│   ├── __init__.py
│   └── decision_center_client.py  # 决策中心通信客户端
└── tests/
    ├── __init__.py
    ├── test_its_multiproc_executor.py
    ├── test_http_server.py
    ├── test_decision_center_client.py
    ├── test_health_monitor.py
    └── test_types.py
```

---

## 8 环境变量配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| VLLM_ITS_DECISION_CENTER_URL | http://127.0.0.1:8080 | 决策中心地址 |
| VLLM_ITS_DECISION_CENTER_TOKEN | - | 决策中心认证Token |
| VLLM_ITS_HTTP_SERVER_PORT_START | 8001 | HTTP服务起始端口 |
| VLLM_ITS_ENABLE_FAULT_KEEP | true | 启用故障保持 |
| VLLM_ITS_ENABLE_PD_REBUILD | true | 启用PD重建链 |
| VLLM_ITS_STRATEGY_TIMEOUT | 300 | 策略等待超时(秒) |
| VLLM_ITS_HEALTH_CHECK_INTERVAL | 5 | 健康检测间隔(秒) |
| VLLM_ITS_MAX_RETRY_COUNT | 3 | Worker重启最大重试次数 |
| VLLM_SERVICE_ID | auto-generated UUID | 服务实例唯一标识 |

---

## 9 功能列表

### 9.1 ITSMultiprocExecutor 功能

| 序号 | 功能名称 | 功能描述 | 状态 |
|------|----------|----------|------|
| F1 | 初始化状态上报 | Executor成功启动Worker后，上报状态至决策中心 | ✅ |
| F2 | 健康检测增强 | 修改FailureCallback执行流程，实现服务保持 | ✅ |
| F3 | 策略同步 | 通过monkey-patch实现EngineCore暂停/恢复 | ✅ |
| F4 | 策略部署接口 | 提供HTTP接口接收决策中心下发的扩缩容策略 | ✅ |
| F5 | 并行策略下发 | 通过VllmConfig传递策略给Worker | ✅ |
| F6 | Worker故障处理 | 超时kill所有Worker | ✅ |
| F7 | 组件状态重置 | 重置scheduler和BlockPool状态 | ✅ |
| F8 | Worker重启 | 完整的worker重启流程 (_restart_all_workers) | ✅ |
| F9 | KV-Cache清理 | 重启前清理 prefix cache | ✅ |
| F10 | 非对称并行支持 | 支持不同TP/DP的实例配置 | ✅ |
| F11 | 状态机管理 | RUNNING/WAITING_STRATEGY 状态切换 | ✅ |
| F12 | HealthMonitor改进 | 添加 _failure_handled 防止重复触发 | ✅ |
| F13 | 请求中止 | 执行策略前中止 scheduler 中未完成的请求，避免 KeyError | ✅ |
| F14 | 故障卡排除 | Worker 重启时从策略解析健康 NPU，设置 ASCEND_RT_VISIBLE_DEVICES 只使用健康卡 | ✅ |
| F15 | 空转过滤-Engine侧 | patch _handle_client_request，当 world_size==0 时丢弃 ADD 请求 | ✅ |
| F16 | 空转过滤-Client侧 | patch DPLBAsyncMPClient，跳过被标记为空转的 engine | ✅ |
| F17 | 空转通知机制 | 发送 (999999,999999) 负载分数通知客户端 engine 空转 | ✅ |
| F18 | 恢复通知机制 | 发送 (0,0) 负载分数通知客户端移除 idle 标记 | ✅ |

### 9.2 ITSNPUWorker 功能

| 序号 | 功能名称 | 功能描述 | 状态 |
|------|----------|----------|------|
| W1 | PD重建链 | 在线 P/D KV-Cache 传输链重建 (update_kv_connector_for_pd) | ✅ |
| W2 | 策略执行 | 从 VllmConfig 读取并执行扩缩容策略 (execute_deploy_from_config) | ✅ |

> 注：ITSNPUWorker 定义在 its_multiproc_executor.py 中，通过 patch.py 注入。故障保持、模型重建由 Executor 层处理。

### 9.3 StrategyHandler 功能

| 序号 | 功能名称 | 功能描述 | 状态 |
|------|----------|----------|------|
| S1 | STOP/DEGRADE/RECOVER | Worker重启由Executor处理（无操作） | ✅ |
| S2 | PD_REBUILD策略执行 | P/D分离场景下的KV-Cache链重建（在线） | ✅ |

---

## 10 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| 1.0 | 2025-05-12 | 初始版本 |
| 2.0 | 2025-05-13 | 根据设计文档完善，增加核心流程、接口定义、故障处理设计 |
| 3.0 | 2026-05-15 | 增加Worker重启实现、非对称并行支持、状态机管理、模型重建功能 |
| 4.0 | 2026-05-16 | 修复缓存清理逻辑：传递 reset_running_requests=True 和 reset_connector=True，增加 encoder_cache 清理，失败时抛出异常而非静默继续 |
| 5.0 | 2026-05-18 | 简化代码：删除冗余的 FaultKeepMixin 和 StrategyHandler 中无用的 STOP/DEGRADE/RECOVER 执行逻辑（Worker重启由Executor处理） |
| 6.0 | 2026-05-19 | 技术栈升级：将 HTTP 服务从 Flask 迁移到 FastAPI，使用 Pydantic 进行请求验证，Uvicorn 作为 ASGI 服务器 |
| 7.0 | 2026-05-20 | 代码优化：修复相对导入路径，统一使用相对导入；修复 EngineParallelConfig.executor_id 类型为 int 并添加默认值 0；EngineCore 改为 EngineCoreProc |
| 8.0 | 2026-05-28 | 1) 更新 engine_id 获取逻辑：非 MoE 模型拼接 data_parallel_index 前缀；2) Worker 重启增加重试机制：失败重试 3 次，3 次都失败则 shutdown 退出；3) 简化超时等待逻辑：移除 while 循环，改为单次 wait(timeout) 后重启；4) HealthMonitor 修复：_healthy_workers 在回调前更新防止重复触发；update_workers 时重置 _failure_handled |
| 9.0 | 2026-05-29 | 1) 移除执行策略前的等待请求逻辑：故障时无需等待，直接执行策略更快恢复；2) 清理 Event 标志：执行策略前清理 recv_new_deployment 和 wait_new_deployment 防止重复触发；3) 目录结构更新：移除 config.py、utils.py，新增 patch.py 和 tests/ 目录；4) 技术栈明确：HTTP 服务使用 FastAPI + Uvicorn；5) 环境变量更新：HTTP_SERVER_PORT→HTTP_SERVER_PORT_START，新增 VLLM_ITS_MAX_RETRY_COUNT、VLLM_SERVICE_ID |
| 10.0 | 2026-06-01 | 1) 修复 KeyError：执行策略前中止 scheduler 中未完成的请求；2) 排除故障卡：Worker 重启时从策略解析健康 NPU，设置 ASCEND_RT_VISIBLE_DEVICES 只使用健康卡 |
| 11.0 | 2026-06-08 | 1) 增加两层空转过滤机制：Engine侧丢弃ADD请求 + Client侧跳过idle engine；2) 增加空转/恢复通知机制：通过负载分数 (999999,999999)/(0,0) 通知客户端；3) 修复策略执行后引擎退出的问题：增加 EngineShutdownState.RUNNING 检查 |