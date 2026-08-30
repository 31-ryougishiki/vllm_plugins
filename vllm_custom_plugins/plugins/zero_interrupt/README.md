# Zero-interrup-plugin

Zero-interruption Inference ITS (Intelligent Transform Service) Plugin for vLLM on Ascend NPUs.

## Overview

This plugin provides fault tolerance and deployment strategy execution for vLLM running on Ascend NPUs. It enables:

- **Fault Keep**: Maintain service availability during worker process failures
- **Strategy Execution**: Execute deployment strategies (scale up/down) from decision center
- **State Reporting**: Report executor state to decision center
- **Smooth Recovery**: Recover service after deployment with minimal interruption

## Architecture

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

┌─────────────────────────────────────────────────────────────────┐
│                        EngineCoreProc                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ITSMultiprocExecutor                     │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │  │
│  │  │ 健康检测线程   │  │ 策略同步线程   │  │ HTTP服务   │  │  │
│  │  │ (Monitor)      │  │ (Strategy)     │  │ (FastAPI)  │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        WorkerProc进程                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ITSNPUWorker                           │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │  │
│  │  │ 故障保持       │  │ 策略接收       │  │ 状态恢复   │  │  │
│  │  │ (Try-Catch)    │  │ (RPC Handler)  │  │ (Recovery) │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design: Idle Engine Filtering

### Signal Mechanism

- `(999999, 999999)` - Idle marker sent via `output_queue` when `world_size==0`
- `(0, 0)` - Recovery marker sent when engine recovers from idle state

### How It Works

1. **Engine Side**: When `world_size==0`, `patched_handle_client_request` drops ADD requests
2. **Client Side**: When receiving `(999999, 999999)`, engine is marked as idle
3. **Routing**: `patched_get_core_engine_for_request` skips idle engines
4. **Recovery**: When receiving `(0, 0)`, idle mark is removed

## Installation

```bash
pip install -e plugins/zero_interrupt
```

## Configuration

The plugin can be configured via environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VLLM_ITS_DECISION_CENTER_URL` | http://127.0.0.1:8080 | Decision center URL |
| `VLLM_ITS_DECISION_CENTER_TOKEN` | - | Authentication token |
| `VLLM_ITS_HTTP_SERVER_PORT_START` | 8001 | HTTP server start port |
| `VLLM_ITS_HEALTH_CHECK_INTERVAL` | 5 | Health check interval (seconds) |
| `VLLM_ITS_STRATEGY_TIMEOUT` | 300 | Strategy execution timeout (seconds) |
| `VLLM_ITS_MAX_RETRY_COUNT` | 3 | Maximum retry attempts |
| `VLLM_ITS_ENABLE_FAULT_KEEP` | true | Enable fault keep mode |
| `VLLM_ITS_ENABLE_PD_REBUILD` | true | Enable PD chain rebuild |
| `VLLM_SERVICE_ID` | auto-generated UUID | Service instance ID |
| `VLLM_ITS_DEEPSEEK_V4` | 0 | When `1`, all conflicting patches/whole-file replacements switch to the `zero_interrupt/deepseekv4/` DeepSeek-V4 implementation; default `0` uses the main 0829 implementation |

> **Dual-family layout**: the main directory under
> `plugins/zero_interrupt/` is the merged `vllm_plugins_0829` implementation.
> The `plugins/zero_interrupt/deepseekv4/` directory is the self-contained
> `vllm_plugins` DeepSeek-V4 implementation. The runtime dispatcher
> (`zero_interrupt/patch.py`) and `setup.py` both read
> `VLLM_ITS_DEEPSEEK_V4` and select the same family, so runtime patches and
> file replacements stay consistent.


## Usage

### Basic Usage

插件通过 vLLM 的 `VLLM_CUSTOM_PATCHES` 环境变量自动加载：

```bash
# 启用 zero_interrupt 插件
export VLLM_CUSTOM_PATCHES=zero_interrupt

# 启动 vLLM（插件会自动应用）
python -m vllm.entrypoints.openai.api_server ...
```

或者通过代码加载：

```python
# 在创建 LLM 引擎之前应用插件
from vllm_custom_plugins.plugins.zero_interrup.patch import apply

apply()

# 现在正常创建 LLM 引擎
from vllm import LLM

llm = LLM(
    model="your-model",
    tensor_parallel_size=4,
    # ... other config
)
```

### API Endpoints

The plugin exposes the following HTTP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/executor/deploy` | POST | Receive deployment strategy |
| `/api/v1/executor/status` | GET | Get executor status |

### Deployment Strategy Format

```json
{
    "deploy_type": "DEGRADE",
    "executor_id": "0",
    "engine_parallel_config": [
        {
            "executor_id": "0",
            "dp": 2,
            "tp": 4,
            "enable_expert_parallel": false,
            "new_dp": 2,
            "new_tp": 2
        }
    ],
    "engine_npu_healthy_state": [{
        "server_count": "1",
        "status": "completed",
        "version": "1.0",
        "server_list": [
            {
                "server_id": "server-1",
                "host_ip": "192.168.1.10",
                "device": [
                    {"npu_id": 0, "device_ip": "192.168.1.10", "rank_id": "0", "npu_healthy": true},
                    {"npu_id": 1, "device_ip": "192.168.1.10", "rank_id": "1", "npu_healthy": true}
                ]
            }
        ]
    }]
}
```

## Supported Scenarios

| Scenario | Model Example | Support |
|----------|---------------|---------|
| MoE + PD分离 + DP/EP | Qwen3-Moe (235B-A22B) | ✓ |
| MoE + PD不分离 + DP/EP | Qwen3-30B-A3B | ✓ |
| 稠密模型 + PD不分离 + TP | Qwen3-Dense | ✓ |
| 异构TP重启 + MoE + PD分离 | DeepSeek-V4-Flash-w8a8-mtp: prefill DP4TP4 -> DP4TP(3,4,4,4) | ✓ (`VLLM_ITS_DEEPSEEK_V4=1`) |

DeepSeek-V4 异构重启的详细设计见
[`deepseekv4/DeepSeekV4_HETERO.md`](deepseekv4/DeepSeekV4_HETERO.md)。

## Development

### Directory Structure

```
plugins/zero_interrup/
├── __init__.py                         # 插件入口
├── patch.py                            # 插件 patch 类
├── engine_core_patch.py                # EngineCore monkey-patch (空转过滤第一层)
├── core_client_patch.py                # DPLBAsyncMPClient patch (空转过滤第二层)
├── common/
│   ├── __init__.py
│   ├── types.py                       # 类型定义
│   └── constants.py                   # 常量配置
├── executor/
│   ├── __init__.py
│   ├── its_multiproc_executor.py      # 主执行器
│   ├── health_monitor.py              # 健康监控
│   ├── strategy_sync.py               # 策略同步线程
│   └── http_server.py                 # HTTP 服务器（FastAPI）
├── worker/
│   ├── __init__.py
│   └── strategy_handler.py            # 策略处理器（仅 PD_REBUILD）
├── communication/
│   ├── __init__.py
│   └── decision_center_client.py      # 决策中心客户端
└── tests/
    ├── __init__.py
    ├── test_decision_center_client.py
    ├── test_health_monitor.py
    ├── test_http_server.py
    ├── test_its_multiproc_executor.py
    └── test_types.py
```

### Running Tests

```bash
pytest tests/ -v
```

## License

Apache License 2.0