# vLLM Custom Plugins Benchmark

本目录包含各插件的性能测试脚本，按插件名称分子目录存放。

## 目录结构

```
benchmark/
├── turboquant/           # TurboQuant KV缓存量化插件
│   ├── tests/            # 单元测试
│   └── README.md
├── ascend_ops/           # Ascend NPU算子插件
│   └── README.md
└── priority_scheduler/   # 优先级调度器插件
    └── README.md
```

## 快速开始

### TurboQuant 性能测试

```bash
cd benchmark/turboquant

# 运行单元测试
pytest tests/ -v

# 运行验证脚本
python verify_turboquant.py
python verify_simple.py

# 运行吞吐量测试
python throughput_test.py
```

### 运行API测试（需要先启动vLLM服务）

```bash
# 1. 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device npu

# 2. 另开终端运行API测试
python test_api.py
```

## 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `VLLM_CUSTOM_PATCHES` | 启用的插件列表 | `turboquant,ascend_ops` |
| `VLLM_TURBOQUANT_ENABLE` | 启用TurboQuant | `1` |
| `VLLM_TURBOQUANT_BITS` | 量化位数 | `3` |

## 依赖安装

```bash
pip install torch scipy numpy requests pytest
```