# vllm_plugins 与 vllm_plugins_0829 合并说明

## 合并原则

- `vllm_plugins`（A）的目标场景是 DeepSeek-V4 异构 TP（vllm/vllm-ascend
  v0.23.0）。
- `vllm_plugins_0829`（B）是并行开发的非对称 TP / zero-interrupt 实现。
- 实际安装目标仓库为 `origin_0.23.0`。
- 冲突实现保留 A/B 双份：**主目录为 B 实现（默认）**，A 实现整体放入
  `vllm_custom_plugins/plugins/zero_interrupt/deepseekv4/` 子目录。
- 无运行时冲突的公共实现或模块替换采用共享版本，不重复维护两份。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `VLLM_ITS_DEEPSEEK_V4` | `0` | `1/true/yes/on` 时，运行时 patch 与 `setup.py` 整文件替换都切换到 `deepseekv4/` 目录下的 DeepSeek-V4 实现；否则使用主目录的 0829 实现 |

`setup.py` 在安装/替换阶段读取该变量，`zero_interrupt/patch.py` 在运行时
读取同一变量，因此安装与运行需保持一致的值。

## 目录结构

```text
plugins/zero_interrupt/
├── ...                       # B（vllm_plugins_0829）实现，默认路径
└── deepseekv4/               # A（vllm_plugins）DeepSeek-V4 实现，完整镜像
    ├── patch.py
    ├── vllm/
    ├── vllm_ascend/
    └── DeepSeekV4_HETERO.md
```

`deepseekv4/` 内的插件绝对导入已统一改写为
`vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.*`，因此两个族各自
自包含，不会互相串用冲突文件。

## 模块合并结论

### 双版本（按 `VLLM_ITS_DEEPSEEK_V4` 选择）

| 模块 | 默认（B） | `VLLM_ITS_DEEPSEEK_V4=1`（A） |
|------|-----------|-------------------------------|
| `zero_interrupt/patch.py` 运行时分发 | B 主逻辑 | A `deepseekv4/patch.py` |
| `setup.py` 整文件替换源目录 | 主目录 | `deepseekv4/` |
| `vllm/config/parallel.py` | B | A |
| `vllm/distributed/parallel_state.py` | B | A |
| `vllm/model_executor/layers/fused_moe/config.py` | B | A |
| `vllm/v1/core/patch_kv_cache_utils.py` | B | A |
| `vllm_ascend/distributed/parallel_state.py` | B | A |
| `vllm_ascend/worker/worker.py` | B | A |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | B | A |
| Qwen 系列模型 patch（qwen2/3_5/3_moe/3_next/3_vl） | B | A |
| `vllm/model_executor/layers/patch_linear.py` | B | A |
| executor / engine-core / utils 等共同文件 | B | A |
| `patch_fused_moe.py` / `patch_eplb_utils.py` | B | A |

A 独有的 DeepSeek-V4 hetero 模块（`patch_hetero_*`、`patch_deepseek_v4*`
等）只存在于 `deepseekv4/` 中，不会在默认路径加载。

### 共享（不分版本）

- `license_verify/__init__.py`、`license_verify/license_reader.py`：保留 A
  （含 `VLLM_CUSTOM_PLUGINS_SKIP_LICENSE` 跳过开关与空路径保护）。
- `security_patch/` 下的 run_batch.py / envs.py / video.py /
  sampling_params.py / extract_hidden_states.py：保留 A（与 origin_0.23.0
  同源，目标仓库上为等价替换）。
- `zero_interrupt/common/types.py`、`decision_center_client.py`：保留 A
  （严格超集）。
- `v1/executor/http_server.py`、`strategy_sync.py`：保留 A（加固校验与
  重复策略转发修复，兼容 B 调用方式）。
- `vllm_ascend/ops/patch_ascend_linear.py`：保留 A（增加 DeepSeek-V4 显式
  shardings 读取，未配置时与 B 行为一致）。
- `vllm_ascend/ops/triton/rotary_embedding.py`：以 A（v0.23 同源）为底，
  合入 B 的 `its_rotary` 开关；未开启时保持 v0.23 原生行为。
- `setup.py` 的路径查找：合入 B 的 `pip show` 优先查找，并保留 A 的
  `import` 回退。

## 安装与运行

```bash
# 默认 0829 实现
VLLM_ITS_DEEPSEEK_V4=0 python3 setup.py          # 或构建安装
export VLLM_CUSTOM_PATCHES=zero_interrupt

# DeepSeek-V4 实现
VLLM_ITS_DEEPSEEK_V4=1 python3 setup.py
export VLLM_ITS_DEEPSEEK_V4=1
export VLLM_CUSTOM_PATCHES=zero_interrupt
```

## 验证记录

- `python -m compileall` 通过（主目录与 `deepseekv4/` 全量语法编译）。
- 用假 `vllm` / `vllm_ascend` 包验证 `setup.py` 在两种环境变量取值下，
  替换目标文件字节级等于对应 patch 源文件。
- 用 stub 验证 `zero_interrupt/patch.py` 在
  `VLLM_ITS_DEEPSEEK_V4=1/0` 下分别路由到 A/B 实现。
- 校验主目录与 B 仓文件一致性（仅预期的共享合并文件有差异），
  `deepseekv4/` 与 A 仓文件一致性（仅绝对导入前缀有预期改写）。

## 注意

- B 主目录按 0829 仓原样保留，其中少数文件基于 v0.18.0 派生；在
  origin_0.23.0 上的可用性以 0829 仓原有验证结论为准。
- `VLLM_ITS_DEEPSEEK_V4` 必须在 `setup.py` 执行期和 vLLM 运行期保持同一
  取值，否则整文件替换与运行时 patch 可能来自不同实现。
