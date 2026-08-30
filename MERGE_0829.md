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
| `VLLM_ITS_DEEPSEEK_V4` | `0` | `1/true/yes/on` 时选择 DeepSeek-V4 patch 族；否则使用 0829 patch 族。**安装阶段不再读取该变量**，只影响运行期 patch 分发与统一替换文件内部的分支选择。 |

`zero_interrupt/patch.py` 在运行时分发 A/B 的 runtime monkey-patch；
已安装到 vllm/vllm_ascend 的统一替换文件在调用时自行判断配置形态或
运行期环境变量，因此 `setup.py` 无需关心该变量。

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

### 运行期双版本（按 `VLLM_ITS_DEEPSEEK_V4` 选择）

| 模块 | 默认（B） | `VLLM_ITS_DEEPSEEK_V4=1`（A） |
|------|-----------|-------------------------------|
| `zero_interrupt/patch.py` 运行时分发 | B 主逻辑 | A `deepseekv4/patch.py` |
| Qwen 系列模型 patch（qwen2/3_5/3_moe/3_next/3_vl） | B | A |
| `vllm/model_executor/layers/patch_linear.py` | B | A |
| executor / engine-core / utils 等共同文件 | B | A |
| `patch_fused_moe.py` / `patch_eplb_utils.py` | B | A |

A 独有的 DeepSeek-V4 hetero 模块（`patch_hetero_*`、`patch_deepseek_v4*`
等）只存在于 `deepseekv4/` 中，不会在默认路径加载。

### 安装期统一文件（setup.py 不再按环境变量选目录）

以下整文件替换源已经合并成单文件，安装时固定从主目录拷贝；运行期由
文件内部逻辑选择分支：

| 安装目标 | 统一方式 |
|---|---|
| `vllm/config/parallel.py` | A 的 `HeterogeneousDPConfig` + B 的 `world_size_across_dp` override |
| `vllm/distributed/parallel_state.py` | A v0.23 原实现 + B 的 `asym_world_size/get_global_rank_asym/init_distributed_environment_asym/asym=True 组网/patch_tensor_parallel_group` |
| `vllm/model_executor/layers/fused_moe/config.py` | hetero TP / 0829 `zero_interrupt_config` / 对称公式三分支 |
| `vllm/v1/core/kv_cache_utils.py` | A v0.23 原实现 + B 的 mixed-page_size / 非对称投影与归一化，运行期按 `VLLM_ITS_DEEPSEEK_V4` 分流 |
| `vllm_ascend/distributed/parallel_state.py` | 直接采用 A（严格超集，含 `init_ascend_model_parallel_asym`） |
| `vllm_ascend/worker/worker.py` | A hetero 路径 + B 旧 asym 路径 + mamba KV 重算修复 |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 安装运行时分发器 + `*_deepseek_v4.py` / `*_0829.py` 两份实现 |
| `vllm_ascend/ops/rotary_embedding.py` | A 为底 + B `its_rotary` 开关 |

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
# 安装：不需要设置 VLLM_ITS_DEEPSEEK_V4
python3 setup.py          # 或构建安装
export VLLM_CUSTOM_PATCHES=zero_interrupt

# 默认 0829 场景
python3 -m vllm.entrypoints.openai.api_server ...

# DeepSeek-V4 场景（只需运行期设置）
export VLLM_ITS_DEEPSEEK_V4=1
python3 -m vllm.entrypoints.openai.api_server ...
```

## 验证记录

- `python -m compileall` 通过（主目录与 `deepseekv4/` 全量语法编译）。
- 用假 `vllm` / `vllm_ascend` 包验证 `setup.py` 在
  `VLLM_ITS_DEEPSEEK_V4=0` 与 `=1` 两种环境下安装的文件字节级一致
  （安装结果不再依赖该变量）。
- 用 stub 验证 `zero_interrupt/patch.py` 与 `patch_qwen3_5.py` 分发器在
  `VLLM_ITS_DEEPSEEK_V4=1/0` 下分别路由到 A/B 实现。
- 校验主目录与 B 仓文件一致性（仅预期的共享合并文件有差异），
  `deepseekv4/` 与 A 仓文件一致性（仅绝对导入前缀有预期改写）。

## 注意

- B 主目录按 0829 仓原样保留，其中少数文件基于 v0.18.0 派生；在
  origin_0.23.0 上的可用性以 0829 仓原有验证结论为准。
- 安装阶段不再依赖 `VLLM_ITS_DEEPSEEK_V4`；该变量只需在 vLLM 运行期
  设置，用于选择 runtime monkey-patch 族与统一替换文件的内部分支。
- `patch_qwen3_5.py` 在安装时会同时安装两份实现；切换场景只需修改
  运行期环境变量并重启服务。
