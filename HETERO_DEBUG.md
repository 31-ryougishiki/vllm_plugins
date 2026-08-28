# vllm_plugins 异构重启适配 v0.23.0 调试备忘录

> 目的：沉淀背景、关键结论和历次问题根因，便于后续在 A3 节点继续调试。
> 适用对象：vllm_plugins 仓 `hetero` 分支，vLLM / vllm-ascend v0.23.0。

---

## 0. 当前进度（更新于 2026-08-28）

### 已确认状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 对称 DP4TP4 正常拉起 + 普通请求推理 | ✅ 已验证 | 输出内容正常；§4.3/§4.4 两类错误均不再出现 |
| trigger 下发到 4 个 executor | ✅ 已验证 | 发送侧 4 个 HTTP 200；四个 DP 均进入策略执行 |
| 全量重启 barrier | ✅ 已通过 | dp0/dp2 日志均有 `Full-restart barrier passed` |
| DP2 重启 worker NPU 可见性 | ✅ 已修复 | `ASCEND_RT_VISIBLE_DEVICES` 改为数值排序（§4.5） |
| DP0 TP=3 worker 模型加载 | ✅ 已验证（10:21 复测） | `8192 is not divisible by 3` 不再出现（§4.7/§4.8） |
| 异构重启后 profile_run / KV cache 重建 | ✅ 已验证（10:21 复测） | `available_memory` 为真实数值，KV 重建成功（§4.9） |
| 异构重启后 MTP draft RoPE 崩溃 | 🔧 修复后不再崩溃 | 该问题只阻塞流程；按 speculative 原理它**不是乱码根因**（§4.10/§4.11） |
| 重复 trigger 静默丢弃 | ✅ 已修复 | `strategy_sync` 重复策略会转发而非 return（§4.6） |
| 异构重启后发请求 | ❌ 单请求可完成，但输出乱码 | 当前主战场；排查方向见 §4.11 |
| PD / Mooncake engine_id 轮换链路 | ✅ P 端已验证；⏳ D 端待验证 | dp0/dp1 均已 `KV connector metadata updated` + recovered 通知 |

### 本地提交（均已合入 `hetero` 分支）

```text
1314043 fix(attention): LCM-align DSA-CP metadata for MoE drafters
2fc1432 fix(hetero): refresh imported forward-context aliases after patch
98b679e fix(linear): replace pre-set output_sizes with divisible scaffold in stock init
89b9974 fix(linear): scaffold Ascend linears with divisible sizes under hetero TP
098dc63 fix(strategy_sync): forward duplicate deployment strategies instead of dropping them
6312296 fix(executor): sort healthy NPU ids numerically for ASCEND_RT_VISIBLE_DEVICES
39484bb fix(deepseek_v4): restore Ascend pluggable linear classes for symmetric inference
```

### 下一步顺序（不要跳步）

1. 当前唯一目标：**先让异构重启后的单请求输出与对称推理逐 token 一致**。
   - 先确认乱码发生在哪些 DP（dp0 请求 / dp1~3 请求）、prefill 阶段还是 decode 阶段。
   - 用相同 seed/prompt 对比 trigger 前后输出，记录第一个分叉 token 和 logits。
2. 按 §4.11 的优先级排查正确性：DSA-CP restore/o_proj → MoE gather/unpad → spec 模式主模型输入映射 → 权重加载。
3. 输出一致后再验证多请求、续推内容和 PD/Mooncake D 端：
   `heterogeneous producer restart rotates engine_id`。

---

## 1. 背景

- 目标功能：当 vLLM 推理服务所在 A3 节点出现 NPU 卡故障时，以**异构方式**重启推理 worker，规避单卡故障。
- 目标场景：单节点 prefill `DP4TP4` 故障一张卡后，以 `DP4TP(3,4,4,4)` 重启：
  - `world_size = 15`
  - 全局 rank 布局：DP0=`[0,1,2]`，DP1=`[3..6]`，DP2=`[7..10]`，DP3=`[11..14]`
  - DP0 使用 `tp_asymmetric_shardings=[2,1,1]`（64 heads → 32/16/16）
- PD 分离：P 端按上述异构拓扑重启，D 端保持 `dp16/tp1`，由 MooncakeHybridConnector 恢复 KV 传输链。
- 模型：`DeepSeek-V4-Flash-w8a8-mtp`，ModelSlim W8A8 量化，MTP + DSA-CP +
  `enable_shared_expert_dp`。
  - 关键结构：`num_attention_heads=64`、`n_routed_experts=256`、
    `o_groups=8`、`o_lora_rank=1024`、`num_key_value_heads=1`。

## 2. 代码仓与分工

| 路径 | 作用 |
|---|---|
| `vllm_plugins/` | 当前适配仓（`hetero` 分支），运行时插件 + setup.py 源码替换 |
| `vllm_plugins_origin/` | 老代码基线（0.18 时代实现） |
| `hetero_cp/` | 参考 demo：直接改 vllm/vllm-ascend v0.23.0 源码拉起异构服务 |
| `origin_0.23.0/` | vllm/vllm-ascend v0.23.0 官方基线 |
| `origin_0.18.0/` | v0.18.0 基线，仅用于对比旧 API |
| `vllm_plugins_hetero_test/` | 远程 A3 安装 / 拉起 / 触发异构重启脚本 |
| `DeepSeek-V4-Flash-w8a8-mtp/` | 模型配置样例 |
| `error.log` | 最近一次报错节选 |

关键对比命令：

```bash
# hetero_cp 相对 v0.23.0 基线的改动清单
git -C hetero_cp/vllm diff 0fc695fc6d..HEAD
git -C hetero_cp/vllm-ascend diff 5cb98caaa..HEAD

# vllm_plugins 改动清单
git -C vllm_plugins log --oneline
git -C vllm_plugins diff HEAD
```

## 3. 插件架构与关键文件

- 入口：`vllm_plugins/vllm_custom_plugins/__init__.py`
  - 通过 `vllm.general_plugins` entry point 加载。
  - `VLLM_CUSTOM_PATCHES=zero_interrupt` 控制是否应用 zero_interrupt patch。
- 插件入口：`vllm_plugins/vllm_custom_plugins/plugins/zero_interrupt/patch.py`
  - 替换 `MultiprocExecutor -> ITSMultiprocExecutor`，
    `WorkerProc -> ITSNPUWorker`。
  - patch `EngineCoreProc/DPEngineCoreProc`。
  - 依次应用 DeepSeek-V4 异构 TP 的运行时 patch。
  - 关键 patch 失败现在会 **fail-fast**（抛 `RuntimeError`）。
- 安装脚本：`vllm_plugins/setup.py` + `build.sh`
  - `build.sh install` 构建 wheel 时，setup.py 会替换以下已安装源码文件（保留 `.bak`）：
    - `vllm/config/parallel.py`
    - `vllm/distributed/parallel_state.py`
    - `vllm/model_executor/layers/fused_moe/config.py`
    - `vllm/v1/core/kv_cache_utils.py`
    - `vllm_ascend/distributed/parallel_state.py`
    - `vllm_ascend/worker/worker.py`
    - `vllm_ascend/ops/rotary_embedding.py`
    - `vllm_ascend/patch/worker/patch_qwen3_5.py`
    - 若干 security_patch 文件
  - 其中 `vllm/config/parallel.py`、`vllm/distributed/parallel_state.py`、
    `fused_moe/config.py` 与 hetero_cp 对应版本完全一致。
- 核心控制面：
  - `vllm/v1/executor/its_multiproc_executor.py`
    - worker 重启、策略执行、全 DP barrier、设备可见性设置。
  - `vllm/v1/engine/engine_core_patch.py`
    - EngineCore busy loop 中消费 `wait_new_deployment/recv_new_deployment`。
    - 容错 `step` / `step_with_batch_queue`。
    - DP 状态同步与全量重启 barrier 使用**同形 collective**（见 §5）。
  - `vllm/v1/executor/http_server.py`
    - 决策中心下发策略的 HTTP 服务，校验 `executor_id` 和 `deploy_type`。
  - `vllm_ascend/ops/patch_hetero_ascend_linear.py`
    - Ascend linear 类的非对称切分 patch。
    - W8A8 模型 `wo_a` 未量化时的 3D reshape（见 §4.3）。
  - `vllm_ascend/attention/patch_deepseek_v4_attention_hetero.py`
    - DSA v1 / DSA-CP 非对称 head、o_proj 恢复、LCM padding。
  - `vllm_ascend/ops/fused_moe/patch_hetero_moe.py`
    - MoE Prepare/Finalize、token dispatcher、256 experts/15 EP rank 余数分布。
  - `vllm_ascend/ops/patch_hetero_custom_ops.py`
    - 已注册自定义算子的异构 gather/reduce/unpad 实现。
  - `vllm_ascend/distributed/kv_transfer/patch_hetero_mooncake.py`
    - MooncakeHybridConnector 异构端口映射、engine_id 轮换、超时 setdefault。

## 4. 历次问题根因与修复

### 4.1 `FutureWrapper.__init__() missing 'get_response'`（已修复）

- 现象：服务能拉起，发请求后 `EngineCore_DP0` 报
  `TypeError: FutureWrapper.__init__() missing 1 required positional argument: 'get_response'`。
- 根因：`its_multiproc_executor.py:collective_rpc` 仍使用 vLLM 0.18 的
  `FutureWrapper` 语义；0.23 的签名是
  `FutureWrapper(futures_queue, get_response, aggregate=...)`，
  且 `__init__` 自己 `appendleft(self)`。
- 修复：
  - 改为 0.23 语义：`FutureWrapper(self.futures_queue, get_response=get_response, aggregate=aggregate)`。
  - 非阻塞返回 future，阻塞返回 `future.result()`。
  - `futures_queue` 类型改为 `deque[FutureWrapper]`。
- 注意：该错误掩盖了后续 worker 前向错误；修完必须继续发真实请求验证。

### 4.2 策略执行时 `batch_queue.clear()` 崩溃（已修复）

- 根因：默认 PP=1 且未开 async scheduling 时 `batch_queue is None`，
  但 `patched_handle_shutdown` 无条件调用 `self.batch_queue.clear()`。
- 修复：加 `if self.batch_queue is not None`。
- 同时修复：
  - abort 请求后调用 `_send_abort_outputs`。
  - 删除 `scheduler.finished_req_ids.clear()`。
  - 去掉“5 秒必须 abort 完”的 assert，改为告警继续。
  - 新增 `patched_step`：默认 `step()` 路径对 `model_output=None` 容错。

### 4.3 DSA-CP 前向 `IndexError: Dimension out of range (expected [-2, 1], got 2)`（已修复）

- 现象：worker `DP0_TP0_EP0` 在正常推理时，于
  `vllm_ascend/attention/context_parallel/dsa_cp.py:1262` 的
  `torch_npu.npu_transpose_batchmatmul(o_proj_input, self.wo_a.weight, ...)`
  报 `IndexError`。
- 根因：DeepSeek-V4-Flash-w8a8 是 ModelSlim W8A8，`wo_a` 被量化配置排除，
  使用 `AscendUnquantizedLinearMethod`，weight 保持 2D；
  而 `npu_transpose_batchmatmul(perm_x2=(0,1,2))` 要求 3D。
  只有 FP8 `ds_linear` 路径会把 `wo_a.weight` 重排成
  `[n_local_groups, input, o_lora_rank]`。
- 修复：
  - `patch_hetero_ascend_linear.py` 中新增 `_reshape_wo_a_for_dsa()`：
    在 `AscendUnquantizedLinearMethod.process_weights_after_loading`
    和 stock `UnquantizedLinearMethod.process_weights_after_loading`
    执行后，对 2D 的 DeepSeek-V4 `wo_a` 做：
    `weight.view(n_local_groups, o_lora_rank, -1).transpose(1,2).contiguous()`。
  - `patch_deepseek_v4_attention_hetero.py` 中给
    `AscendDSACPImpl.process_weights_after_loading` 增加防御性 3D reshape。
- 若后续出现同类报错，先检查：
  1. `wo_a.weight.dim()` 是否为 2；
  2. quant method 是否为 unquantized；
  3. 上述 patch 是否在模型加载前应用。

### 4.4 对称正常推理 `expanded size (1) must match (4)` / 输出混乱（已修复）

- 场景：**对称 DP4TP4 正常拉起服务、尚未触发异构重启**。报错发生在
  `AscendDeepseekV4ForCausalLM.forward`（主模型），不是 MTP draft。
- 现象：
  - 先报 `dsa_cp.py` 的 `output[...] = self._apply_wo_b(...)`：
    `RuntimeError: expanded size (1) must match existing size (4)`；
  - 若用“裁剪 output 行数”的方式绕过该报错，服务不再崩，但**输出内容混乱**。
- 根因（对称场景就成立）：
  - `patch_deepseek_v4.py` 之前**无条件**把 `deepseek_v4` 模块里的
    `ColumnParallelLinear / MergedColumnParallelLinear / RowParallelLinear`
    换成了插件里的 vLLM `*Asymmetric` 子类。
  - vLLM 的 pluggable-layer 注册表只认基类名 `RowParallelLinear`；
    `RowParallelLinearAsymmetric.__name__` 不在 OOT 注册表里，因此
    对称启动时 `wo_b/down_proj` 实例化的是**普通 vLLM RowParallel**，
    而不是 `AscendRowParallelLinear`。
  - 普通 RowParallel 的 forward 只做 TP all_reduce，**没有 FlashComm1 的
    token 维 reduce_scatter**。DSA-CP restore 返回全量 padded 流
    （TP4 + 1 token → 4 行），wo_b 不再把 4 行收敛成 output 的 1 行，
    于是形状冲突。
  - 上一版用 `_align_dsa_o_proj_output()` 裁剪前缀只是把错误“吞掉”：
    非 rank0 的 padding 行被当成有效输出写回，所以能跑但输出乱。
- 修复（fail-closed，而不是裁剪掩盖）：
  - **撤销 `AscendDSACPImpl.forward` 的整函数替换**，恢复 stock
    `output[...] = self._apply_wo_b(...)`，让形状错误继续以异常暴露。
  - **删除 `patch_deepseek_v4.py` 里对模块全局 linear 类的无条件替换**。
    对称启动保持 stock 类名，使 pluggable-layer 注册表正常分发到
    `AscendRowParallelLinear / SequenceRowParallelOp`。
  - 异构 ratio 分片不再依赖 vLLM `*Asymmetric` 子类，统一由
    `patch_hetero_ascend_linear.py` 直接 patch `Ascend*ParallelLinear`
    自身完成（该 patch 已按 §4.3/§9.6 支持 ratios）。
  - 保留 `_build_local_token_metadata` 的 draft-builder LCM 保护
    （`_is_dsa_cp_draft_builder`，见 §9.3/§9.4）：那是异构重启后 draft
    metadata 的独立隐患。
    **注意：该“draft 一律不 LCM”的保护后来被 §4.10 修正——MoE drafter
    仍要 LCM，只有稠密 drafter 才跳过。**
  - 保留 `patch_hetero_model_runner._patched_profile_run()` 的 PCP
    双重取整修复。
- 验证顺序：先对称普通请求（`grep "expanded size" logs/prefill/` 应为空，
  且输出语义正常），再触发异构重启。

### 4.5 异构重启 worker 启动 `aclInit error 107001 / Invalid device ID`（已修复）

- 现象：触发 `trigger_hetero_restart.sh` 后，DP0（TP=3）能拉起，但
  DP2 新 worker 进程报：
  `RuntimeError: aclInit, error code is 107001 ... Invalid device ID.
  Expected value: [0, 0).`
- 根因：`its_multiproc_executor._init_workers()` 过滤健康 NPU 后，
  用 `sorted(new_npu_id_list)`（**字符串排序**）生成
  `ASCEND_RT_VISIBLE_DEVICES`。DP2 的物理卡是 `8,9,10,11`，
  字符串排序得到 `10,11,8,9`。NPU runtime 不认可这种非升序
  列表，子进程 aclInit 时按 0 个可见设备初始化，报 Invalid device ID。
- 修复：所有生成 `ASCEND_RT_VISIBLE_DEVICES` 的地方统一改为
  `sorted(..., key=int)`，包括独立部署分支、mm_scene 分支和
  物理卡 ID 到逻辑 ID 的映射列表。
- 验证：重启后 grep `Setting ASCEND_RT_VISIBLE_DEVICES`，DP2 应为
  `8,9,10,11`，且无 `Invalid device ID`。

### 4.6 重复 trigger 时个别 DP 收到 200 但无反应（已修复）

- 现象：发送侧全部 HTTP 200，但某个 DP（尤其 dp0）日志没有
  `parsing deploy request / Strategy received via HTTP`，或只有
  HTTP 收到日志、没有后续执行日志。
- 根因候选：
  1. `StrategySyncThread.on_strategy_received()` 原来对
     `_current_strategy == strategy` 的重复请求**直接 return**，
     HTTP 仍返回 200。上一轮失败/部分投递后重放同一份 trigger 时，
     该 DP 会静默丢弃，而其它 DP 正常执行。
  2. HTTP 200 来自**旧的残留进程**：8001 端口上还挂着上一轮 dp0，
     新拉起 dp0 的 ITS server 没绑定成功；curl status 会显示旧进程信息。
  3. dp0 EngineCore 尚未注册 `_trigger_busy_loop_callback`（busy loop
     还没走到 `_handle_shutdown`），HTTP 只 set 了
     `recv_new_deployment`，没有 WAKEUP；通常下一次循环会补处理，
     但若已 `_paused_for_restart` 就卡住。
- 修复：
  - 重复策略不再静默 return：记录 warning 并**再次回调 executor**，
    由 executor 的执行路径保证幂等/重试。
  - 排障时按日志顺序定位：
    `parsing deploy request` → `Strategy received via HTTP` →
    `Strategy sync received` → `Received deployment strategy` →
    `Paused busy_loop` → `restarting workers of EVERY DP instance`。
    缺哪一段就查对应进程/端口/回调注册。
  - 端口自查：
    `curl http://127.0.0.1:8001/api/v1/executor/status`
    应返回 `port: 8001`，且与 dp0 日志的 `_http_port` 一致。

### 4.7 异构重启后 DP0 worker `8192 is not divisible by 3`（已修复）

- 现象：trigger 后 DP0（TP=3，ratios=[2,1,1]）worker 加载模型时，
  `wo_a = ColumnParallelLinear` 报：
  `AssertionError: 8192 is not divisible by 3`。
  位置：`patch_hetero_ascend_linear._patched_col_init` →
  stock `AscendColumnParallelLinear.__init__` 的 `divide(output_size, tp_size)`。
- 根因：§4.4 修复后不再把模块 linear 类换成 vLLM `*Asymmetric` 子类，
  异构模型改走被 patch 的 `AscendColumnParallelLinear` /
  `AscendRowParallelLinear`。但 stock Ascend init **先做整除校验**，
  而 `wo_a/wo_b` 的 8192 维度在 TP=3 + `[2,1,1]` 下本来就不整除。
- 修复：
  - `_patched_col_init` / `_patched_row_init` 在 ratios 生效时，
    先用 `_ceil_divisible(value, tp_size)` 构造一个可整除的临时维度
    调 stock init（只用于搭 scaffolding），随后恢复真实
    `input_size/output_size/output_sizes`，再调用 `_rebuild_*` 按
    `get_tp_partition_size(..., ratios)` 重建权重。
  - 对称路径仍然原样调用 stock init，行为不变。
- 注意：后续若出现其它 Ascend linear 子类（如 QKV）在异构 TP 下
  `divide` 报错，同样需要做“临时可整除维度 + 重建真实分区”的处理。

### 4.8 异构重启后 DP0 worker 仍报 `8192 is not divisible by 3`（已修复）

- 现象：本次 A3 复测触发 trigger 后，DP0（TP=3）三个新 worker 都在
  `wo_a` 的 `AscendColumnParallelLinear.__init__` 处失败：
  `AssertionError: 8192 is not divisible by 3`。traceback 位置仍是
  stock `linear.py:405` 的
  `[divide(output_size, self.tp_size) for output_size in self.output_sizes]`。
- 根因：§4.7 的“临时可整除尺寸”修复在调用 stock init **之前**就执行了
  `self.output_sizes = real_output_sizes or [output_size]`。而 stock
  `AscendColumnParallelLinear.__init__` 的可整除判断是
  `if hasattr(self, "output_sizes")`，读的是**实例属性**
  `self.output_sizes`，不是入参 `output_sizes`。所以即使传入
  `output_size=8193, output_sizes=None`，stock init 仍按预置的
  `[8192]` 做 `divide(8192, 3)`，修复被绕过。
- 修复：异构分支在调用 stock init 前，把 `self.output_sizes` 临时替换为
  `[uniform_output_size]`（可整除的 scaffolding 尺寸），stock init
  返回后再恢复真实尺寸并执行 ratio 重建。普通 Column 与 Merged 子类
  （预先设置 `output_sizes`）两条路径都已覆盖；Row 路径原实现已正确。
- 回归：新增 `vllm_ascend/ops/tests/test_patch_hetero_ascend_linear.py`，
  用最小 stock-init 替身精确复现 `hasattr(self, "output_sizes")` 行为，
  覆盖 Column / Merged / Row 三条 scaffolding 路径。

### 4.9 异构重启后 profile_run 报 `shape '[4, 8184, 4096]' is invalid`（已修复，A3 已验证）

- 现象：§4.8 修复后，DP0 三个 + DP1 四个新 worker 模型（含 MTP）加载成功，但
  `determine_available_memory -> profile_run -> _dummy_run -> MLP/MoE` 时报：
  ```
  patch_hetero_custom_ops.py:164
      x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
  RuntimeError: shape '[4, 8184, 4096]' is invalid for input of size 125706240
  ```
  DP0 同一位置的输入 size 为 167608320，同样无法 view。dp0/dp1 日志均已复现。
  随后 EngineCore `_reinitialize_kv_cache` 拿到 `[None, None, None, None]`，
  又报 `TypeError: '<=' not supported between 'NoneType' and 'int'`。
- 根因（patch 绑定时序，不是算子公式）：
  - `patch.py` 在应用 hetero 系列 patch **之前**就
    `from vllm_ascend.worker.worker import NPUWorker`；这会连带 import
    `vllm_ascend/worker/model_runner_v1.py`，而后者在模块顶部执行了
    `from vllm_ascend.ascend_forward_context import set_ascend_forward_context`。
  - 之后 `apply_hetero_forward_context_patch()` 只替换了
    `afc.set_ascend_forward_context` 这个模块属性，**没有刷新
    `model_runner_v1`（以及若干 spec_decode proposer）里已经绑定到旧函数
    对象的别名**。于是实际跑 profile_run 的仍是 stock forward context，
    `_EXTRA_CTX.per_dp_tp_sizes / per_dp_padded_lengths` 从未被写入。
  - custom op 检测不到 per-DP 布局，走同形 EP all_gather 分支，用
    `dp_size * padded_length` 去 view 异构 gather 结果，形状不匹配。
  - KV cache 报 `NoneType <= int` 是次生错误：worker profile_run 失败后
    `collective_rpc` 容错返回 `[None,...]`，被原样传进
    `get_kv_cache_configs`。
- 修复：
  - `patch_hetero_tp.apply_hetero_forward_context_patch()` 在替换
    `afc` 属性后，遍历 `sys.modules`，把所有捕获了原函数对象的
    `set_ascend_forward_context` / `set_mc2_tokens_capacity` 模块别名
    一并指向 patched 版本。
  - `_reinitialize_kv_cache` 对 `available_memory` 含 `None` 的情况
    fail-closed，直接给出“worker profile_run 失败”的明确错误，不再把
    None 传入 `kv_cache_utils`。
- 回归：新增
  `vllm_ascend/patch/tests/test_patch_hetero_tp.py`，模拟“先 import、
  后 patch”的模块别名场景，断言旧别名被刷新且无关属性不被覆盖。

### 4.10 异构重启后真实请求在 MTP draft 前向报 RoPE `dim0 must be equal`（修复后不再崩溃）

- 现象：profile_run / KV cache 重建 / Mooncake 元数据恢复全部通过，
  DP0 收到真实请求后，worker 在 `sample_tokens ->
  propose_draft_token_ids -> _run_merged_draft` 的 MTP draft 第一层报：
  ```
  dsa_cp.py:1369 inplace_partial_rotary_mul
  RuntimeError: EZ1008 ... dim0 must be equal
  CheckInputShapes ... inplace_partial_rotary_mul_a3_tiling.cpp
  ```
- 根因：§9.3 之前假设“MTP draft 的 `flash_comm_v1_enabled=False`，
  draft metadata 绝不能 LCM 对齐”。但该结论只对**稠密 drafter**成立。
  DeepSeek-V4 MTP 是 **MoE drafter**，`set_ascend_forward_context` 会走
  `is_drafter_moe_model()` 分支并把 FlashComm1 保持为 True；此时
  `maybe_pad_and_reduce` 已把 draft 隐状态按 `lcm(tp_sizes)` padding 并
  reduce_scatter 成 `padded_length / tp` 行，而 `_patched_build_local_token_metadata`
  因 `_is_dsa_cp_draft_builder` 标记跳过了 LCM，只做本地 `tp_size` 对齐。
  两者行数不同，RoPE kernel 报 `dim0 must be equal`。
- 修复：
  - `_patched_build_local_token_metadata` 对 draft builder 不再一律跳过 LCM；
    只有稠密 drafter（`is_drafter_moe_model() == False`）跳过，MoE drafter
    与主模型一样用 `lcm(tp_sizes)` 对齐。
  - 判断逻辑统一为：`use_lcm = (非 draft builder) or is_drafter_moe_model`。
- 回归：新增
  `vllm_ascend/attention/tests/test_patch_deepseek_v4_attention_hetero.py`，
  覆盖“主模型 LCM / MoE drafter LCM / 稠密 drafter 不 LCM”三种情况。
- 现状：该修复后单请求不再崩溃；但当前乱码是**独立的主模型正确性问题**
  （draft 不参与最终采样），排查见 §4.11。

### 4.11 当前问题：单请求能跑完但输出乱码（进行中）

- 现象：trigger 异构重启、profile_run、KV 重建、Mooncake 元数据恢复全部通过；
  发送单个请求能正常返回，但文本是乱码。
  10:21 复测日志佐证：dp0 `available_memory=[25.4GB,28.0GB,28.0GB]`、
  `KV cache re-initialized successfully`、`KV connector metadata updated
  successfully`，dp0/dp1 均发送 recovered 通知；随后请求路径不再有
  crash，只有乱码。
- 性质：**已经越过了所有“形状/集合通信错误”，进入数值正确性问题**。
  此时张量形状、collective 大小都合法，但 token 顺序、head 顺序、专家
  结果或权重分片被错误排列。

- **总原则：draft 模型不影响最终输出。**
  speculative decoding 的最终 token 永远由 target 模型自己的 logits 验证/
  采样得到；draft 错误只会改变接受率、耗时，或造成 draft 路径 crash，
  **不可能成为“输出乱码”的根因**。因此乱码只可能来自主模型的
  attention/MoE/权重加载，或 target logits→sampler 的输入映射。
  “关闭 MTP 后输出变了/没变”不能用来判定 draft 是否有 bug；它只能说明
  spec 模式下主模型的 batch 布局（verify batch、draft token 位置、
  attention metadata、logits indices）是否被正确喂给主模型。

- 排查优先级（从最可能到最不可能）：

  1. **DSA-CP attention 的 head 恢复 / o_proj**
     - `_restore_tp_head_layout` 的异构分支：`head_sizes/head_offsets`、
       `input_split_sizes/output_split_sizes` 是否按 `[2,1,1]` 正确；
     - `n_local_heads / n_local_groups` 是否在 Attention 外层与
       `AscendDSACPImpl` 内部两处都被修正；
     - `wo_a` 3D reshape 的 `n_local_groups`（DP0 rank0 应为 4 组，不能是
       `o_groups // tp_size = 2`），以及 `wo_b` 是否保持
       `AscendRowParallelLinear` 的 SP reduce_scatter。

  2. **MoE gather/unpad 与 shared/routed 输出拼接**
     - `_maybe_all_gather_and_maybe_unpad_impl` 的 rank-by-rank unpad walk：
       每个 rank 的 slot 都必须按 `uniform_rank` 前进，即使该 rank 没有
       实际 token；
     - `_maybe_pad_and_reduce_impl` 的 `padded_x` 偏移与尾部清零；
     - `patch_hetero_moe_runner` 中 shared/routed 两条分支谁长补谁，补的
       必须是 padding 行、且最后不能把 padding 当有效行输出。

  3. **spec 模式下的主模型输入/输出映射**
     - 开启 MTP 时，主模型一次 verify 的是 `[已接受 token + 1 个 draft
       token]` 的 batch；若 `query_start_loc`、positions、slot_mapping 或
       `token_indices_to_sample` 在异构 TP 下错位，主模型拿到错误输入，
       最终输出就会乱码。排查对象仍是**主模型 forward 的输入装配**，不是
       draft 模型本身。

  4. **异构权重加载**
     - `_patched_col_init/_patched_row_init` 的双重 create + 重建后，
       `weight_loader` 的 `get_tp_partition_offset` 必须与
       `get_tp_partition_size` 用同一组 ratios；
     - W8A8 的 `weight_scale/weight_offset` 等 per-channel 参数也要走同
       一套 offset，不能沿用 `tp_rank * shard_size`。

- 最小隔离实验（按顺序做）：
  1. 相同 seed/prompt，分别记录**对称启动**和**异构重启后**的返回文本，
     找出第一个分叉 token；
  2. 看乱码请求是否只发生在 DP0（TP=3 + ratios）还是所有 DP；若只有
     DP0，直接锁定 `[2,1,1]` 非对称路径；
  3. 关闭 MTP 重测可以**简化主模型 batch**（去掉 verify 里的 draft 位置），
     便于锁定“普通 batch 主模型错”还是“spec 模式下主模型输入装配错”；
     但无论结果如何，根因仍归主模型；
  4. 在 `_restore_tp_head_layout` 和 MoE prepare/finalize 出口打印：
     `hidden_states.shape`、`local_attn_output.shape`、
     `n_local_heads/n_local_groups`、`wo_a.weight.dim()`（§9.2 的现场清单）；
  5. 对比 DP0 rank0 的每层 hidden_states 与对称运行的对应值（同一输入）。

- 铁律：**不要用“裁剪 / 丢弃多出来的行 / 只取 rank0”来压掉形状差异**。
  乱码说明某处已经悄悄错位，裁剪只会把错误固化（§4.4 教训）。

## 5. 异构重启流程要点（0.23 适配结论）

1. **所有 DP 都必须重启**。
   `_strategy_requires_full_restart()` 对 DEGRADE / PD_REBUILD / RECOVER
   的异构拓扑变化返回 True，决策中心必须给每个 executor 下发完整
   `engine_parallel_config`。
2. **全量重启 barrier 与 DP 业务同步使用同形 collective**：
   - `ParallelConfig.sync_dp_state` 是 2 个 int32 的 SUM all_reduce。
   - `_has_global_unfinished_reqs` 也使用该形状（带 10s 超时，超时 abort）。
   - `_barrier_for_full_restart` 也使用 2 个 int32 SUM all_reduce。
   - 这样即使某个 DP 还卡在业务同步中，也不会与 barrier 形状不匹配而交叉死锁。
3. **worker 全局 rank 重算**：
   `WorkerProc.rank/local_rank` 保持 DP 内本地 rank；
   全局 torch rank 由替换后的 `init_distributed_environment` 按
   `get_rank_offset_for_dp(dp_rank)` 累加得到（0/3/7/11）。
4. **MoE 256 experts / EP=15**：
   - 专家分布采用 `base + (ep_rank < remainder)`，不能用均匀除法。
   - 256 % 15 != 0 时 `_select_a3_moe_comm_method` 强制回退 `ALLGATHER`。
5. **DSA-CP attention 非对称**：
   - `n_local_heads/n_local_groups` 按 `tp_sharding_ratios` 切分。
   - `_restore_tp_head_layout` 用带 `input_split_sizes/output_split_sizes`
     的非均匀 `all_to_all_single`。
6. **MooncakeHybridConnector**：
   - 异构 DP 端口偏移 = `get_rank_offset_for_dp(dp_rank)`（0/3/7/11），
     不是 `dp_rank * tp_size`。
   - 元数据按**绝对 handshake_port** 匹配。
   - P 端全量重启轮换 `engine_id`，D 端按
     `(engine_id, handshake_port)` 重新拉取远端元数据。

## 6. 已确认未覆盖 / 后续需继续处理的项

1. **不支持启动期异构拓扑**：
   - `--heterogeneous-dp-config` CLI 未接入（`vllm/engine/patch_arg_utils.py`
     目前是死代码）。
   - `vllm/v1/engine/utils.py` 的设备隔离 offset patch、
     `vllm/v1/engine/core.py` 的启动期 world_size 重算、Ray actor 路径
     均未 patch。
   - 当前只能“对称启动 → 策略触发异构重启”。
2. **`_reinitialize_kv_cache` 未重建 scheduler 的 `KVCacheManager`**：
   - worker 端 KV cache 已重建，scheduler 端 block pool 仍是启动时配置。
   - 若新拓扑 `num_gpu_blocks` 变化，可能出现越界或内存利用不足。
3. **setup.py 仍直接替换源码文件**：
   - 当前依赖替换 `parallel.py/parallel_state.py/fused_moe config` 等，
     未完全符合“DeepSeek-V4 相关修改用 patch”的要求。
4. **非异构 PD_REBUILD 的健康实例 RPC 更新路径**：
   - 已补 `NPUWorker.update_kv_connector_for_pd`，但底层
     `StrategyHandler._execute_pd_rebuild` 仍是 0.18 风格，未在
     A3 上验证非异构 PD 恢复。

## 7. 远程 A3 验证流程

```bash
# 假设工作路径 /opt/its/z30055003
cd /opt/its/z30055003/vllm_plugins_hetero_test

# 1. 安装（会自动执行 setup.py 源码替换）
bash install_vllm_plugins.sh

# 2. 拉起 4 个 prefill engine（初始对称 DP4TP4）
nohup bash launch_prefill_hetero_test.sh \
  > /opt/its/z30055003/logs/launch.log 2>&1 &

# 3. 先发一个普通请求，确认对称推理正常（第一回归站）
#    重点确认两类错误都不再出现：
#    a) npu_transpose_batchmatmul 的 IndexError（wo_a 2D，§4.3）
#    b) output[...] 的 expanded size (1) must match (4)
#       （主模型 wo_b 失去 SP reduce_scatter，§4.4）
grep -R "expanded size of the tensor" logs/prefill/     # 应为空
grep -R "Dimension out of range" logs/prefill/          # 应为空

# 4. 触发异构重启：DP4TP4 -> DP4TP(3,4,4,4)
bash trigger_hetero_restart.sh

# 5. 观察日志关键字
grep -R "restarting workers of EVERY DP instance" logs/prefill/
grep -R "Full-restart barrier passed" logs/prefill/
grep -R "heterogeneous producer restart rotates engine_id" logs/prefill/
# 本轮新增必查：
grep -R "Invalid device ID" logs/prefill/               # 应为空（§4.5）
grep -R "8192 is not divisible by 3" logs/prefill/      # 应为空（§4.7）
grep -R "Duplicate deployment strategy" logs/prefill/   # 若重复 trigger 应出现 warning 而非静默（§4.6）

# 6. 重启后再发请求，确认续推/PD 链路正常
```

常用排障日志位置：

- prefill：`logs/prefill/dp{0..3}.log`
- 策略 HTTP：`curl http://127.0.0.1:8001/api/v1/executor/status`
- 停服：`pkill -f "vllm serve /opt/its/model/DeepSeek-V4-Flash-w8a8-mtp-self"`

## 8. 当前已知错误速查表

| 错误 | 根因 | 状态 |
|---|---|---|
| `FutureWrapper.__init__() missing 'get_response'` | collective_rpc 0.18 语义 | 已修复 |
| `AttributeError: 'NoneType' object has no attribute 'clear'`（策略触发时） | `batch_queue=None` | 已修复 |
| `IndexError: Dimension out of range (expected [-2, 1], got 2)`，位于 `dsa_cp.py` 的 `npu_transpose_batchmatmul` | W8A8 未量化 `wo_a` 2D | 已修复 |
| `RuntimeError: expanded size (1) must match existing size (4)`，位于 `dsa_cp.py` 的 `output[...] = _apply_wo_b(...)`；裁剪绕过会导致输出混乱 | `patch_deepseek_v4.py` 无条件把模块 linear 类换成 vLLM `*Asymmetric`，绕过 Ascend pluggable-layer 注册，`wo_b` 失去 SP reduce_scatter | 已修复（不再替换模块类；异构 ratio 由 `patch_hetero_ascend_linear` 处理） |
| 异构重启后挂死 | DP 同步 / barrier collective 形状不匹配 | 已修复（同形 2×int32 SUM） |
| 只给故障 DP 下发策略 | barrier 120s 超时 fail-closed，EngineCore 退出 | 设计如此；决策中心必须广播完整策略 |
| `inplace_partial_rotary_mul` / `EZ9999: Inner Error` | `local_cos` 行数（tokens_per_rank）与 q 行数不一致；draft 分支的最终规则见 §4.10：MoE drafter 要 LCM，稠密 drafter 不 LCM | 已修复（§4.10） |
| 异构重启后 worker `aclInit 107001 / Invalid device ID, Expected [0,0)` | `ASCEND_RT_VISIBLE_DEVICES` 被 `sorted()` 按字符串排序，`8,9,10,11` 变成 `10,11,8,9`，NPU runtime 解析失败 | 已修复（`sorted(..., key=int)`） |
| 异构重启后 DP0 worker `8192 is not divisible by 3`（`wo_a/wo_b` 初始化） | stock `AscendColumnParallelLinear`/`AscendRowParallelLinear` 先做整除校验；TP=3 时 8192 不整除 | 已修复（临时可整除维度搭 scaffolding，再按 ratios 重建权重） |
| 同上错误在 §4.7 修复后复现 | stock col init 读预置的 `self.output_sizes` 而非入参，scaffolding 尺寸被绕过 | 已修复（§4.8） |
| 异构重启后 profile_run `shape '[4, 8184, 4096]' is invalid`，KV 重建 `NoneType <= int` | `model_runner_v1` 等模块在 patch 前已 import，`set_ascend_forward_context` 别名仍指向 stock 函数，`_EXTRA_CTX` 无 per-DP 布局 | 已修复（§4.9，刷新已导入模块别名） |
| 真实请求 MTP draft 前向 `inplace_partial_rotary_mul` `dim0 must be equal` | MoE drafter 实际 FlashComm1=True、隐状态按 LCM 分片；draft builder 被错误地跳过 LCM，只用本地 TP 对齐 | 已修复（§4.10） |

---

## 9. DSA-CP 关键形状与 patch 易错点（速查）

> 本节是 §4.3/§4.4 反复踩坑后的结论，后续改 attention / MoE / spec_decode
> patch 前先读这一节，能少走弯路。

### 9.1 形状不变量

记：

```text
N_real  = common_attn_metadata.num_actual_tokens
N_in    = common_attn_metadata.num_input_tokens   # 已按本地 TP 或 DP 对齐后的值
N_pad   = _build_local_token_metadata 算出的 num_tokens_pad
L       = N_pad / tp_size                         # tokens_per_rank，本地 chunk 行数
```

DSA-CP 前向必须满足：

1. `q`、`local_cos`、attention 输出都是 **L 行**；三者行数不一致会先死在
   `inplace_partial_rotary_mul`（NPU 侧通常报 `EZ9999`）。
2. `_restore_tp_head_layout()` 返回的是**全量 padded 流**：
   `N_pad = L * tp_size` 行，不是本地行数。有效 token 是它的前缀
   `[0, N_real)`，padding 在尾部。
3. `wo_b` 是唯一会把行数收敛回来的环节，但**前提是它确实是
   `AscendRowParallelLinear`**：
   - SP 时 `SequenceRowParallelOp.matmul_and_reduce` 做 token 维
     reduce_scatter，`N_pad -> N_pad/tp_size = L`；
   - 如果 wo_b 被换成了普通 vLLM `RowParallelLinear`/`*Asymmetric` 子类，
     forward 只有 TP all_reduce，**行数保持 N_pad**，立即触发
     `output[...]` expanded-size。
4. `output` buffer 行数 = 进入 `AscendDeepseekSparseAttention.forward` 时
   `hidden_states.shape[0]`。对称 SP 主模型下它等于 L；不要假设它等于 N_pad。
5. **不要用裁剪 `_apply_wo_b` 结果行数的方式绕过形状错误**。形状不一致
   说明前面的线性层/SP 分发已经错了；裁剪会把 padding 行当有效输出写回，
   表现为“能跑但输出混乱”。正确做法是修复产生 N_pad 行而不是 L 行的
   上游根因（见 §9.5 PluggableLayer 陷阱）。

### 9.2 报错 → 根因快速对照

| 报错 | 第一时间怀疑 |
|---|---|
| `expanded size (1) must match existing size (4)`，位于 `output[...] = _apply_wo_b` | `wo_b` 不是 AscendRowParallelLinear（最常见：模块 linear 类被换成未注册的 `*Asymmetric` 子类），SP reduce_scatter 缺失 |
| `IndexError: Dimension out of range (expected [-2, 1], got 2)`，位于 `npu_transpose_batchmatmul` | `wo_a.weight` 还是 2D（W8A8 未量化路径漏 reshape） |
| `inplace_partial_rotary_mul` / `EZ9999` | `local_cos` 与 q 行数不一致；先查该 forward 是主模型、MoE drafter 还是稠密 drafter，再按 §9.3 判断它实际是否 LCM 分片 |
| 非对称 all_to_all 尺寸/卡死 | `_hetero_head_ratios` 未生效、`n_local_heads/n_local_groups` 仍是均匀值，或 split_sizes 计算错误 |

现场先打这几个值：

```python
forward_ctx = get_forward_context()
print(
    hidden_states.shape[0],           # output buffer 行数
    output.shape[0],
    _EXTRA_CTX.flash_comm_v1_enabled,
    _EXTRA_CTX.flashcomm_v2_enabled,
    forward_ctx.is_draft_model,
    vllm_config.parallel_config.is_heterogeneous_tp,
    self.n_local_heads, self.n_local_groups,
    self.wo_a.weight.dim(),
)
```

### 9.3 `_build_local_token_metadata` 对齐规则

- 非异构：`align = tp_size`，保持 stock 行为。
- 异构**主模型 SP**：`align = lcm(所有 per-DP tp_size)`，例如
  `lcm(3,4,4,4)=12`。
- **draft builder 是否 LCM，取决于 drafter 类型，不能一刀切**（§4.10）：
  - **MoE drafter（DeepSeek-V4 MTP）**：`set_ascend_forward_context` 对
    `is_drafter_moe_model()` 为 True 的 drafter 仍把
    `flash_comm_v1_enabled` 设为 True，hidden stream 会先按
    `lcm(tp_sizes)` padding 再 reduce_scatter；因此 draft metadata **必须
    LCM**，否则 q 行数与 `local_cos` 不一致，报
    `inplace_partial_rotary_mul ... dim0 must be equal`。
  - **稠密 drafter**：`flash_comm_v1_enabled=False`，hidden stream 没有
    LCM padding；此时才跳过 LCM，只按本地 `tp_size` 对齐。
- 区分方式：给 draft 的 `AscendDSACPMetadataBuilder` 打
  `_is_dsa_cp_draft_builder=True`；再由 `is_drafter_moe_model()` 决定
  该 draft builder 是否使用 LCM。

### 9.4 draft builder 的 patch 时机陷阱

- `draft_attn_groups` 及其 metadata builders 是在
  `SpecDecodeBaseProposer.initialize_attn_backend()` 里创建的（runner 的
  `initialize_metadata_builders` 阶段），**不是 `AscendSpecDecodeBaseProposer.__init__`**。
- 所以任何“按 draft builder 分流”的 patch 都不能写在 `__init__` 包装里；
  必须在 `initialize_attn_backend` 的包装中，先调用原实现、再遍历
  `draft_attn_groups[*].metadata_builders` 打标记。
- 当前实现：`patch_hetero_spec_decode._patched_initialize_attn_backend`。

### 9.5 patch 安装方式（容易踩的 Python 坑）

- `target_func.__code__ = new_func.__code__`：
  目标函数对象保留自己的 `__globals__`（目标模块 namespace）。
  适合“整函数复制体”。此时 `new_func` 里引用到的、目标模块没有的名字，
  必须显式注入目标模块 `__dict__`（如 `_align_dsa_o_proj_output`）。
- `TargetClass.method = wrapper_func`：
  wrapper 的 `__globals__` 是当前 patch 模块。适合需要调用 saved original
  的包装器；保存 original 后绝不能再用 `__code__` 替换同一个函数对象，
  否则 saved original 的 `__code__` 会被一起改掉，wrapper 递归调用自己。
- `AscendFusedMoE.__init__` 等带 `__class__` closure（零参 super）的函数，
  不能用没有同样 closure 的新函数 `__code__` 替换；必须整函数绑定。
- **PluggableLayer 注册只按 `cls.__name__` 分发**：`RowParallelLinear`
  会实例化成 `AscendRowParallelLinear`，但 `RowParallelLinearAsymmetric`
  这个名字不在 OOT 注册表里，实例化后是普通 vLLM RowParallel，丢失
  FlashComm1 custom op。**永远不要把模型模块里的 linear 基类全局替换成
  自定义子类**；要扩展行为就 patch `AscendRowParallelLinear` 自身
  （`patch_hetero_ascend_linear.py` 的做法）。

### 9.6 wo_a 3D reshape 的两个口径

- 最终目标布局：`[n_local_groups, input_per_group, o_lora_rank]`，
  由 `weight.view(n_local_groups, o_lora_rank, -1).transpose(1,2).contiguous()` 得到。
- DSA-CP impl 的 `process_weights_after_loading` 用 `self.n_local_groups`
  （`_patched_dsa_cp_init` 已按 ratios 修正）。
- Ascend/VLLM `UnquantizedLinearMethod.process_weights_after_loading` 的
  `_reshape_wo_a_for_dsa()` 里**不能写 `o_groups // tp_size`**：
  DP0 tp=3 + `[2,1,1]` 时 rank0 是 4 组，均匀公式算成 2 会漏 reshape。
  必须 `get_tp_partition_size(o_groups, tp_rank, tp_size, ratios)`，
  并用 `weight.shape[0] == n_local_groups * o_lora_rank` 做防护。
- 检查顺序：`wo_a.weight.dim()`；quant method 是否 unquantized；
  patch 是否在模型加载前生效。

### 9.7 profile_run 包装陷阱

- 包装已有方法前先读原实现：`NPUModelRunner.profile_run` 自己已经做
  PCP 的 `ceil(max/(pcp*2))*2`。wrapper 再预做一次会把
  `100 -> 50 -> 26`。
- 正确姿势：只对原实现没有覆盖的新分支（异构非 PCP 的
  `lcm(tp_sizes)` 下对齐）做预调整，其余原样调用 `_ORIGINAL_PROFILE_RUN`。

### 9.8 对称正常拉起是回归第一站

- 打上 patch 后**先测 DP4TP4 对称普通请求，再触发异构重启**。
- 对称场景下所有 `is_heterogeneous_tp` 分支必须是 no-op；如果对称路径
  报错，优先怀疑无条件应用的外层 patch（forward/metadata/quant method
  包装），而不是异构分支本身。
- 典型例子：§4.3 修完 wo_a 后，对称主模型在 §4.4 报 expanded-size；
  第一次尝试用裁剪 output 绕过，结果输出混乱。真正根因是
  `patch_deepseek_v4.py` 的模块类替换绕过 Ascend pluggable-layer 注册。
  **形状报错时优先查“当前实例化的类是不是 Ascend 类”，不要先裁剪张量。**
- 安装后快速自检（应在模型加载完成后执行，输出 True）：
  ```python
  import vllm_ascend.models.deepseek_v4 as m
  from vllm.model_executor.layers.linear import RowParallelLinear
  from vllm_ascend.ops.linear import AscendRowParallelLinear

  # patch_deepseek_v4 绝不能替换模块全局 linear 类
  assert m.RowParallelLinear is RowParallelLinear
  # 对称/异构的 wo_b 实例都必须是 AscendRowParallelLinear，
  # 否则 SequenceRowParallelOp 的 SP reduce_scatter 缺失。
  layer = next(
      l for l in model.model.layers if hasattr(l, "self_attn")
  ).self_attn
  assert isinstance(layer.wo_b, AscendRowParallelLinear)
  print(True)
  ```
