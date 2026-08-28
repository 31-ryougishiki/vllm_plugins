# vllm_plugins 异构重启适配 v0.23.0 调试备忘录

> 目的：沉淀背景、关键结论和历次问题根因，便于后续在 A3 节点继续调试。
> 适用对象：vllm_plugins 仓 `hetero` 分支，vLLM / vllm-ascend v0.23.0。

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

### 4.4 DSA-CP 输出写回 `expanded size (1) must match (4)`（已修复）

- 场景：**对称 DP4TP4 正常拉起服务、尚未触发异构重启**，打上插件 patch 后
  发普通请求（MTP decode/profiling）即复现。不是只有异构重启后才出现。
- 现象：`dsa_cp.py` 的 `output[...] = self._apply_wo_b(...)` 报
  `RuntimeError: The expanded size of the tensor (1) must match the
  existing size (4) at non-singleton dimension 0. Target sizes: [1, 4096].
  Tensor sizes: [4, 4096]`。
- 根因（对称场景就成立）：
  - `_restore_tp_head_layout()` 返回的是 **全量 padded token 流**
    （TP=4、1 个 decode token 时是 4 行）。SP 场景下 `wo_b` 的
    `matmul_and_reduce` 会做 token 维 reduce_scatter，把 4 行收敛回
    output buffer 的 1 行；但 MTP draft 前向
    `flash_comm_v1_enabled=False`，`wo_b` 退回
    `tensor_model_parallel_all_reduce`，**保留 4 行**。
  - `AscendDeepseekSparseAttention.forward` 按当前 `hidden_states`
    分配 output buffer（draft 不带 SP padding，只有 1 行），写回时形状冲突。
  - 之所以在“打 patch 后”才看到该错误，是因为 §4.3 先把 2D `wo_a`
    reshape 成 3D，前向得以越过 `npu_transpose_batchmatmul` 的
    `IndexError`，才暴露出下一处形状不匹配。
  - 异构 LCM 部分是**另一层**隐患：patch 的 `_build_local_token_metadata`
    曾无条件把 `num_input_tokens` 对齐到 `lcm(3,4,4,4)=12`。异构重启后
    MTP draft 前向同样没有 LCM padding，会让 local_cos / attention 输出
    比真实 hidden 更宽，所以一并修正。
- 修复：
  - `patch_deepseek_v4_attention_hetero.py` 新增
    `AscendDSACPImpl.forward` 的整函数 patch：写回 output 前用
    `_align_dsa_o_proj_output()` 把 o_proj 结果裁剪/补零到 output 的
    `shape[0]`。
  - `AscendSpecDecodeBaseProposer.initialize_attn_backend` 的 patch 在
    draft attention groups/builder 创建后给 DSA-CP draft builder 打上
    `_is_dsa_cp_draft_builder` 标记；`_build_local_token_metadata` 对
    draft builder 不再做 LCM 对齐，保持原始的本地 `tp_size` 对齐。
  - `patch_hetero_ascend_linear._reshape_wo_a_for_dsa()` 改为按
    `tp_asymmetric_shardings` 计算 `n_local_groups`（DP0 [2,1,1] 下是
    4/2/2，而不是统一 `8//3=2`），避免 DP0 rank0 的 wo_a 漏 reshape。
  - 顺手修复 `patch_hetero_model_runner._patched_profile_run()`：PCP
    分支原本在 wrapper 和原始 `profile_run()` 里各做一次 token 对齐，
    导致 `max_num_tokens` 被连续两次向下取整；现在 PCP 完全交给原始
    实现处理，只对异构非 PCP 场景预对齐 `lcm(tp_sizes)`。

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

# 3. 先发一个普通请求，确认对称推理正常
#    （重点确认 DSA-CP 的 npu_transpose_batchmatmul 不再报 IndexError）

# 4. 触发异构重启：DP4TP4 -> DP4TP(3,4,4,4)
bash trigger_hetero_restart.sh

# 5. 观察日志关键字
grep -R "restarting workers of EVERY DP instance" logs/prefill/
grep -R "Full-restart barrier passed" logs/prefill/
grep -R "heterogeneous producer restart rotates engine_id" logs/prefill/

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
| `RuntimeError: expanded size (1) must match existing size (4)`，位于 `dsa_cp.py` 的 `output[...] = _apply_wo_b(...)` | 对称拉起时 MTP draft 前向无 SP padding：restore 返回全量 padded token 流，`wo_b` 退化为 all_reduce 不收敛行数；异构重启后 draft metadata 还会被 LCM 对齐 | 已修复 |
| 异构重启后挂死 | DP 同步 / barrier collective 形状不匹配 | 已修复（同形 2×int32 SUM） |
| 只给故障 DP 下发策略 | barrier 120s 超时 fail-closed，EngineCore 退出 | 设计如此；决策中心必须广播完整策略 |
