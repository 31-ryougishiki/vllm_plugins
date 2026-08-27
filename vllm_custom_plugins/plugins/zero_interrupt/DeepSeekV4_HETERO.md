# DeepSeek-V4 异构重启设计（DP4TP4 -> DP4TP(3,4,4,4)）

## 1 背景

参考 `hetero_cp` 仓在 vllm/vllm-ascend v0.23.0 上的直拉异构 demo，在
`vllm_plugins` 的 `zero_interrupt` 插件中实现：

- 单节点 prefill `DP4TP4` 故障一张卡后，以
  `DP4TP(3,4,4,4)` 重启全部 4 个 DP executor；
- 全局 world_size 为 15（3+4+4+4），全局 rank 布局为
  DP0:[0,1,2]、DP1:[3..6]、DP2:[7..10]、DP3:[11..14]；
- PD 分离场景中 P 端按上述异构拓扑重启，D 端仍为 dp16/tp1，
  MooncakeHybridConnector 按绝对 handshake port 重映射 KV 传输链；
- DeepSeek-V4 使用 `enable_dsa_cp` + MTP + `enable_shared_expert_dp` 的
  服务配置，DP0 的 tp=3 使用 `tp_asymmetric_shardings=[2,1,1]`。

## 2 策略输入

决策中心下发 PD_REBUILD（或 DEGRADE）策略的 `engine_parallel_config`
需要覆盖所有 DP rank，并携带 `tp_asymmetric_shardings`：

```json
{
  "deploy_type": "PD_REBUILD",
  "executor_id": "0",
  "engine_parallel_config": [
    {"executor_id":"0","dp":4,"tp":4,"data_parallel_rank":0,
     "new_dp":4,"new_tp":3,"enable_expert_parallel":true,
     "tp_asymmetric_shardings":[2,1,1]},
    {"executor_id":"1","dp":4,"tp":4,"data_parallel_rank":1,
     "new_dp":4,"new_tp":4,"enable_expert_parallel":true},
    {"executor_id":"2","dp":4,"tp":4,"data_parallel_rank":2,
     "new_dp":4,"new_tp":4,"enable_expert_parallel":true},
    {"executor_id":"3","dp":4,"tp":4,"data_parallel_rank":3,
     "new_dp":4,"new_tp":4,"enable_expert_parallel":true}
  ]
}
```

`tp_asymmetric_shardings` 未提供时沿用 vllm_plugins 旧逻辑：
`ori_tp` 均摊到 `new_tp` 个 rank，余数给最后一个 rank
（4 -> 3 为 `[1,1,2]`）。DeepSeek-V4 64 heads 推荐显式使用
`[2,1,1]`（32/16/16 heads）。

## 3 重启流程（与旧实现的关键差异）

1. **所有 DP 都重启**：`ITSMultiprocExecutor._strategy_requires_full_restart`
   检测到 `new_tp != tp` 或 `tp_asymmetric_shardings` 时，PD_REBUILD
   下不再区分“故障实例/健康实例”，所有 executor 都走
   `_cleanup_and_restart_workers()`。原因是 MoE EP 通信组和全局
   world_size 从 16 变为 15，健康 DP 只做 KV connector RPC 更新会继续
   使用旧通信域。
2. **完整异构拓扑注入**：`_update_vllm_config_for_restart` 把
   `heterogeneous_dp_config`、`global_world_size`、`global_start_rank`
   和当前 executor 的 `tp_asymmetric_shardings` 写入
   `additional_config["zero_interrupt_config"]`。
3. **ParallelConfig 扩展**：`vllm/config/parallel.py` 新增
   `HeterogeneousDPConfig` / `heterogeneous_dp_config` /
   `is_heterogeneous_tp` / `get_tp_size_for_dp` /
   `get_sharding_ratios_for_dp` / `get_rank_offset_for_dp`。
4. **全局 rank 偏移**：`WorkerProc.rank/local_rank` 保持 DP 内本地 rank；
   全局 torch.distributed rank（0/3/7/11）由 worker 的
   `init_distributed_environment_asym -> get_global_rank_asym` 按
   `data_parallel_rank` 累加前序 DP 的 `new_tp` 得到。
5. **设备隔离不变**：故障卡从 DP0 的 `ASCEND_RT_VISIBLE_DEVICES` 中
   剔除，DP0 仅剩 3 张健康卡；其他 executor 保持 4 张卡。

## 4 DeepSeek-V4 patch 清单

所有 DeepSeek-V4 相关修改均以运行时 patch 实现，目录名与
vllm/vllm-ascend 源码对应：

| 文件 | 作用 |
|------|------|
| `vllm_ascend/models/patch_deepseek_v4.py` | 绑定非对称 linear 类；MoE shared expert 复制；MoE 余数专家分布；attention head/group 非对称；load_weights sink 分片 |
| `vllm_ascend/models/patch_deepseek_v4_mtp.py` | MTP load_weights sink 分片 |
| `vllm_ascend/attention/patch_deepseek_v4_attention_hetero.py` | DSA v1/DSA-CP 本地 head、LCM padding、o_proj 非对称 all_to_all |
| `vllm_ascend/ops/fused_moe/patch_hetero_moe.py` | Prepare/Finalize、token dispatcher、selector，以及 v0.23 `AscendFusedMoE.__init__` 的 256/15 余数专家分布 |
| `vllm/model_executor/layers/fused_moe/runner/patch_hetero_moe_runner.py` | MoERunner shared/routed 输出长度补齐 |
| `vllm_ascend/ops/patch_hetero_custom_ops.py` | 已注册 MoE 自定义算子的异构 gather/reduce/unpad 实现 |
| `vllm_ascend/patch/patch_hetero_tp.py` | forward context 的 per-DP padded lengths、LCM MC2 capacity、A3 回退 ALLGATHER |
| `vllm_ascend/worker/patch_hetero_model_runner.py` | DP metadata 经 EP group 同步；profile_run LCM 对齐 |
| `vllm_ascend/spec_decode/patch_hetero_spec_decode.py` | 异构下 MTP 复用目标 per-DP TP group |
| `vllm/model_executor/layers/patch_hetero_vocab.py` | vocab/logits 维度 LCM padding |
| `vllm/model_executor/layers/patch_hetero_parameter.py` | v2 参数加载器（column/merged/row）按非对称 offset 切 checkpoint |
| `vllm/model_executor/model_loader/patch_hetero_default_loader.py` | EP weight filter 的异构 ep_size/ep_rank |
| `vllm/config/patch_speculative_hetero.py` | draft ParallelConfig 继承异构拓扑 |
| `vllm_ascend/distributed/kv_transfer/patch_hetero_mooncake.py` | MooncakeHybridConnector 异构 TP 端口偏移、绝对 port 映射、producer rank 选择 |

## 5 验证

- 单元测试：`vllm/v1/executor/tests/test_hetero_utils.py`
- NPU 单机集成验证（prefill DP4TP4 -> DP4TP(3,4,4,4)）：
  1. 启动 16 卡 DP4TP4 DeepSeek-V4-Flash-w8a8-mtp + decode dp16/tp1；
  2. 注入一张卡故障，决策中心下发第 2 节策略；
  3. 检查 4 个 executor 日志均出现 “restarting workers of EVERY DP
     instance”；
  4. 检查 worker 全局 rank：DP0=0/1/2，DP1=3/4/5/6，DP2=7/8/9/10，
     DP3=11/12/13/14；
  5. 请求续推结果与故障前同 seed 基准逐 token 对齐。
