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

`tp_asymmetric_shardings` 未提供时由插件推导 DeepSeek-V4 golden 拓扑：
`ori_tp` 均摊到 `new_tp` 个 rank，余数给**最靠前**的 rank
（4 -> 3 为 `[2,1,1]`，32/16/16 heads）。显式提供 `[2,1,1]` 结果相同；
`get_tp_asymmetric_shardings` 与 `get_heterogeneous_dp_config` 必须使用同一
余数分配方向，否则 worker 通信域配置与权重/head 切分会不一致。

## 3 重启流程（与旧实现的关键差异）

1. **所有 DP 都重启**：`ITSMultiprocExecutor._strategy_requires_full_restart`
   检测到 `new_tp != tp` 或 `tp_asymmetric_shardings` 时，DEGRADE 和
   PD_REBUILD 都不再区分“故障实例/健康实例”，所有 executor 都走
   `_cleanup_and_restart_workers()`。原因是 MoE EP 通信组和全局
   world_size 从 16 变为 15，健康 DP 只做 KV connector RPC 更新会继续
   使用旧通信域。
2. **跨 DP 重启 barrier**：`_cleanup_and_restart_workers()` 在杀 worker
   之前通过 EngineCore 的 CPU DP group 做 all-reduce rendezvous（默认
   120s 超时）。barrier 使用与 `ParallelConfig.sync_dp_state` 相同的
   2-element SUM collective，因此即使某个 DP 仍在
   `_has_global_unfinished_reqs` 同步中，也不会因 collective 形状不同
   交叉死锁。若决策中心只给故障 DP 下发策略，barrier 会超时并 fail
   closed，而不是让新 15-rank `init_process_group` 无限等待。
3. **P 端 engine_id 轮换**：producer 全量重启时
   `_execute_deployment_strategy` 强制轮换 `kv_transfer_config.engine_id`
   （`<旧id>-<uuid>`）。decode 侧 `KVCacheRecvingThread` 以
   `(engine_id, handshake_port)` 为 key 缓存远端 KV 元数据，轮换后 D 会
   对新 key 重新拉取新 TransferEngine/KV 基地址，避免 P 重启后 D 使用旧
   地址导致 PD 链路失败。
4. **完整异构拓扑注入**：`_update_vllm_config_for_restart` 把
   `heterogeneous_dp_config`、`global_world_size`、`global_start_rank`
   和当前 executor 的 `tp_asymmetric_shardings` 写入
   `additional_config["zero_interrupt_config"]`。
5. **ParallelConfig 扩展**：`vllm/config/parallel.py` 新增
   `HeterogeneousDPConfig` / `heterogeneous_dp_config` /
   `is_heterogeneous_tp` / `get_tp_size_for_dp` /
   `get_sharding_ratios_for_dp` / `get_rank_offset_for_dp`。
6. **全局 rank 偏移**：`WorkerProc.rank/local_rank` 保持 DP 内本地 rank；
   全局 torch.distributed rank（0/3/7/11）由 worker 调用
   `init_distributed_environment` 时在 `is_heterogeneous_tp` 分支按
   `get_rank_offset_for_dp(data_parallel_rank)` 累加前序 DP 的
   `tp_size` 得到。
7. **设备隔离不变**：故障卡从 DP0 的 `ASCEND_RT_VISIBLE_DEVICES` 中
   剔除，DP0 仅剩 3 张健康卡；其他 executor 保持 4 张卡。
8. **RECOVER 同样全量重启**：从 `DP4TP(3,4,4,4)` 恢复对称拓扑也会重建
   world_size 与通信组，因此 `_strategy_requires_full_restart` 在
   `RECOVER` 且当前为异构/当前 tp 与备份 tp 不一致时同样要求全部 DP
   rame 进入 barrier 后一起重启。

> 已知限制：本插件支持“对称启动 -> 策略触发异构重启”。直接以异构拓扑
> 启动（hetero_cp 的 `--heterogeneous-dp-config`）以及 EngineCore 启动期
> 的设备隔离/Ray 路径尚未接入，当前仍依赖每个 executor 独立设置
> `ASCEND_RT_VISIBLE_DEVICES`。

## 3.1 重启端口与幂等边界（场景 2/3 关键）

- **worker-world 端口固定**：ITS executor 初始化时在 DP 端口池中额外保留
  一个固定端口，并在每次 `_init_workers()`（含 scale-to-zero executor）
  重新钉回 `ParallelConfig`。worker 进程不再各自从本地端口列表 pop，避免
  DP16→15 时存活 executor 消费端口、缩零 executor 不消费，导致 RECOVER 时
  恢复卡与其余 15 卡使用不同 TCPStore 端口而永久挂起。
- **barrier 端口**：存活组 barrier 仍使用两槽轮转池；首次 RECOVER 且本
  executor 没有 DEGRADE 备份（说明它可能错过了前一次缩容策略）时，确定性
  选择第二槽端口，并强制重建目标 dp_group。健康 executor 在本次 RECOVER
  轮转一次后，两端端口池重新对齐。
- **重复策略幂等**：DEGRADE / PD_REBUILD 全量重启若目标拓扑与当前
  `heterogeneous_dp_config`/tp/dp 完全一致，执行器只上报成功、不再杀
  worker；RECOVER 只有在本 executor 持有非空 `backup_parallel_config`
  且拓扑一致时才跳过，否则按全量重启处理。没有 DEGRADE 备份的 executor
  （例如故障卡缩容时 POST 失败、RECOVER 时才恢复可达）即使本地 dp/tp
  已经等于 RECOVER 目标，也强制走全量重启 barrier：健康 executor 已经
  重建了目标 dp_group，跳过 rendezvous 会让对端永久等待。
- **PD 元数据只在真实重启后刷新**：幂等跳过的策略不会轮换
  `engine_id`、清空/刷新 scheduler connector 元数据，避免 scheduler 侧
  新 engine_id 与未重启 worker 的旧 engine_id 不一致。

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
     instance” 与 “Full-restart barrier passed”；
  4. 检查 P 端日志出现 “heterogeneous producer restart rotates engine_id”；
  5. 检查 worker 全局 rank：DP0=0/1/2，DP1=3/4/5/6，DP2=7/8/9/10，
     DP3=11/12/13/14；
  6. 请求续推结果与故障前同 seed 基准逐 token 对齐。
