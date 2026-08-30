# vllm_plugins 异构重启适配 v0.23.0 调试备忘录（精简版）

> 本文档已精简：历次已修复问题的长篇幅说明不再保留，只保留当前场景定义、
> 后续开发方向、关键控制面结论与操作入口。如需历史根因，见 `git log` 与
> 对应 commit message。
>
> 适用对象：vllm_plugins 仓 `merge-unified-install` 分支（当前推荐调试分支，
> 已合并 vllm_plugins_0829），vLLM / vllm-ascend v0.23.0。
> 相关测试仓使用 `vllm_plugins_hetero_test` 的 `merge-0829-adapt` 分支。
>
> `hetero` 分支仍是原始 DeepSeek-V4-only 实现；合并说明见
> `vllm_plugins/MERGE_0829.md`。

---

## 0. 场景清单（当前共 3 个 PD 场景）

| 场景 | 初始状态 | 目标状态 | 变化侧 |
|---|---|---|---|
| **场景 1** | P=`DP4TP4`，D=`DP16TP1` | P=`DP4TP(3,4,4,4)`，D=`DP16TP1` | 只动 P：P 坏 1 卡转异构 |
| **场景 2** | P=`DP4TP4`，D=`DP16TP1` | P=`DP4TP4`，D=`DP16TP1 -> DP15TP1` | 只动 D：D 坏 1 卡缩容 |
| **场景 3** | 场景 1+2 后的状态：P=`DP4TP(3,4,4,4)`，D=`DP15TP1` | P=`DP4TP4`，D=`DP16TP1` | P、D 都 RECOVER |

说明：

- 手动直连 executor 和 DecisionMakingCenter 只是**触发方式**，不是新场景编号。
- D 单机版 `run_decode_fault_alone.sh` / `run_decode_recover_alone.sh`
  分别是场景 2 / 场景 3 的子测试。
- **hetero_cp 只适配场景 1**：它直接以 `DP4TP(3,4,4,4)` 启动，等价于
  场景 1 的**目标拓扑**。它**尚未适配**场景 2、场景 3 和
  DecisionMakingCenter 控制面，因此只能作为场景 1 数据面的 golden
  reference；场景 2/3 只能以“恢复后输出与场景 1/2 基线一致”做端到端校验。

---

## 0.1 0829 合并后的关键约定（必读）

本次合并把 `vllm_plugins`（DeepSeek-V4）与 `vllm_plugins_0829`
（Qwen 非对称 / zero-interrupt）合并进同一仓，并按以下方式分流：

| 项 | 结论 |
|---|---|
| 推荐代码分支 | `vllm_plugins` → `merge-unified-install`；测试脚本仓 → `merge-0829-adapt` |
| 运行期开关 | `VLLM_ITS_DEEPSEEK_V4=1` 走 DeepSeek-V4 patch 族；未设置/`0` 走 0829 patch 族 |
| 安装阶段 | `setup.py` 安装**统一替换文件**，**不再读取** `VLLM_ITS_DEEPSEEK_V4`；该变量只影响运行期 |
| DeepSeek-V4 runtime 文件 | `plugins/zero_interrupt/deepseekv4/`（A 完整镜像，绝对导入已改为 `...zero_interrupt.deepseekv4.*`） |
| 0829 runtime 文件 | `plugins/zero_interrupt/` 主目录（默认） |
| 整文件替换源 | 统一文件位于 `plugins/zero_interrupt/vllm/...` 与 `plugins/zero_interrupt/vllm_ascend/...` 主目录，内部按配置/运行期环境分支 |
| `patch_qwen3_5.py` | 已合并为单文件：对称 TP 走 v0.23 原实现，非对称 TP 走 0829 shardings 切分，310P 保持原路径 |

Debug 场景 1/2/3 的 launch 脚本默认导出 `VLLM_ITS_DEEPSEEK_V4=1`；
**手工启动 vllm 时必须自己导出**，否则会静默落到 0829 runtime patch，
DeepSeek-V4 hetero 逻辑不加载。

启动日志中应能看到：

```text
VLLM_ITS_DEEPSEEK_V4=1: applying DeepSeek-V4 patch family
```

如未出现该行，先检查运行期环境变量，而不是查控制面逻辑。

合并分支关系：

```text
vllm_plugins:
  hetero                # 原始 A（DeepSeek-V4-only）
  merge-0829            # 第一步合并：B 主目录 + deepseekv4 子目录
  merge-unified-install # 当前推荐：整文件替换统一 + 运行期分流

vllm_plugins_hetero_test:
  main                  # 原测试脚本
  merge-0829-adapt      # 适配合并后的 launch/install/文档
```

---

## 0.2 测试仓脚本映射（vllm_plugins_hetero_test）

三个场景的**完整测试入口**是 `pd_hetero/` 下的端到端编排脚本，它们自动完成
“健康检查 → 基线请求 → 触发 → 等待恢复 → 复测 → 输出对比”；
`decision_center/` 入口只是给编排脚本设置 `TRIGGER_MODE=dc` 的薄包装，
旧手动触发脚本则只做“下发策略”，不负责请求校验。

| 场景 | 端到端编排（P 节点执行） | 决策中心入口（P 节点） | 手动触发 | D 单机子测试（D 节点） |
|---|---|---|---|---|
| 场景 1 | `pd_hetero/run_scenario1.sh`（`TRIGGER_MODE=manual/dc`） | `decision_center/run_scenario1_dc.sh` | `trigger_hetero_restart.sh` | —（golden 对照用 `hetero_cp/`） |
| 场景 2 | `pd_hetero/run_scenario2.sh`（`TRIGGER_MODE=ssh/local/skip/dc`） | `decision_center/run_scenario2_dc.sh` | `pd_hetero/decode/trigger_decode_fault.sh` | `pd_hetero/decode/run_decode_fault_alone.sh` |
| 场景 3 | `pd_hetero/run_scenario3.sh`（`RECOVER_TARGET=both/prefill/decode`） | `decision_center/run_scenario3_dc.sh` | `trigger_prefill_recover.sh` + `pd_hetero/decode/trigger_decode_recover.sh` | `pd_hetero/decode/run_decode_recover_alone.sh` |
| 全流程 | — | `decision_center/run_all_dc.sh`（场景 1 → 2 → 3） | — | — |

配套脚本（不构成场景编号）：

| 脚本 | 作用 |
|---|---|
| `install_vllm_plugins.sh` | 安装 wheel + 统一整文件替换，并做 import / patch / tool-parser 校验 |
| `launch_prefill_hetero_test.sh` | P 节点拉起对称 `DP4TP4`（手动模式基座，默认无决策中心） |
| `pd_hetero/decode/launch_decode_pd.sh` | D 节点拉起 `DP16TP1` |
| `pd_hetero/common.sh` | 三个编排脚本的公共函数库：HTTP/日志等待、warmup、请求、输出对比、proxy/决策中心/D 端触发封装 |
| `pd_hetero/proxy_instance.py` | PD 代理实例 `add/remove` 工具 |
| `decision_center/launch_prefill_dc.sh` / `launch_decode_dc.sh` | 决策中心模式拉起 P/D 并注册（统一 `VLLM_SERVICE_ID`） |
| `pd_hetero/proxy/start_proxy_pd.sh` + `load_balance_proxy_server.py` | PD 负载均衡代理（P 节点，默认端口 8000） |
| `pd_hetero/send_pd_request.py` | 经代理发请求，输出 `RESULT_TEXT=` / `FINISH_REASON=` |
| `pd_hetero/check_decode_unchanged.sh` | 校验场景 1 中 D 端 16 engine 健康且日志无 restart 记录 |
| `decision_center/trigger_fault.sh` / `repair_devices.sh` | 决策中心通用触发/恢复命令 |

---

## 1. 后续开发方向

### 方向 A：Decode DP16 -> DP15 的异构推理实现与精度问题解决

对应**场景 2**。

已完成：

- 存活 executor 15-rank barrier，不等待故障 dp15；
- EngineCore `dp_group` 缩容后替换为 15-rank 组；
- KV extra config 对原始 dp16 / 当前 dp15 的校验放行；
- scheduler `KVCacheManager` 重启后重建；
- 决策中心触发脚本与动态 decoder 数量发现。

待解决 / A3 复测：

- D 缩容重启后的**真实推理精度**：输出必须与缩容前基线逐 token 一致；
- 故障 executor **永久不可达**场景的独立回归（当前 barrier 已不依赖它）；
- D 端 Mooncake `(engine_id, handshake_port)` 轮换链路验证；
- 决策中心若因整除等约束选择其它合法 DP 数，需确认端到端输出仍一致。

### 方向 B：PD 分离场景的异构实现

对应**场景 1 + 场景 2 的跨节点 PD 链路**。

已完成：

- 场景 1 脚本：P 转异构、D 不变，代理 warmup 覆盖全部 decoder；
- 场景 2 脚本：D 缩容、P 不变，代理摘除故障 decoder；
- DecisionMakingCenter 触发方式；
- P 端 `engine_id` 轮换与 D 端 Mooncake 元数据恢复 patch。

待解决 / A3 复测：

- 场景 1 异构重启后输出与对称基线、hetero_cp golden 逐 token 对比；
- PD 代理 recompute 拼接、全 decoder warmup、输出污染问题的最终确认；
- D 端按新 `(engine_id, handshake_port)` 恢复 KV 传输链的完整验证；
- 场景 2 D 缩容后的 PD 链路输出一致性验证。

### 方向 C：RECOVER 场景的实现

对应**场景 3**。

已完成：

- P：`DP4TP(3,4,4,4) -> DP4TP4` RECOVER；
- D：`DP15TP1 -> DP16TP1` RECOVER，恢复 executor 15；
- 纯 DP 恢复也走全量 barrier，并重建包含全部 16 个 rank 的目标
  EngineCore `dp_group`；
- 重复 RECOVER 的幂等保护；
- DecisionMakingCenter `/repair/devices` 一键恢复脚本。

待解决 / A3 复测：

- `RECOVER_TARGET=both/prefill/decode` 三种范围的真实验证；
- 恢复后 P、D 输出均与降级前基线一致；
- 恢复后的 executor 15 必须参与 `Full-restart barrier passed`（不是
  skipped），且 16 个 D engine `/health` 全部就绪；
- RECOVER 后再次 DEGRADE 的连续扩缩容稳定性。

---

## 2. 当前状态（截至 2026-08-29）

| 项目 | 状态 |
|---|---|
| 0829 合并 + 统一整文件替换 + 运行期 patch 族开关 | ✅ 已提交（`merge-unified-install` / `merge-0829-adapt`），待 A3 双路径复测 |
| 对称 DP4TP4 正常拉起 + 普通请求 | ✅ 已验证 |
| hetero_cp 直接拉起场景 1 异构拓扑 + 推理 | ✅ 输出正确；**仅覆盖场景 1** |
| 场景 1 控制面（P 全量重启、KV 重建、Mooncake 恢复） | 🔧 代码已提交，待 A3 最终复测 |
| 场景 2 控制面（D 缩容、存活组 barrier） | 🔧 代码已提交，待 A3 推理与精度复测 |
| 场景 3 RECOVER 控制面 | 🔧 代码已提交，待 A3 复测 |
| DecisionMakingCenter 适配 | 🔧 executor_id/role/状态上报已对齐，待 A3 联调 |
| 重启后 scheduler KVCacheManager 重建 | 🔧 代码已提交，待 A3 复测 |
| 旧手动触发脚本兼容性 | ✅ 保留：数字 executor_id 与 `exe-...` id 均接受 |

已修复问题的历史根因不再在本文展开，需要时查看：

```bash
git -C vllm_plugins log --oneline --all
git -C vllm_plugins_hetero_test log --oneline --all
```

---

## 3. 代码仓与分工

| 路径 | 作用 |
|---|---|
| `vllm_plugins/` | 当前适配仓。推荐分支 `merge-unified-install`；原始 A 分支 `hetero`；中间合并分支 `merge-0829` |
| `vllm_plugins_0829/` | 0829 并行开发仓（已合并，保留作对照） |
| `vllm_plugins_origin/` | 老代码基线 |
| `hetero_cp/` | 参考 demo；**只覆盖场景 1** 目标拓扑，直接改 vllm/vllm-ascend 源码 |
| `origin_0.23.0/` | vllm/vllm-ascend v0.23.0 官方基线 |
| `origin_0.18.0/` | v0.18.0 基线 |
| `vllm_plugins_hetero_test/` | A3 安装 / 拉起 / 触发 / 校验脚本；推荐分支 `merge-0829-adapt` |
| `DecisionMakingCenter/` | 决策中心参考代码（本次适配**未修改**） |
| `DeepSeek-V4-Flash-w8a8-mtp/` | 模型配置样例 |

场景 1 数据面对比命令：

```bash
git -C hetero_cp/vllm diff 0fc695fc6d..HEAD
git -C hetero_cp/vllm-ascend diff 5cb98caaa..HEAD
git -C vllm_plugins log --oneline --all --decorate -20
git -C vllm_plugins_hetero_test log --oneline --all --decorate -10
```

---

## 4. 关键控制面结论

1. **所有拓扑变化都必须全量重启**：场景 1/2/3 的 DEGRADE / PD_REBUILD /
   RECOVER 均由全部相关 executor 同时重启 worker。
2. **barrier 与 DP 业务同步同形**：`sync_dp_state`、全量重启 barrier 均
   使用 2×int32 SUM collective，避免跨 collective 交叉死锁。
3. **scale-to-zero 不参与 barrier**：DP16→15 时存活 executor 用重新编号
   的 15-rank gloo 组；故障 executor 跳过 barrier，稍后自行清理进入
   `Idle mode (dp=0)`。
4. **EngineCore dp_group 必须与目标拓扑一致**：
   - DP16→15：替换为 15-rank 组；
   - DP15→16 RECOVER：重建 16-rank 目标组后统一 barrier；
   - P 异构 TP 恢复（DP 数不变）：沿用原 4-rank 组。
5. **worker 全局 rank 重算**：异构场景按 `get_rank_offset_for_dp()` 累加
   （0/3/7/11），不能用 `dp_rank * tp_size` 均匀公式。
6. **DSA-CP 非对称 head 切分**：`n_local_heads/n_local_groups` 与
   all_to_all split 都必须按 `tp_sharding_ratios`；P 4→3 时决策中心未给
   ratios，插件 fallback 为 golden 的 `[2,1,1]`。
7. **Mooncake 异构端口偏移**：同样使用累计 offset；P 全量重启强制轮换
   `engine_id`，D 按新 `(engine_id, handshake_port)` 重新拉取元数据。
8. **DecisionMakingCenter 适配**：
   - 注册时保存返回的 `exe-<service>-<engine>-<n>` executor id；
   - HTTP deploy 同时接受数字 id 与 `exe-...` id；
   - 部署状态优先上报 `exe-...` id；
   - 上报 role 使用 `P_ROLE` / `D_ROLE`（不是 `kv_producer/consumer`）；
   - 同一服务的所有 executor 必须使用同一个 `VLLM_SERVICE_ID`。
9. **0829 合并后的 patch 族选择**：
   - 安装阶段不选族；`VLLM_ITS_DEEPSEEK_V4` 只影响运行期。
   - DeepSeek-V4 场景必须 `VLLM_ITS_DEEPSEEK_V4=1`，并在启动日志确认
     `applying DeepSeek-V4 patch family`。
   - 整文件替换已统一，切换 0829/DeepSeek-V4 不需要重装，只需改环境变量
     重启服务。

---

## 5. 操作入口速查

### 5.1 安装与拉起（手动模式）

```bash
# 安装：不再需要 VLLM_ITS_DEEPSEEK_V4（setup.py 安装统一替换文件）
cd /opt/its/z30055003/vllm_plugins_hetero_test
bash install_vllm_plugins.sh

# P 节点：对称 DP4TP4（脚本默认运行期 VLLM_ITS_DEEPSEEK_V4=1）
bash launch_prefill_hetero_test.sh

# D 节点：DP16TP1（脚本默认运行期 VLLM_ITS_DEEPSEEK_V4=1）
bash pd_hetero/decode/launch_decode_pd.sh
```

手工启动时（不经 launch 脚本）必须显式导出：

```bash
export VLLM_ITS_DEEPSEEK_V4=1
export VLLM_CUSTOM_PATCHES=zero_interrupt
python3 -m vllm.entrypoints.openai.api_server ...
```

### 5.2 决策中心模式

```bash
# P 节点 7.246.78.75
bash decision_center/launch_prefill_dc.sh

# D 节点 7.246.78.76
bash decision_center/launch_decode_dc.sh

# 场景 1（P 故障）：在 P 节点
DECODE_HOST=7.246.78.76 bash decision_center/run_scenario1_dc.sh

# 场景 2（D 故障）：在 P 节点
bash decision_center/run_scenario2_dc.sh

# 场景 3（RECOVER）：在 P 节点
bash decision_center/run_scenario3_dc.sh

# 全流程
DECODE_HOST=7.246.78.76 bash decision_center/run_all_dc.sh
```

通用触发命令：

```bash
# 故障
bash decision_center/trigger_fault.sh <node_ip> <npu_id>

# 恢复：所有坏卡必须在一次请求中上报
bash decision_center/repair_devices.sh <node_ip>:<npu_id> ...
```

### 5.3 关键日志检查

```bash
# patch 族路由（每个进程都应出现；没出现说明环境变量未设置）
grep -R "applying DeepSeek-V4 patch family" logs/prefill/ logs/decode/

# 决策中心分配 executor id（P/D 每个进程都应出现）
grep -R "assigned executor_id" logs/prefill/ logs/decode/

# 场景 1：P 全量重启与 KV 恢复
grep -R "Full-restart barrier passed" logs/prefill/
grep -R "KV connector metadata updated successfully" logs/prefill/

# 场景 2：健康 dp0-14 passed；故障 dp15 skipped 后进入 Idle
grep -R "Full-restart barrier passed" logs/decode/
grep -R "Full-restart barrier skipped" logs/decode/
grep -R "Idle mode (dp=0)" logs/decode/

# 场景 3：16 个 D executor 全部 passed（恢复的 dp15 不是 skipped）
grep -R "Full-restart barrier passed" logs/decode/
```

### 5.4 旧手动触发脚本（仍兼容）

以下脚本只负责“向 executor 下发策略”，不带请求与输出校验；完整测试优先用
5.5 的 `pd_hetero/run_scenario*.sh`：

```bash
bash trigger_hetero_restart.sh                # 场景 1 手动
bash trigger_prefill_recover.sh               # 场景 3 P 恢复手动
bash pd_hetero/decode/trigger_decode_fault.sh   # 场景 2 手动
bash pd_hetero/decode/trigger_decode_recover.sh # 场景 3 D 恢复手动
```

### 5.5 PD 端到端编排脚本（场景完整测试入口）

- **场景 1 `pd_hetero/run_scenario1.sh`**（P 节点）：
  1. 健康检查：P 4 个 engine + D 16 个 engine + 代理 `/healthcheck`；
  2. PD 链路 warmup 覆盖全部 16 个 decoder，吸收首次 `recomputed`，
     防止“正确文本 + 无关数据”的拼接污染；
  3. 基线请求（temperature=0 贪心）存 `logs/pd_scenario1/pre_hetero.json`；
  4. 触发 P `DP4TP(3,4,4,4)`（`manual` 直连 executor / `dc` 决策中心）；
  5. 等待 P 4 个 `/health` + `Full-restart barrier passed` +
     `KV connector metadata updated successfully`；
  6. `check_decode_unchanged.sh` 证明 D 端 16 engine 未被重启；
  7. 复测请求存 `post_hetero.json`，与基线 `choices[0].text` 默认完全一致。

- **场景 2 `pd_hetero/run_scenario2.sh`**（P 节点）：
  1. 健康检查 P 对称 + D `DP16TP1`，发基线请求；
  2. 记录 P 端 restart 日志计数（用于证明 P 未变）；
  3. 触发 D 降级：`ssh` / `local` / `skip`（已在 D 节点触发过）/
     `dc`（决策中心）；故障 executor 缩到 `new_dp=0/new_tp=0`；
  4. `dc` 模式下动态探测实际存活 decoder 数（决策中心可能因专家数整除
     约束选择其它合法 DP 数），并摘除全部不可用 decoder；
  5. 从代理摘除故障 decoder，warmup 覆盖存活 decoder，发复测请求；
  6. 对比两次输出，并确认 P 端 restart 计数无新增。

- **场景 3 `pd_hetero/run_scenario3.sh`**（P 节点）：
  1. 前置校验：P 4 engine 在线（异构亦可）、D 15 健康 + 故障 executor
     空转、代理在线；
  2. P RECOVER `DP4TP(3,4,4,4) -> DP4TP4`，等待 barrier + KV 元数据恢复，
     并确认 D 的 15 个 decoder 在 P 轮换 engine_id 后仍健康；
  3. D RECOVER `DP15TP1 -> DP16TP1`：向全部 16 个 executor 下发，恢复的
     executor 15 必须 `Full-restart barrier passed`（不是 skipped）；
  4. 恢复的 decoder 15 加回代理，warmup 覆盖当前全部 decoder，发复测请求；
  5. 按 `RECOVER_TARGET` 自动选择基线对比：
     `both` → 场景 1 的 `pre_hetero.json`；`prefill` → 场景 2 的
     `post_decode_fault.json`；`decode` → 场景 1 的 `post_hetero.json`；
     也可用 `BASELINE_OUTPUT` 显式指定。

- **D 单机子测试**（D 节点，不依赖 P 和代理）：
  `run_decode_fault_alone.sh` = 场景 2 的“基线 → 降级 → 15 卡健康 →
  复测对比”；`run_decode_recover_alone.sh` = 场景 3 的“15 健康 + 1 idle →
  RECOVER → 16 卡健康 → 对比 `decode_fault_alone/pre_fault.json`”。

- **决策中心薄包装**：`decision_center/run_scenario{1,2,3}_dc.sh` 只是设置
  `TRIGGER_MODE=dc` 及节点/IP 后调用对应的 `pd_hetero/run_scenario*.sh`；
  `run_all_dc.sh` 按 1 → 2 → 3 串联。决策中心模式要求 P/D 的 20 个
  executor 使用同一个 `VLLM_SERVICE_ID`（默认 `pd-hetero-service`），
  场景 3 恢复 both 时必须在一次 `repair/devices` 请求中上报全部坏卡。

### 5.6 测试脚本关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_ITS_DEEPSEEK_V4` | 1（所有 launch 脚本） | DeepSeek-V4 patch 族；手工启动时必须自己导出 |
| `DECODE_HOST` | 必填 | decode 节点 IP |
| `SSH_DECODE` | 空 | `TRIGGER_MODE=ssh` 时必填，如 `root@<ip>` |
| `TRIGGER_MODE` | 场景1 `manual`；场景2/3 `ssh` | `manual` 直连 executor；`ssh` / `local` / `skip`（场景2）；`dc` 决策中心 |
| `RECOVER_TARGET` | `both` | 场景3 恢复范围：`both` / `prefill` / `decode` |
| `FAULT_NPU` | 3 | 场景1 故障卡，必须属于 DP0 的 NPU 0..3 |
| `DECODE_FAULT_NPU` | 15 | 场景2/3 故障卡，范围 0..15 |
| `REQUIRE_OUTPUT_MATCH` | 1 | 1=异构/缩容/恢复前后输出必须完全一致；0=仅要求非空 |
| `BASELINE_OUTPUT` | 按 `RECOVER_TARGET` 自动选择 | 场景3 对比基线 JSON 路径 |
| `WARMUP_REQUESTS` | 场景1=16；场景2 基线=16、降级后=15 | 成功预热请求数，必须覆盖全部 active decoder |
| `WARMUP_RETRIES` / `WARMUP_INTERVAL` | 30 / 10 | 预热仍 recompute/失败时的额外重试次数与间隔（秒） |
| `START_PREFILL` / `START_PROXY` | 1 | 编排脚本是否自动拉起 P / 代理 |
| `RESTART_TIMEOUT` | 900 | 等待全量重启 / KV 恢复的超时秒数 |
| `DECISION_CENTER_URL` | `http://7.246.78.79:8088` | `TRIGGER_MODE=dc` 时使用 |
| `VLLM_SERVICE_ID` | `pd-hetero-service`（dc launch 脚本） | P/D 所有 executor 必须一致 |
| `PROXY_PORT` | 8000 | PD 负载均衡代理端口（P 节点） |

---

## 6. 关键实现文件

路径均相对 `vllm_plugins/vllm_custom_plugins/plugins/zero_interrupt/`。

### 6.1 DeepSeek-V4 runtime patch（`deepseekv4/` 镜像）

| 文件 | 作用 |
|---|---|
| `deepseekv4/vllm/v1/executor/its_multiproc_executor.py` | worker 重启、策略执行、存活组 barrier、dp_group 替换 |
| `deepseekv4/vllm/v1/engine/engine_core_patch.py` | busy loop 消费策略、DP 同步、engine_id 轮换 |
| `deepseekv4/vllm/v1/executor/http_server.py` | 策略 HTTP，接受数字/`exe-...` executor id |
| `deepseekv4/common/communication/decision_center_client.py` | 注册、状态上报、executor id 解析 |
| `deepseekv4/vllm/v1/executor/utils.py` | barrier geometry、RECOVER 判定、sharding fallback |
| `deepseekv4/vllm_ascend/ops/patch_hetero_ascend_linear.py` | 非对称 TP 权重 scaffolding 与 ratio-aware loader |
| `deepseekv4/vllm_ascend/attention/patch_deepseek_v4_attention_hetero.py` | DSA-CP 非对称 head / o_proj 恢复 |
| `deepseekv4/vllm_ascend/ops/fused_moe/patch_hetero_moe.py` | MoE Prepare/Finalize、256 experts 余数分布 |
| `deepseekv4/vllm_ascend/distributed/kv_transfer/patch_hetero_mooncake.py` | Mooncake 异构端口映射、engine_id 轮换 |
| `deepseekv4/vllm_ascend/patch/patch_hetero_tp.py` | forward context 别名刷新、per-DP 布局 |

### 6.2 setup.py 统一整文件替换源（主目录，安装期不选族）

| 文件 | 作用 |
|---|---|
| `vllm/config/parallel.py` | `HeterogeneousDPConfig`（A）+ `world_size_across_dp` override（B） |
| `vllm/distributed/parallel_state.py` | v0.23 组网 + 0829 `asym_world_size/get_global_rank_asym/init_distributed_environment_asym/asym=True` 组网 |
| `vllm/model_executor/layers/fused_moe/config.py` | hetero TP / 0829 `zero_interrupt_config` / 对称公式三分支 |
| `vllm/v1/core/patch_kv_cache_utils.py` | v0.23 KV cache + 0829 mixed-page_size / 非对称投影 |
| `vllm_ascend/distributed/parallel_state.py` | A 严格超集，含 `init_ascend_model_parallel_asym` |
| `vllm_ascend/worker/worker.py` | DeepSeek-V4 hetero 路径 + 0829 legacy asym 路径 + mamba KV 修复 |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 单文件：对称走 v0.23，非对称走 0829 shardings |
| `vllm_ascend/ops/triton/rotary_embedding.py` | v0.23 同源 + 0829 `its_rotary` 开关 |

---

## 7. 回归铁律

1. **先测对称 DP4TP4，再触发任何异构/RECOVER**。对称路径必须完全无感。
2. **不要用“裁剪/丢弃多出的行/只取 rank0”压形状错误**。形状不一致说明
   上游切分或 linear 实例化已错，裁剪只会把错误固化。
3. **DSA-CP 里 `wo_b` 必须是 `AscendRowParallelLinear`**，否则丢失 SP
   reduce_scatter。
4. **MoE drafter 要 LCM 对齐，稠密 drafter 不 LCM**；draft 错误只影响
   接受率/耗时，不能成为最终输出错误的根因。
5. **修改 forward-context / metadata 相关 patch 后，检查已导入模块别名**
   （先 import 后 patch 会导致旧函数别名残留）。
6. **重复 trigger / RECOVER 必须幂等**：重复策略不能静默丢弃，也不能
   在无 barrier 保护下重复杀 worker。
7. **改动任何场景后，旧手动触发脚本也要回归**：数字 executor_id 路径
   必须仍然可用。
8. **合并仓调试 DeepSeek-V4 前，先确认运行期 patch 族**：`VLLM_ITS_DEEPSEEK_V4`
   必须是 `1`，且日志出现 `applying DeepSeek-V4 patch family`；安装阶段
   不再设置该变量。
