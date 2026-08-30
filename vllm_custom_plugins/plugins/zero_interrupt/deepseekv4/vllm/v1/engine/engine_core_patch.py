#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""EngineCore patch for ITS plugin.

本模块提供 monkey-patch 功能，将 ITS 部署策略执行集成到 EngineCore 的 busy loop 中。

设计思路：
- 策略通过 VllmConfig.additional_config 传递
- 故障时：Worker 捕获异常，设置标志，等待策略
- 超时时：重启失败的 workers
- 健康的 workers 继续运行
"""

import time
from typing import Any, cast
from concurrent.futures import Future
import torch
from torch.distributed import ProcessGroup

from vllm.logger import logger
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreRequestType, EngineShutdownState
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import RequestStatus
from vllm.v1.outputs import ModelRunnerOutput
from datetime import timedelta


from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.constants import VLLM_ITS_STRATEGY_TIMEOUT, VLLM_ITS_MAX_RETRY_COUNT
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.types import DeployType, UpdateEngineInfo
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
    get_pd_scheduler_connector_topology,
)

# 模块级变量，用于跟踪回调是否已注册
_callback_registered = False
# 标记是否已发送空转通知，避免重复发送
_idle_notification_sent = False

def is_pd_separated(vllm_config) -> bool:
    return (vllm_config.kv_transfer_config is not None and 
            (vllm_config.kv_transfer_config.kv_role == "kv_producer" or vllm_config.kv_transfer_config.kv_role == "kv_consumer"))

def sync_dp_state_fast_timeout(
    dp_group: ProcessGroup,
    has_unfinished: bool,
    pending_pause: bool,
    timeout_seconds: float = 10,
) -> tuple[bool, bool] | None:
    """v0.23 ``ParallelConfig.sync_dp_state`` with a bounded wait.

    Uses the exact same two-element SUM all-reduce shape as
    ``ParallelConfig.sync_dp_state`` so it remains interchangeable with the
    full-restart barrier and with other DP ranks that are still inside this
    same collective.  Returns ``(has_unfinished_global, pause_consensus)``,
    or ``None`` when the collective timed out and had to be abandoned.
    """
    tensor = torch.tensor(
        [int(has_unfinished), int(pending_pause)],
        dtype=torch.int32,
        device="cpu",
    )
    work = torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group,
        async_op=True,
    )
    completed = False
    try:
        completed = bool(work.wait(timeout=timedelta(seconds=timeout_seconds)))
    except Exception as e:
        logger.info(
            "[sync_dp_state_fast_timeout] all_reduce wait raised: %s", e
        )
    if not completed:
        # Do not leave a live collective behind: the next DP collective on
        # this group could otherwise pair with the abandoned one and corrupt
        # both results.
        try:
            work.abort()
        except Exception as abort_exc:  # noqa: BLE001
            logger.warning(
                "[sync_dp_state_fast_timeout] failed to abort timed-out "
                "all_reduce: %s",
                abort_exc,
            )
        return None

    dp_size = int(dp_group.size())
    pause_count = int(tensor[1].item())
    has_unfinished_global = int(tensor[0].item()) > 0 or (
        pause_count % dp_size != 0
    )
    pause_consensus = pause_count == dp_size
    return has_unfinished_global, pause_consensus

def patch_engine_core() -> None:
    """Patch EngineCore._handle_shutdown 以支持部署策略执行。

    流程如下：
    1. 检查 RecvNewDeployment 信号是否到达
       - 如果未收到：检查 WaitNewDeployment
         - 也没有收到：继续正常服务
         - 收到了：等待 RecvNewDeployment（带超时）
           - 超时：重启 workers
           - 未超时：执行部署策略
       - 如果收到了：直接执行部署策略
    2. 恢复 Worker Monitor 线程
    3. 继续循环
    """
    try:
        from vllm.v1.engine.core import EngineCoreProc, DPEngineCoreProc
    except ImportError:
        logger.warning("Could not import EngineCoreProc, skipping patch")
        return

    original_handle_shutdown = getattr(EngineCoreProc, "_handle_shutdown", None)
    original_has_work = getattr(EngineCoreProc, "has_work", None)
    original_handle_client_request = getattr(EngineCoreProc, "_handle_client_request", None)

    original_has_global_unfinished_reqs = getattr(DPEngineCoreProc, "_has_global_unfinished_reqs", None)
    original_execute_dummy_batch = getattr(EngineCoreProc, "execute_dummy_batch", None)
    original_step = getattr(EngineCoreProc, "step", None)

    original_step_with_batch_queue_dp = getattr(DPEngineCoreProc, "step_with_batch_queue", None)
    original_step_with_batch_queue = getattr(EngineCoreProc, "step_with_batch_queue", None)
    
    if original_handle_shutdown is None:
        logger.warning("EngineCore._handle_shutdown not found")
        return

    # ------------------------------------------------------------------
    # 1. patched_has_work：空闲/暂停时返回 False，避免调度
    # ------------------------------------------------------------------
    def patched_has_work(self):
        """
        patched_has_work：空闲/暂停时返回 False，避免调度
        """
        # TODO: [lqf] 这里的生效场景是什么?
        paused = getattr(self, "_paused_for_restart", False)

        if paused:
            # Clear batch_queue when paused to prevent step_with_batch_queue
            # from trying to process stale entries after worker restart
            if hasattr(self, 'batch_queue') and self.batch_queue:
                self.batch_queue.clear()
                logger.debug("has_work: cleared batch_queue due to paused=True")
            logger.debug("has_work: paused=True, returning False")
            return False

        return original_has_work(self)

    # ------------------------------------------------------------------
    # 3. patched_has_global_unfinished_reqs 【关键】
    #    空闲时强制 local_unfinished=False，但正常参与 all_reduce
    # ------------------------------------------------------------------
    def patched_has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        executor = getattr(self, "model_executor", None)
        world_size = getattr(executor, "world_size", None) if executor else None
        paused = getattr(self, "_paused_for_restart", False)

        # scale-to-zero executor：存活 executor 已经用新的 N-rank dp_group
        # 取代旧 group，本 executor 没有通信对端。跳过 all_reduce，避免
        # 每次 DP sync 都等 10s 超时并 abort 一个无人参与的 collective。
        if (
            world_size == 0
            and getattr(self, "_its_dp_sync_excluded", False)
        ):
            logger.debug(
                "_has_global_unfinished_reqs: scale-to-zero executor "
                "excluded from DP sync, returning False"
            )
            return False

        # 空闲或暂停时：强制 local_unfinished = False
        if executor and (world_size == 0 or paused):
            local_unfinished = False
            logger.debug(f"_has_global_unfinished_reqs: forcing local_unfinished=False in idle/paused mode")

        # v0.23 two-phase DP pause protocol.  Use sync_dp_state's exact
        # two-element SUM collective (with a bounded wait) so this call can
        # pair with the full-restart barrier and with peers that are already
        # inside the same collective; bypassing pending_pause here would
        # break EngineCore.pause()/resume_scheduler() permanently.
        self.step_counter += 1
        logger.debug(f"[patched_has_global_unfinished_reqs][4.1] step_counter={self.step_counter}")
        mode = getattr(self, "zero_interrupt_mode", None)

        if mode == "degrade":
            # NOTE: 缩容场景下，每次前向进行一次同步（效率极低），用于规避idle-executor阻塞恢复命令执行
            ar_every_n_step = 1
        else:
            ar_every_n_step = 32

        if self.step_counter % ar_every_n_step != 0:
            return True

        pending_pause = bool(getattr(self, "pending_pause", False))
        result = sync_dp_state_fast_timeout(
            self.dp_group, local_unfinished, pending_pause
        )
        if result is None:
            # Timed out while a peer may be executing the deployment
            # strategy.  Keep the loop alive so this executor can reach
            # _handle_shutdown and join the full-restart barrier on the next
            # iteration.
            return True
        has_unfinished_global, pause_consensus = result
        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug(
                "[patched_has_global_unfinished_reqs] DP pause consensus reached"
            )

        logger.debug(f"[patched_has_global_unfinished_reqs][4.2] all-reduce done")
        return has_unfinished_global

    # ------------------------------------------------------------------
    # 2. patched_handle_client_request：world_size==0 时丢弃 ADD 请求
    #    避免请求堆积在空转 executor 的 scheduler 中
    # ------------------------------------------------------------------
    def patched_handle_client_request(self, request_type: EngineCoreRequestType, request: Any):
        """当 world_size==0 时丢弃 ADD 请求，避免请求堆积在 scheduler。

        空转 executor（dp=0/tp=0）在收到新部署策略后 world_size 变为 0，
        但仍然会从 input_queue 接收 ADD 请求。如果不丢弃，这些请求会
        堆积在 scheduler 中，造成内存浪费且无法处理。
        """
        executor = getattr(self, "model_executor", None)
        world_size = getattr(executor, "world_size", None) if executor else None

        if world_size == 0 and request_type == EngineCoreRequestType.ADD:
            logger.debug("Dropping ADD request in idle executor (world_size=0)")
            return

        return original_handle_client_request(self, request_type, request)

    # ------------------------------------------------------------------
    # 4. patched_execute_dummy_batch：world_size==0 时跳过执行
    # ------------------------------------------------------------------
    def patched_execute_dummy_batch(self):
        """当 world_size==0 或 paused 时跳过 dummy batch 执行，避免调用 collective_rpc 报错"""
        executor = getattr(self, "model_executor", None)
        world_size = getattr(executor, "world_size", None) if executor else None
        paused = getattr(self, "_paused_for_restart", False)

        if world_size == 0 or paused:
            logger.debug(f"execute_dummy_batch: world_size={world_size}, paused={paused}, skipping")
            return

        return original_execute_dummy_batch(self)

    # ------------------------------------------------------------------
    # 6. patched_handle_shutdown（不含进程退出逻辑）
    # ------------------------------------------------------------------
    def patched_handle_shutdown(self, *args, **kwargs):
        global _callback_registered
        logger.debug("========================================[patched_handle_shutdown] start====================================================")
        executor = getattr(self, "model_executor", None)
        logger.debug(f"+++[mzm]+++++executor={executor}+++[mzm]+++++")
        if not _callback_registered and executor is not None:
            _register_executor_callback(self, executor)
            _callback_registered = True

        # 初始化暂停状态标志
        if not hasattr(self, "_paused_for_restart"):
            self._paused_for_restart = False

        try:
            if executor is not None and hasattr(executor, "wait_new_deployment"):
                recv_set = getattr(executor, "recv_new_deployment", None)
                wait_set = executor.wait_new_deployment.is_set() if hasattr(executor, "wait_new_deployment") else False
                is_moe_model = getattr(self, "step_counter", False)
                is_dp_engine_core = isinstance(self, DPEngineCoreProc)

                if recv_set and recv_set.is_set():
                    logger.info("RecvNewDeployment received, executing strategy")
                    recv_set.clear()
                    if hasattr(executor, 'wait_new_deployment'):
                        executor.wait_new_deployment.clear()
                    # 阻塞新请求进入，立即中止正在执行的请求
                    self.process_input_queue_block = True
                    # 立即中止所有正在执行的请求，并把 abort 结果通知客户端
                    aborted_reqs = self.scheduler.finish_requests(
                        None, RequestStatus.FINISHED_ABORTED
                    )
                    self._send_abort_outputs(aborted_reqs)
                    # 等待请求中止完成（短暂等待即可）
                    for _ in range(50):  # 最多等待5秒
                        if not self.scheduler.has_unfinished_requests():
                            break
                        time.sleep(0.1)
                    if self.scheduler.has_unfinished_requests():
                        logger.warning(
                            "Requests still unfinished after 5s abort wait; "
                            "continuing with deployment strategy anyway."
                        )

                    # 设置暂停并执行策略
                    self._paused_for_restart = True # TODO: [lqf] 实际上没有并发的流程，这个flag好像并没有作用
                    logger.debug("++++[mzm]++++RecvNewDeployment received, executing strategy++++[mzm]++++")
                    _execute_deployment_strategy(self, executor)
                    # 仅在 world_size > 0 时恢复 worker monitor，idle mode 不需要
                    if getattr(executor, "world_size", 0) > 0:
                        _resume_worker_monitor(executor)

                    # 重置step_counter & 向coordinator更新wave
                    # 目前扩/缩容相当于服务重启，所以将请求step_counter重置是合理的
                    # 目的：防止各dp调用_has_global_unfinished_reqs时step_counter不一致导致卡死

                    # 如果self.step_counter == 0, 可以认为已经reset并向coordinator上传信息，不再重复上传
                    # 代码逻辑参考run_busy_loop中的reset逻辑
                    # wave: 相当于一个batch的请求，直到某个瞬间所有请求都计算完毕时，一个wave结束
                    if (is_dp_engine_core and self.step_counter != 0) and \
                        (self.dp_rank == 0 or not self.has_coordinator):
                        # Notify client that we are pausing the loop.
                        logger.info(
                            "[patched_handle_shutdown] drop current wave & inform coordinator wave done(when degrade/recover)"
                        )
                        # In the coordinator case, dp rank 0 sends updates to the
                        # coordinator. Otherwise (offline spmd case), each rank
                        # sends the update to its colocated front-end process.
                        client_index = -1 if self.has_coordinator else 0
                        self.output_queue.put_nowait(
                            (
                                client_index,
                                EngineCoreOutputs(wave_complete=self.current_wave),
                            )
                        )
                    if is_dp_engine_core:
                        self.current_wave += 1
                        self.step_counter = 0
                    self.engines_running = False # force the engine-core to wait for new req after degrade/recover
                    # TODO: [lqf] 这里要再次清理请求? 防止缩容过程中用户下发了新的请求?
                    aborted_reqs = self.scheduler.finish_requests(
                        None, RequestStatus.FINISHED_ABORTED
                    )
                    self._send_abort_outputs(aborted_reqs)
                    for _ in range(50):
                        if not self.scheduler.has_unfinished_requests():
                            break
                        time.sleep(0.1)
                    # 不要 clear finished_req_ids：scheduler 用该集合跟踪
                    # 已完成请求，清空会破坏后续 scheduler output 跟踪。
                    if self.batch_queue is not None:
                        self.batch_queue.clear()
                elif wait_set:
                    logger.info("WaitNewDeployment received, waiting for RecvNewDeployment")
                    executor.wait_new_deployment.clear()
                    self._paused_for_restart = True
                    self.process_input_queue_block = True

                    timeout = getattr(executor, "_deployment_timeout", VLLM_ITS_STRATEGY_TIMEOUT)
                    received = recv_set.wait(timeout=timeout) if recv_set else False

                    if received:
                        logger.info("RecvNewDeployment received, executing strategy")
                        recv_set.clear()
                        # 阻塞新请求进入，立即中止正在执行的请求
                        aborted_reqs = self.scheduler.finish_requests(
                            None, RequestStatus.FINISHED_ABORTED
                        )
                        self._send_abort_outputs(aborted_reqs)
                        for _ in range(50):
                            if not self.scheduler.has_unfinished_requests():
                                break
                            time.sleep(0.1)
                        _execute_deployment_strategy(self, executor)
                    else:
                        logger.warning("Deployment timeout, restarting workers")
                        if not _restart_workers_with_strategy(self, executor):
                            logger.error("Restart failed, entering idle mode")
                            if executor and hasattr(executor, "world_size"):
                                executor.world_size = 0
                    # 仅在 world_size > 0 时恢复 worker monitor
                    if getattr(executor, "world_size", 0) > 0:
                        _resume_worker_monitor(executor)
                    # 重置step_counter
                    # 目前扩/缩容相当于服务重启，所以将请求step_counter重置是合理的
                    # 目的：防止各dp调用_has_global_unfinished_reqs时step_counter不一致导致卡死
                    if is_dp_engine_core:
                        self.current_wave = 0
                        self.step_counter = 0

            # 如果 shutdown 未请求，保持运行等待请求
            if self.shutdown_state == EngineShutdownState.RUNNING:
                return True

            # 调用原始方法（通常返回 True）
            if original_handle_shutdown:
                return original_handle_shutdown(self, *args, **kwargs)
            return True

        except Exception as e:
            logger.error(f"Handle shutdown failed: {e}", exc_info=True)
            raise
        finally:
            # busy_loop每次都会执行到此处
            # 确保状态一定被重置
            self._paused_for_restart = False

             # vllm默认为True，会被trigger_busy_loop先设置为False
            self.process_input_queue_block = True



    def patched_step(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """v0.23 ``EngineCoreProc.step`` with ITS fault handling.

        With default PP=1 / async-scheduling-off, ``max_concurrent_batches``
        is 1 and ``batch_queue`` is None, so ``step_fn`` points at this
        method -- the patched ``step_with_batch_queue`` above is never used.
        Keep the same scheduling behaviour as v0.23 but do not feed a None
        model output into ``scheduler.update_from_output`` after a worker
        failure; the health monitor will trigger the deployment flow.
        """
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()

        executor = getattr(self, "model_executor", None)
        world_size = getattr(executor, "world_size", None) if executor else None
        paused = getattr(self, "_paused_for_restart", False)
        if world_size == 0 or paused:
            return {}, False

        exec_future = self.model_executor.execute_model(
            scheduler_output, non_block=True
        )
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = exec_future.result()
            if model_output is None:
                try:
                    model_output = self.model_executor.sample_tokens(
                        grammar_output
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[patched_step] sample_tokens failed after execute "
                        "model returned None: %s",
                        exc,
                    )
                    model_output = None

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        if model_output is None:
            logger.warning(
                "[patched_step] model output is None (worker failure); "
                "skipping scheduler update for this step."
            )
            return {}, False

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        return (
            engine_core_outputs,
            scheduler_output.total_num_scheduled_tokens > 0,
        )

    def patched_step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """
        对故障处理逻辑进行了需改，功能代码从vllm原生step_with_batch_queue代码拷贝而来
        """
        logger.debug("[patched_step_with_batch_queue]: start")
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            # has_requests() 为 True 只是说明 scheduler 还有“未完成请求”或
            # “尚未在下一次 schedule() 中返回的 finished 请求”。PD 场景里
            # WAITING_FOR_REMOTE_KVS 的请求也会让这里持续为 True，属于正常
            # 的 1ms-yield 路径。为避免每个 step 都刷 INFO，只在状态变化或
            # 每 30s 打印一次运行/等待计数与请求状态，便于判断是否卡住。
            running, waiting = self.scheduler.get_request_counts()
            now = time.monotonic()
            last_log_ts = getattr(self, "_last_hetero_request_log_ts", None)
            last_counts = getattr(self, "_last_hetero_request_counts", None)
            if (
                last_log_ts is None
                or now - last_log_ts >= 30
                or last_counts != (running, waiting)
            ):
                request_states = []
                for req_id, req in list(
                    getattr(self.scheduler, "requests", {}).items()
                )[:10]:
                    status = getattr(req, "status", None)
                    request_states.append(
                        f"{req_id}:{getattr(status, 'name', status)}"
                    )
                self._last_hetero_request_log_ts = now
                self._last_hetero_request_counts = (running, waiting)
                logger.info(
                    "step_with_batch_queue: has_requests=True "
                    "running=%d waiting=%d batch_queue=%d requests=%s",
                    running,
                    waiting,
                    len(batch_queue),
                    request_states,
                )
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                # patched: 获取零中断相关变量状态
                executor = getattr(self, "model_executor", None)
                world_size = getattr(executor, "world_size", None) if executor else None
                paused = getattr(self, "_paused_for_restart", False)

                if world_size == 0 or paused:
                    # patched: 针对DP域下所有worker都关停的情况, 直接返回，不执行任何动作
                    return None, True
                else:                    
                    exec_future = self.model_executor.execute_model(
                        scheduler_output, non_block=True
                    )

            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                # future入队
                # Step 1: 调度 Batch A → 提交 GPU 执行 → 放入 batch_queue (不等结果)  
                # Step 2: 调度 Batch B → 提交 GPU 执行 → 放入 batch_queue (不等结果)  
                #          ↓ 队列满了，才阻塞等待 Batch A 的结果  
                # Step 3: pop Batch A → future.result() → 处理输出  
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False
        
        # batch_queue 只在 max_concurrent_batches > 1 时启用，用于将调度与 GPU 执行重叠

        # Block until the next result is available.
        # future	Future[ModelRunnerOutput]	sample_tokens() 的非阻塞 Future，持有采样结果
        # scheduler_output	SchedulerOutput	本批次的调度结果（哪些请求、分配了多少 token 等）
        # exec_future	Future[Any]	execute_model() 的非阻塞 Future，持有模型执行结果
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            # 故障发生时返回None，目前对None的返回统一忽略
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                # exec_model_fut.result()
                # raise RuntimeError("unexpected error")
                # patched: 不抛出异常，否则主进程会退出
                pass


        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()

        # patched: 故障场景，提前退出
        if model_output is None:
            logger.info(f"[step_with_batch_queue] encounter err, and return from step_with_batch_queue directly.")
            return None, model_executed

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed


    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    def _send_idle_notification(engine_core):
        """发送空转通知给客户端。

        当 world_size==0 时，向 output_queue 发送带有特殊负载分数的
        EngineCoreOutputs，让客户端知道这个 engine 是空转的。
        coordinator不应向空转的core派发任务（coordinator只在dp>1的场景下会出现）
        
        空转定义: 某个DP域中的所有worker都被关停，此时engine_core处于空转状态。
        """
        global _idle_notification_sent

        if _idle_notification_sent:
            return

        # 检查 output_queue 是否可用
        if not hasattr(engine_core, 'output_queue'):
            return

        # 创建带有特殊负载分数的输出，告知客户端此 engine 空转
        # 使用 999999 作为空转标记
        stats = SchedulerStats(
            num_waiting_reqs=999999,
            num_running_reqs=999999,
            step_counter=0,
            current_wave=0
        )
        outputs = EngineCoreOutputs(scheduler_stats=stats)
        outputs.engine_index = engine_core.engine_index

        try:
            vllm_config = engine_core.vllm_config
            # 1. 推理结果（token 输出）不经过 Coordinator，直接 EngineCore → Client，由故障executor负责通知给所有client
            # 2. Coordinator 只传递 stats 和 wave 控制消息，是控制平面，不是数据平面
            # Both EngineCoreProc (non-DP) and DPEngineCoreProc (DP) inherit from the same base class.
            # so engine_core.addresses.outputs is available for both types
            num_output_sockets = len(engine_core.addresses.outputs)
            for client_idx in range(num_output_sockets):
                engine_core.output_queue.put_nowait((client_idx, outputs)) # 发给所有client
            logger.info(f"Engine {engine_core.engine_index} sent recovered notification to all {num_output_sockets} clients")
            if isinstance(engine_core, DPEngineCoreProc):
                engine_core.output_queue.put_nowait((-1, outputs))  # 发给coordinator
                logger.info(f"Engine {engine_core.engine_index} sent idle notification to coordinator")
            _idle_notification_sent = True
        except Exception as e:
            logger.warning(f"Failed to send idle notification: {e}")

    def _send_recovered_notification(engine_core):
        """发送恢复通知给客户端。

        当 world_size 从 0 变为非 0 时调用，发送带有正常负载分数的通知，
        让客户端移除 idle 标记。
        """
        global _idle_notification_sent

        # 无条件重置标志，以便后续能再次发送 idle 通知
        _idle_notification_sent = False

        if not hasattr(engine_core, 'output_queue'):
            return

        # 创建带有正常负载分数的输出，告知客户端此 engine 已恢复
        # 使用 (0, 0) 作为恢复标记
        stats = SchedulerStats(
            num_waiting_reqs=0,
            num_running_reqs=0,
            step_counter=0,
            current_wave=0
        )
        outputs = EngineCoreOutputs(scheduler_stats=stats)
        outputs.engine_index = engine_core.engine_index

        try:
            # 与 idle 通知保持一致，使用本 EngineCore 实际连接的输出
            # socket 数量，不要用 _api_process_count：两者在多 client /
            # external LB / hybrid LB 下并不相等，漏发会导致客户端无法
            # 清除 idle 标记。
            num_output_sockets = len(engine_core.addresses.outputs)
            for client_idx in range(num_output_sockets):
                engine_core.output_queue.put_nowait((client_idx, outputs)) # 发给所有client
            logger.info(f"Engine {engine_core.engine_index} sent recovered notification to all {num_output_sockets} clients")

            # 只有DP>1时才有coordinator，此时使用DPEngineCoreProc，特殊负载分数才应生效
            if isinstance(engine_core, DPEngineCoreProc):
                engine_core.output_queue.put_nowait((-1, outputs))
                logger.info(f"Engine {engine_core.engine_index} sent recovered notification to coordinator")
        except Exception as e:
            logger.warning(f"Failed to send recovered notification: {e}")


    def _execute_deployment_strategy(engine_core, executor):
        if not (hasattr(executor, "current_strategy") and executor.current_strategy):
            return False
        logger.info("Executing deployment strategy")
        success = False
        if hasattr(executor, "handle_new_deployment"):
            # 1. 通过修改executor的vllm_config, 可以修改worker connector的engine-id
            vllm_config = executor.vllm_config

            # 幂等 RECOVER / 重复 DEGRADE 会在 executor 内短路，不会真正
            # 重启 worker。此时绝不能轮换 engine_id 或刷新 connector 元数据：
            # scheduler 侧会被改成新 id，而 worker 仍持有旧 id。
            try:
                will_restart = bool(
                    executor.will_restart_workers_for_strategy(
                        executor.current_strategy
                    )
                )
            except Exception as exc:
                logger.warning(
                    "[PD] failed to check whether workers will restart; "
                    "falling back to full-restart flag: %s",
                    exc,
                )
                try:
                    will_restart = bool(
                        executor._strategy_requires_full_restart(
                            executor.current_strategy
                        )
                    )
                except Exception:
                    will_restart = False

            # 仅在 PD 分离场景且确实会重启 worker 时执行 engine_id 更新逻辑
            if will_restart and is_pd_separated(vllm_config):
                from uuid import uuid4

                ori_engine_id = vllm_config.kv_transfer_config.engine_id
                # 异构 TP 全量重启后 P 侧 15 个 worker 会重建
                # TransferEngine/KV cache，te_rpc_port 和 KV 基地址全部变化。
                # decode 侧 KVCacheRecvingThread 以 (engine_id, handshake_port)
                # 为 key 缓存远端元数据且不会主动失效，若 P 保持原 engine_id，
                # D 会继续使用旧地址传输，PD 链路必然失败。因此 producer
                # 全量重启时强制轮换 engine_id，让 D 对新 key 重新拉取元数据。
                full_restart = bool(
                    executor._strategy_requires_full_restart(
                        executor.current_strategy
                    )
                )
                kv_role = getattr(
                    vllm_config.kv_transfer_config, "kv_role", None
                )
                update_info = getattr(
                    executor.current_strategy, "update_engine_info", None
                )
                if update_info is None and full_restart and kv_role == "kv_producer":
                    vllm_config.kv_transfer_config.engine_id = (
                        f"{ori_engine_id}-{uuid4().hex}"
                    )
                    logger.info(
                        "[PD] heterogeneous producer restart rotates "
                        "engine_id %s -> %s",
                        ori_engine_id,
                        vllm_config.kv_transfer_config.engine_id,
                    )
                elif update_info is None:
                    # 兼容既有 prefix-uuid-suffix 三段 engine_id 的更新逻辑。
                    ori_engine_id_components = (
                        ori_engine_id.replace("_", "-").split("-")
                    )
                    if len(ori_engine_id_components) == 3:
                        prefix, ori_uuid, suffix = ori_engine_id_components
                        vllm_config.kv_transfer_config.engine_id = (
                            f"{prefix}-{uuid4().hex}-{suffix}"
                        )
                        logger.info(
                            "[PD] reset executor's "
                            "vllm_config.kv_transfer_config.engine_id=%s",
                            vllm_config.kv_transfer_config.engine_id,
                        )

                # 2. 直接修改scheduler connector的engine-id
                sched_kv_connector = engine_core.scheduler.get_kv_connector()
                assert sched_kv_connector is not None, "[PD][_execute_deployment_strategy] No KV connector found in scheduler"
                assert hasattr(sched_kv_connector, "engine_id"),"[PD] No engine_id attribute in scheduler KV connector"
                sched_kv_connector.engine_id = vllm_config.kv_transfer_config.engine_id
                sched_kv_connector.connector_scheduler.engine_id = vllm_config.kv_transfer_config.engine_id
                logger.info(f"[PD] reset executor's sched_kv_connector.connector_scheduler.engine_id={vllm_config.kv_transfer_config.engine_id}")


                for node_meta in sched_kv_connector.connector_scheduler.multi_nodes_meta_mapping.values():
                    node_meta["engine_id"] = vllm_config.kv_transfer_config.engine_id

                # 3. 更新 executor.current_strategy.update_engine_info
                executor.current_strategy.update_engine_info = UpdateEngineInfo(
                    orig_engine_id=ori_engine_id,
                    new_engine_id=vllm_config.kv_transfer_config.engine_id
                )

                # 4. 更新worker的engine_id
                if sched_kv_connector.connector_worker:
                    connector_worker = sched_kv_connector.connector_worker
                    connector_worker.engine_id = (
                        vllm_config.kv_transfer_config.engine_id
                    )
                    worker_handshake_metadata = getattr(
                        connector_worker,
                        "xfer_handshake_metadata",
                        None,
                    )
                    if worker_handshake_metadata is not None:
                        worker_handshake_metadata.engine_id = (
                            vllm_config.kv_transfer_config.engine_id
                        )

            success = executor.handle_new_deployment()

            # PD 分离且 worker 重启后，scheduler connector 里保存的仍是旧
            # worker 的 handshake 元数据。异构重启后端口/engine_id 都会变，
            # 必须从新 worker 重新收集并刷新，否则 decode 端会按旧端口
            # 拉 KV，PD 链路无法恢复。幂等跳过的策略没有新 worker，不要刷新。
            if success and will_restart and is_pd_separated(vllm_config):
                if hasattr(executor, "update_kv_connector_metadata"):
                    executor.update_kv_connector_metadata(engine_core)
            
            # 仅 PD 分离且 worker 确实重启后刷新 scheduler connector 拓扑。
            if success and will_restart and is_pd_separated(vllm_config):
                # 5. 更新 scheduler connector 的 side_channel_port 及拓扑字段。
                # 异构 TP（DP4TP(3,4,4,4)）下 DP 端口偏移是累计的
                # 0/3/7/11，不能再用 dp_rank * tp_size 的均匀公式。
                # tp_size/max_device_id 必须在每次部署都刷新：场景 3 P
                # RECOVER 后 parallel_config.is_heterogeneous_tp 已变回
                # False，只刷新异构分支会继续向 D 端通告旧的 tp=3 /
                # 15-device 布局，导致 D 按错误 producer 布局选 rank，
                # RECOVER 后输出与基线不一致。
                parallel_config = vllm_config.parallel_config
                (
                    side_channel_port,
                    scheduler_tp_size,
                    scheduler_max_device_id,
                ) = get_pd_scheduler_connector_topology(
                    parallel_config,
                    vllm_config.kv_transfer_config.kv_port,
                )
                sched_kv_connector = engine_core.scheduler.get_kv_connector()
                sched_kv_connector.connector_scheduler.side_channel_port = side_channel_port
                sched_kv_connector.connector_scheduler.tp_size = scheduler_tp_size
                sched_kv_connector.connector_scheduler.max_device_id = scheduler_max_device_id
                logger.info(f"[PD] reset executor's sched_kv_connector.connector_scheduler.side_channel_port={side_channel_port}")


        if success:
            # TODO: [lqf] 优化一下写法
            # MoE场景下，engine_core根据是否为degrade，修改has_unfinished_request的dp-group allreduce执行频率
            if executor.current_strategy.deploy_type == DeployType.DEGRADE:
                setattr(engine_core, "zero_interrupt_mode", "degrade")
            elif executor.current_strategy.deploy_type == DeployType.RECOVER:
                setattr(engine_core, "zero_interrupt_mode", "normal")
            elif executor.current_strategy.deploy_type == DeployType.STOP:
                setattr(engine_core, "zero_interrupt_mode", "stop")

        logger.debug(f"++++[mzm]++++Deployment strategy execution result: {success}++++[mzm]++++")
        # 任何策略（包括 PD_REBUILD 的 scale-to-zero executor）只要
        # world_size 变为 0，都必须向 coordinator/client 发送空转通知。
        # 否则 DPLB 客户端不知道该 engine 已 idle，继续向其路由 ADD，
        # 而 idle engine 会丢弃这些请求导致客户端永久等待。
        if success and (
            getattr(executor, "world_size", 1) == 0
            or executor.current_strategy.deploy_type == DeployType.STOP
        ):
            logger.debug(
                "++++[mzm]++++Strategy deploy type: %s, world_size: %s",
                executor.current_strategy.deploy_type,
                getattr(executor, "world_size", "unknown"),
            )
            logger.info("Strategy resulted in world_size=0, entering idle mode")
            _send_idle_notification(engine_core)
        # 如果 world_size 变为非 0，发送恢复通知给客户端
        if success and getattr(executor, "world_size", 0) > 0:
            _send_recovered_notification(engine_core)
        return success

    def _restart_workers_with_strategy(engine_core, executor):
        """使用当前策略重启 workers。

        设计：使用 VllmConfig.additional_config["zero_interrupt_config"] 中的策略重启 workers。
        对于 PD_REBUILD 策略，重启后还要更新 KV connector 元数据。

        重试逻辑：
        - 如果重启失败，最多重试 3 次
        - 连续 3 次失败后，shutdown executor 并返回 False

        Args:
            engine_core: EngineCore 实例
            executor: Executor 实例

        Returns:
            True 成功，False 失败
        """
        max_retries = VLLM_ITS_MAX_RETRY_COUNT
        for attempt in range(1, max_retries + 1):
            logger.info(f"Restart attempt {attempt}/{max_retries}")
            success = False
            if hasattr(executor, "restart_workers_with_strategy"):
                try:
                    success = executor.restart_workers_with_strategy()
                except Exception as e:
                    logger.error(f"Restart failed: {e}")
            if success:
                return True
            # 重启失败，检查是否有新策略下发
            # 防止在重启过程中决策中心下发了新策略，此时执行策略可能恢复服务
            if executor.current_strategy:
                # 清理事件标志，防止重复触发
                executor.recv_new_deployment.clear()
                if hasattr(executor, 'wait_new_deployment'):
                    executor.wait_new_deployment.clear()
                if _execute_deployment_strategy(engine_core, executor):
                    return True
            time.sleep(1)
        # 所有重试都用完了，shutdown executor
        logger.error(f"All {max_retries} worker restart attempts failed, shutting down executor")
        if hasattr(executor, "shutdown"):
            try:
                executor.shutdown()
            except Exception as e:
                logger.error(f"Error during executor shutdown: {e}")

        # 设置强制退出标志以退出 busy loop
        engine_core._its_force_exit = True
        return False

    def _resume_worker_monitor(executor):
        # idle mode (world_size=0) 时不启动健康监控，因为没有 workers 要监控
        if getattr(executor, "world_size", 0) == 0:
            logger.debug("Skipping worker monitor resume: world_size=0 (idle mode)")
            return
        strategy = getattr(executor, "current_strategy", None)
        if strategy and strategy.deploy_type != DeployType.STOP:
            if hasattr(executor, "_health_monitor") and executor._health_monitor:
                if not executor._health_monitor.is_running():
                    logger.info("Resuming Worker Monitor thread")
                    executor._health_monitor.start()

    def _register_executor_callback(engine_core, executor):
        """注册回调以从 executor 触发 busy_loop。

        这允许 executor 的 health_monitor 在检测到新部署信号时唤醒 busy_loop。

        Args:
            engine_core: EngineCoreProc 实例
            executor: Executor 实例
        """

        # 设置 process_input_queue_block = False 并发送 WAKEUP 信号
        # 这将导致 busy_loop：
        # 1. 立即退出 _process_input_queue（通过 WAKEUP）
        # 2. 在下一次迭代时跳过推理（通过 process_input_queue_block=False）
        def trigger_busy_loop():
            """触发 busy_loop 立即响应部署信号。

            通过设置 process_input_queue_block=False 和发送 WAKEUP 信号，
            唤醒阻塞中的 busy_loop，使其立即检查并处理部署信号。
            """
            # 1. 必须wake(发送请求至engine_core.input_queue)
            # 原因: 让busy_loop执行到handle_shutdown函数
            # 如果不wake，engine-core可能会卡在process_input_queue等任务
            # 2. wake以后，不能干任何事情，应该直接跳转至handle_shutdown函数
            # 否则，干任何事情都可能让系统卡在某个位置(e.g. execute-dummy-batch/execute-model)
            try:
                logger.info("Triggering busy_loop: setting block=False, sending WAKEUP")

                # 步骤 1：设置标志跳过推理
                engine_core.process_input_queue_block = False
                logger.info(f"process_input_queue_block set to: {engine_core.process_input_queue_block}")

                # 步骤 2：发送 WAKEUP 以取消阻塞 input_queue.get()
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
                logger.info("WAKEUP sent to input_queue")

                logger.info("Triggered busy_loop: WAKEUP sent, process_input_queue_block=False")
            except Exception as e:
                logger.error(f"Failed to trigger busy_loop: {e}")

        # 在 executor 上注册回调并存储 engine_core 引用
        executor._trigger_busy_loop_callback = trigger_busy_loop
        executor._engine_core_ref = engine_core
        logger.info("Registered busy_loop trigger callback on executor")

    # ------------------------------------------------------------------
    # 应用所有 patch
    # ------------------------------------------------------------------
    if original_has_global_unfinished_reqs:
        DPEngineCoreProc._has_global_unfinished_reqs = patched_has_global_unfinished_reqs
        logger.info("PATCH: DPEngineCoreProc._has_global_unfinished_reqs (idle mode forces local_unfinished=False)")

    if original_handle_shutdown:
        EngineCoreProc._handle_shutdown = patched_handle_shutdown
        logger.info("PATCH: EngineCoreProc._handle_shutdown (deployment strategy, no early exit)")

    if original_has_work:
        EngineCoreProc.has_work = patched_has_work
        logger.info("PATCH: EngineCoreProc.has_work (pause when idle)")

    if original_handle_client_request:
        EngineCoreProc._handle_client_request = patched_handle_client_request
        logger.info("PATCH: EngineCoreProc._handle_client_request (drop ADD when world_size=0)")

    if original_execute_dummy_batch:
        EngineCoreProc.execute_dummy_batch = patched_execute_dummy_batch
        logger.info("PATCH: EngineCoreProc.execute_dummy_batch (skip when world_size=0)")

    if original_step:
        EngineCoreProc.step = patched_step
        logger.info("PATCH: EngineCoreProc.step (fault-tolerant None model output)")

    if original_step_with_batch_queue_dp:
        DPEngineCoreProc.step_with_batch_queue = patched_step_with_batch_queue
        logger.info("PATCH: DPEngineCoreProc.step_with_batch_queue (handle exception during degrade/recover procedure)")

    if original_step_with_batch_queue:
        EngineCoreProc.step_with_batch_queue = patched_step_with_batch_queue
        logger.info("PATCH: EngineCoreProc.step_with_batch_queue (handle exception during degrade/recover procedure)")


    logger.info("All patches applied successfully")
