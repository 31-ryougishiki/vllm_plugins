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
from vllm.config import ParallelConfig
from datetime import timedelta


from vllm_custom_plugins.plugins.zero_interrupt.common.constants import VLLM_ITS_STRATEGY_TIMEOUT, VLLM_ITS_MAX_RETRY_COUNT
from vllm_custom_plugins.plugins.zero_interrupt.common.types import DeployType, UpdateEngineInfo

# 模块级变量，用于跟踪回调是否已注册
_callback_registered = False
# 标记是否已发送空转通知，避免重复发送
_idle_notification_sent = False

def is_pd_separated(vllm_config) -> bool:
    return (vllm_config.kv_transfer_config is not None and 
            (vllm_config.kv_transfer_config.kv_role == "kv_producer" or vllm_config.kv_transfer_config.kv_role == "kv_consumer"))

def has_unfinished_dp_fast_timeout(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
    """
        代码逻辑拷贝自ParallelConfig.has_unfinished_dp, 但设置较短的timeout。目的是让core_busy_loop不要阻塞。
        
        目前timeout设置时间为10s。原因代码在推理时，上层同步时间不应发生10s的等待时间。
        所以用10s作为分界线，用于处理部分executor接收到策略并执行，而部分executor需同步后才能进入下一个循环执行扩/缩容策略。
        
        问题根因：dp_group包含了idle-executor和active-executor的，而idle-executor在worker层面
        没法与active-executor管理的worker同步。所以会出现别的executor处于扩容，而idle-executor处于同步的死锁状态。
        
        允许timeout的方案：
        当idle-executor发生timeout后，能继续执行到idle-executor的core_busy_loop只能不断试错，
        core_busy_loop保证其它active-executor需要进行通信的时候它能及时进行通信
    """
    tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")

    work = torch.distributed.all_reduce(tensor, group=dp_group, async_op=True)
    try:
        work.wait(timeout=timedelta(seconds=10))  # 10秒超时(远小于默认值)
    except Exception as e:
        logger.info(f"[has_unfinished_dp]all_reduce 超时，执行异常处理, e={e}")

        # TODO: 应该统一返回什么? 
        # 1. 返回 True（engines_running = True）
        # 下一轮循环继续正常执行：handle_shutdown → _process_input_queue() → _process_engine_step()
        # 如果本 rank 有请求，执行真实 batch；如果没有但其他 rank 有，执行 dummy batch（保证 MoE expert 层的 TP/EP 集合通信对齐）
        # 前 31 步必然走这条路（step_counter % 32 != 0 直接返回 True，跳过 all-reduce）
        # 2. 返回 False（engines_running = False）
        # 执行以下收尾动作，然后引擎进入空闲等待：
        # 发送 wave 完成通知：dp_rank == 0（有 coordinator 时）或每个 rank（无 coordinator 时）向 output_queue 写入 EngineCoreOutputs(wave_complete=self.current_wave)，通知 coordinator/前端当前 wave 已结束
        # 递增 wave 计数：self.current_wave += 1，self.step_counter = 0
        # 下一轮进入空闲：_process_input_queue() 中 has_work() 返回 False（engines_running=False 且无本地请求），引擎阻塞在 input_queue.get(block=True) 等待新请求或 START_DP_WAVE 消息
        return False
    aggregated_has_unfinished = bool(tensor.item())
    return aggregated_has_unfinished

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

        # 空闲或暂停时：强制 local_unfinished = False
        if executor and (world_size == 0 or paused):
            local_unfinished = False
            logger.debug(f"_has_global_unfinished_reqs: forcing local_unfinished=False in idle/paused mode")

        # 原始has_global_unfinished_reqs代码逻辑
        self.step_counter += 1
        logger.debug(f"[patched_has_global_unfinished_reqs][4.1] step_counter={self.step_counter}")
        mode = getattr(self, "zero_interrupt_mode", None)

        if mode == "degrade":
            # NOTE: 缩容场景下，每次前向进行一次同步（效率极低），用于规避idle-executor阻塞恢复命令执行
            # 后续缩容过程中能重新建立gloo通信组时去掉这部分逻辑
            ar_every_n_step = 1
        else:
            ar_every_n_step = 32

        if self.step_counter % ar_every_n_step != 0:
            return True

        if mode == "degrade":
            res = ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        else:
            res = has_unfinished_dp_fast_timeout(self.dp_group, local_unfinished)

        logger.debug(f"[patched_has_global_unfinished_reqs][4.2] all-reduce done")
        return res
        # return original_has_global_unfinished_reqs(self, local_unfinished)

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
                    # 立即中止所有正在执行的请求
                    self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
                    # 等待请求中止完成（短暂等待即可）
                    for _ in range(50):  # 最多等待5秒
                        if not self.scheduler.has_unfinished_requests():
                            break
                        time.sleep(0.1)
                    assert not self.scheduler.has_unfinished_requests(), "still has_unfinished_requests after wating 5 sec"

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
                    self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
                    for _ in range(50):
                        if not self.scheduler.has_unfinished_requests():
                            break
                        time.sleep(0.1)
                    self.scheduler.finished_req_ids.clear() # hack
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
                        self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
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



    def patched_step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """
        对故障处理逻辑进行了需改，功能代码从vllm原生step_with_batch_queue代码拷贝而来
        """
        logger.info("[patched_step_with_batch_queue]: start")
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            logger.info("step_with_batch_queue: self.scheduler.has_requests() == True")
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
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

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
            vllm_config = engine_core.vllm_config
            api_server_count = vllm_config.parallel_config._api_process_count
            for client_idx in range(api_server_count):
                engine_core.output_queue.put_nowait((client_idx, outputs)) # 发给所有client
            logger.info(f"Engine {engine_core.engine_index} sent recovered notification to all {api_server_count} clients")

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

            # 仅在 PD 分离场景（kv_transfer_config 不为 None）下执行 engine_id 更新逻辑
            if is_pd_separated(vllm_config):
                from uuid import uuid4
                ori_engine_id = vllm_config.kv_transfer_config.engine_id
                ori_engine_id_components = ori_engine_id.replace('_', '-').split('-')
                if len(ori_engine_id_components) == 3:
                    prefix, ori_uuid, suffix = ori_engine_id_components
                    vllm_config.kv_transfer_config.engine_id = f"{prefix}-{uuid4().hex}-{suffix}"
                    logger.info(f"[PD] reset executor's vllm_config.kv_transfer_config.engine_id={vllm_config.kv_transfer_config.engine_id}")

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
                    worker_handshake_metadata = sched_kv_connector.connector_worker.xfer_handshake_metadata
                    worker_handshake_metadata.engine_id = vllm_config.kv_transfer_config.engine_id

            success = executor.handle_new_deployment()

            # PD 分离且 worker 重启后，scheduler connector 里保存的仍是旧
            # worker 的 handshake 元数据。异构重启后端口/engine_id 都会变，
            # 必须从新 worker 重新收集并刷新，否则 decode 端会按旧端口
            # 拉 KV，PD 链路无法恢复。
            if success and is_pd_separated(vllm_config):
                if hasattr(executor, "update_kv_connector_metadata"):
                    executor.update_kv_connector_metadata(engine_core)
            
            # 仅 PD 分离场景
            if is_pd_separated(vllm_config):
                # 5. 更新 scheduler connector 的 side_channel_port
                # 异构 TP（DP4TP(3,4,4,4)）下 DP 端口偏移是累计的
                # 0/3/7/11，不能再用 dp_rank * tp_size 的均匀公式。
                parallel_config = vllm_config.parallel_config
                if getattr(parallel_config, "is_heterogeneous_tp", False):
                    port_offset = parallel_config.get_rank_offset_for_dp(
                        parallel_config.data_parallel_rank
                    )
                else:
                    pcp_size = parallel_config.prefill_context_parallel_size
                    port_offset = (
                        parallel_config.data_parallel_rank
                        * parallel_config.tensor_parallel_size
                        * parallel_config.pipeline_parallel_size
                        * pcp_size
                    )
                side_channel_port = (
                    vllm_config.kv_transfer_config.kv_port + port_offset
                )

                sched_kv_connector.connector_scheduler.side_channel_port = side_channel_port
                if getattr(parallel_config, "is_heterogeneous_tp", False):
                    # request_finished_all_groups 用 scheduler 的 tp_size
                    # 向 decode 端通告 prefill TP 布局。异构重启后 DP0 只有
                    # 3 个 rank，仍用旧的 4 会选到不存在的 producer rank。
                    sched_kv_connector.connector_scheduler.tp_size = (
                        parallel_config.tensor_parallel_size
                    )
                    sched_kv_connector.connector_scheduler.max_device_id = (
                        parallel_config.world_size_across_dp
                    )
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
        # STOP/DEGRADE: 如果 world_size 变为 0，进入空闲模式，发送空转通知
        if success and executor.current_strategy.deploy_type in (DeployType.STOP, DeployType.DEGRADE):
            logger.debug(f"++++[mzm]++++Strategy deploy type: {executor.current_strategy.deploy_type}, world_size: {getattr(executor, 'world_size', 'unknown')}++++[mzm]+++")
            if getattr(executor, "world_size", 1) == 0:
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

    if original_step_with_batch_queue_dp:
        DPEngineCoreProc.step_with_batch_queue = patched_step_with_batch_queue
        logger.info("PATCH: DPEngineCoreProc.step_with_batch_queue (handle exception during degrade/recover procedure)")

    if original_step_with_batch_queue:
        EngineCoreProc.step_with_batch_queue = patched_step_with_batch_queue
        logger.info("PATCH: EngineCoreProc.step_with_batch_queue (handle exception during degrade/recover procedure)")


    logger.info("All patches applied successfully")
