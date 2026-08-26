#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""CoreClient patch for ITS plugin.

本模块提供对 DPLBAsyncMPClient 的 patch，实现客户端级别的空转 engine 过滤。

设计思路：
- 通过 process_engine_outputs 检测 engine 的负载分数
- 当 engine 的负载分数异常高（999999, 999999）时，标记为空转
- 在 get_core_engine_for_request 中跳过空转的 engine
"""

from vllm.logger import logger
import traceback

# 模块级变量，用于追踪空转的 engine
_idle_engines: set[int] = set()
_patch_applied = False


def patch_dplb_client() -> None:
    """Patch DPLBAsyncMPClient 以支持空转 engine 过滤。

    当某个 engine 变为空转状态（world_size=0）时，
    客户端应该跳过该 engine，不将请求发送过去。
    """
    global _patch_applied

    if _patch_applied:
        logger.debug("DPLB client patch already applied")
        return

    try:
        from vllm.v1.engine.core_client import DPLBAsyncMPClient
    except ImportError:
        logger.warning("Could not import DPLBAsyncMPClient, skipping client patch")
        return

    # 保存原始方法
    original_get_core_engine = DPLBAsyncMPClient.get_core_engine_for_request
    original_process_outputs = DPLBAsyncMPClient.process_engine_outputs

    # ------------------------------------------------------------------
    # Patched get_core_engine_for_request
    # 跳过被标记为空转的 engine
    # ------------------------------------------------------------------
    def patched_get_core_engine_for_request(self, request):
        """跳过空转的 engine，避免请求发送到坏卡。

        空转 engine 的特征：
        - world_size=0，不再处理请求
        - 负载分数被设置为极高值 (999999, 999999)
        """
        global _idle_engines
        logger.debug(f"+++[mzm]++++Current idle engines: {_idle_engines}+++[mzm]++++")
        # 如果没有空转 engine，直接调用原始方法
        if not _idle_engines:
            logger.info("No idle engines, using original selection")
            return original_get_core_engine(self, request)

        # 检查是否有可用的非空转 engine
        num_engines = len(self.core_engines)
        active_engines = [i for i in range(num_engines) if i not in _idle_engines]
        logger.info(f"Idle engines: {_idle_engines}, Active engines: {active_engines}")

        if not active_engines:
            # 所有 engine 都空转，fallback 到原始逻辑
            logger.info("All engines are idle, falling back to original selection")
            return original_get_core_engine(self, request)

        # 临时保存 lb_engines 并过滤
        saved_lb_engines = self.lb_engines
        saved_core_engines = self.core_engines

        # 只保留活跃 engine 的负载信息
        self.lb_engines = [saved_lb_engines[i] for i in active_engines]
        self.core_engines = [saved_core_engines[i] for i in active_engines]

        try:
            result = original_get_core_engine(self, request)
            # 将返回的 engine 转换回原始索引
            if result in self.core_engines:
                actual_index = active_engines[self.core_engines.index(result)]
                # 重新映射回原始 engine
                result = saved_core_engines[actual_index]
                logger.info(f"Request routed to engine {actual_index} (filtered idle: {_idle_engines})")
            return result
        finally:
            # 恢复原始列表
            self.lb_engines = saved_lb_engines
            self.core_engines = saved_core_engines

    # ------------------------------------------------------------------
    # Patched process_engine_outputs
    # 检测空转 engine：当负载分数异常高时标记为空转
    # ------------------------------------------------------------------
    async def patched_process_engine_outputs(self, outputs):
        """处理 engine 输出，检测空转状态。

        当 engine 的负载分数为 (999999, 999999) 时，说明该 engine
        已被标记为空转（通过部署策略），应该跳过。
        
        注: 这是client的函数，不是coordinator的函数
            TP1DP4MoE场景下，共4个DPLBAsyncMPClient实例，每个client_idx(0,1,2,3)对应一个实例
            多个DPLBAsyncMPClient只是为了并发，每个client 都连接到所有 4个（所有） EngineCore
        """
        # logger.info("Call stack for patched_process_engine_outputs:")
        # traceback.print_stack()
        global _idle_engines
        logger.debug(f"+++[mzm]+++++Process engine outputs, outputs: {outputs}, type: {type(outputs)}")
        # 检测负载分数异常高的情况
        if hasattr(outputs, 'scheduler_stats') and outputs.scheduler_stats is not None:
            stats = outputs.scheduler_stats
            num_waiting = getattr(stats, 'num_waiting_reqs', 0)
            num_running = getattr(stats, 'num_running_reqs', 0)

            # 如果分数异常高（999999, 999999 是空转标记），标记为空转
            if num_waiting >= 999999 and num_running >= 999999:
                engine_idx = getattr(outputs, 'engine_index', None)
                if engine_idx is not None and engine_idx not in _idle_engines:
                    logger.info(f"***********Marking engine {engine_idx} as idle (abnormal load score)")
                    _idle_engines.add(engine_idx)
            # 如果分数为 (0, 0) 且 engine 之前是 idle 的，说明已恢复，移除 idle 标记
            elif num_waiting == 0 and num_running == 0:
                engine_idx = getattr(outputs, 'engine_index', None)
                if engine_idx is not None and engine_idx in _idle_engines:
                    logger.info(f"***********Engine {engine_idx} recovered, removing idle mark")
                    _idle_engines.discard(engine_idx)
            # 其他正常分数：如果 engine 在 idle_engines 中，不移除（保持 idle 状态直到收到恢复通知）
            # 注意：这里不需要处理，因为 idle 的 engine 不会发送正常 stats

        # 调用原始处理逻辑
        logger.debug(f"+++++[mzm]+++++Calling original process_outputs for {type(outputs)}")
        return await original_process_outputs(self, outputs)

    # 应用 patch
    DPLBAsyncMPClient.get_core_engine_for_request = patched_get_core_engine_for_request
    DPLBAsyncMPClient.process_engine_outputs = patched_process_engine_outputs

    _patch_applied = True
    logger.info("PATCH: DPLBAsyncMPClient (skip idle engines)")


def get_idle_engines() -> set[int]:
    """获取当前被标记为空转的 engine 索引集合。"""
    return _idle_engines.copy()


def clear_idle_engines() -> None:
    """清除所有空转标记。"""
    global _idle_engines
    _idle_engines.clear()
    logger.info("Cleared all idle engine marks")