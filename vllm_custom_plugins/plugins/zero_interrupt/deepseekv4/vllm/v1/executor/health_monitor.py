#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""ITS 插件健康监控模块。

本模块提供针对 worker 进程的健康监控增强功能，支持故障保持能力。
"""


import threading
import time
import weakref
from collections.abc import Callable

from vllm.logger import logger
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.constants import VLLM_ITS_HEALTH_CHECK_INTERVAL
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.types import ExecutorState

class ITSHealthMonitor:
    """ITS 插件增强健康监控器。

    此监控器跟踪 worker 进程健康状态并在故障时提供故障保持功能以维持服务可用性。
    """

    def __init__(
        self,
        workers: list,
        failure_callback: Callable[[], None] | None = None,
        fault_keep_enabled: bool = True,
        check_interval: int = VLLM_ITS_HEALTH_CHECK_INTERVAL,
    ):
        """初始化健康监控器。

        Args:
            workers: Worker 进程句柄列表
            failure_callback: Worker 故障检测到时的回调
            fault_keep_enabled: 启用故障保持模式
            check_interval: 健康检查间隔（秒）
        """
        self._workers = workers
        self._failure_callback = failure_callback
        self._fault_keep_enabled = fault_keep_enabled
        self._check_interval = check_interval

        self._monitor_thread: threading.Thread | None = None
        self._running = False
        self._worker_status: dict[int, str] = {}
        self._healthy_workers: set[int] = set()
        self._failure_handled = False  # Prevent repeated callback triggers
        self._stop_event = threading.Event()  # 用于停止线程的事件

    def start(self) -> None:
        """启动健康监控线程。"""
        if self._running:
            logger.warning("Health monitor already running")
            return

        # 确保事件是清除状态，防止上次 stop() 后事件仍处于设置状态
        self._stop_event.clear()
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._run_monitor,
            daemon=True,
            name="ITS-HealthMonitor",
        )
        self._monitor_thread.start()
        logger.info("Health monitor started")

    def _run_monitor(self) -> None:
        """主监控循环。"""
        while self._running:
            # 等待间隔时间或等待 stop() 信号
            # wait(timeout) 返回 True 如果事件被设置，False 如果超时
            self._stop_event.wait(timeout=self._check_interval)

            # wait() 返回后，必须再次检查 _running 标志
            # 因为可能存在竞态：wait() 超时返回后，stop() 刚好被调用
            if not self._running:
                break

            try:
                self._check_worker_health()
            except Exception as e:
                logger.error(f"Error in health check: {e}")

    def _check_worker_health(self) -> None:
        """检查所有 worker 进程的健康状态。"""
        if not self._workers:
            return

        current_healthy = set()
        dead_workers = []

        for idx, worker in enumerate(self._workers):
            try:
                proc = worker.proc
                if proc is None:
                    # Worker 未启动或已清理
                    self._worker_status[idx] = "not_started"
                elif proc.is_alive():
                    current_healthy.add(idx)
                    self._worker_status[idx] = "healthy"
                else:
                    dead_workers.append(idx)
                    self._worker_status[idx] = "dead"
            except Exception as e:
                logger.warning(f"Error checking worker {idx}: {e}")
                self._worker_status[idx] = "unknown"

        # Detect newly dead workers
        previously_healthy = self._healthy_workers
        new_dead = previously_healthy - current_healthy

        # 在调用回调之前更新 _healthy_workers
        # 这可以防止相同的故障在回调触发另一个健康检查周期时被再次检测到
        self._healthy_workers = current_healthy

        if new_dead:
            self._handle_worker_failure(new_dead)

        if dead_workers:
            logger.warning(f"Dead workers detected: {dead_workers}")

    def _handle_worker_failure(self, failed_workers: set[int]) -> None:
        """使用故障保持模式处理 worker 故障。

        设计：通过回调触发死 worker 的重启。

        Args:
            failed_workers: 失败 worker 索引集合
        """
        # 防止重复触发回调
        if self._failure_handled:
            logger.debug("Failure already handled, skipping duplicate callback")
            return

        logger.warning(f"Worker failure detected: {failed_workers}")

        # Update status
        for worker_idx in failed_workers:
            self._worker_status[worker_idx] = "dead"

        # Mark failure as handled
        self._failure_handled = True

        # 再次检查 _running 状态，确保 stop() 没有正在停止监控线程
        # 这可以防止 stop() 和 _handle_worker_failure 之间的竞态条件
        if not self._running:
            logger.info("Health monitor is stopping, skipping failure callback")
            return

        # 设计：触发回调以重启死的 workers
        if self._failure_callback:
            logger.info("Triggering failure callback to restart dead workers")
            self._failure_callback()

    def stop(self) -> None:
        """停止健康监控。

        设置 _failure_handled = True 阻止后续 callback。
        """
        # 无论如何都要设置 _failure_handled = True，防止竞态条件
        self._failure_handled = True

        if not self._running:
            return

        self._running = False
        # 触发事件以立即中断线程的 wait()
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3)
            self._monitor_thread = None
        # 重置事件状态，为下次 start() 做准备
        self._stop_event.clear()
        logger.info("Health monitor stopped")

    def get_worker_status(self) -> dict[int, str]:
        """获取所有 workers 的状态。

        Returns:
            映射 worker 索引到状态的字典
        """
        return self._worker_status.copy()

    def get_healthy_workers(self) -> set[int]:
        """获取健康 worker 索引集合。

        Returns:
            健康 worker 索引集合
        """
        return self._healthy_workers.copy()

    def is_running(self) -> bool:
        """检查监控器是否运行。

        Returns:
            运行中返回 True，否则返回 False
        """
        return self._running

    def reset_failure_state(self) -> None:
        """Workers 重启后重置故障状态。

        应在 workers 重启后调用，以允许健康监控器检测新的故障。
        """
        self._failure_handled = False
        logger.debug("Failure state reset")

    def update_workers(self, workers: list) -> None:
        """重启后更新 workers 引用。

        应在 workers 重启后调用，以确保健康监控器跟踪新的 workers。

        Args:
            workers: Worker 进程句柄的新列表
        """
        self._workers = workers
        self._healthy_workers = set()
        self._worker_status = {}
        # 注意：只有当 workers 非空时才重置 _failure_handled
        # 如果 workers 为空（idle mode），不应该重置，因为 stop() 已经设置了 _failure_handled=True
        if workers:
            self._failure_handled = False  # Reset flag so new failures can be detected
        logger.info("Workers reference updated in health monitor")


class ITSFailureCallback:
    """ITS 插件故障回调处理器。

    设计：
    - Worker 死亡由健康监控器检测
    - Executor 将处理清理/恢复
    """

    def __init__(
        self,
        executor,
        fault_keep_enabled: bool = True
    ):
        """初始化故障回调。

        Args:
            executor: Executor 实例
            fault_keep_enabled: 启用故障保持模式
        """
        self._executor = weakref.ref(executor)
        self._fault_keep_enabled = fault_keep_enabled

    def __call__(self) -> None:
        """处理 worker 故障。"""
        executor = self._executor()
        if not executor:
            return

        logger.warning("Worker failure detected")

        if self._fault_keep_enabled:
            # 设置状态为等待策略
            executor.executor_state = ExecutorState.WAITING_STRATEGY

            # Signal deployment event to notify EngineCore
            if hasattr(executor, 'wait_new_deployment'):
                logger.info("##############Worker failure: setting wait_new_deployment event")
                executor.wait_new_deployment.set()
                if hasattr(executor, '_trigger_busy_loop_callback'):
                    executor._trigger_busy_loop_callback()
                logger.info("Worker failure: wait_new_deployment event set")

        else:
            # 无故障保持 - 标记为失败并关闭
            if hasattr(executor, "is_failed"):
                executor.is_failed = True
            # Trigger shutdown via executor's shutdown mechanism
            if hasattr(executor, "shutdown"):
                logger.info("Shutting down executor due to worker failure")
                executor.shutdown()