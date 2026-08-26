# SPDX-License-Identifier: Apache-2.0
"""
Async KV Offload Plugin

提供异步 KV offload 能力，不阻塞推理主流程

默认行为：非阻塞模式 - 保存操作在后台异步执行，不等待完成

环境变量：
- VLLM_ASYNC_KV_OFFLOAD=1: 启用异步 KV offload
- VLLM_ASYNC_KV_OFFLOAD_MODE: 运行模式
    - nonblocking (默认): 非阻塞，立即返回
    - sync: 同步等待完成
    - async: 异步等待（需要 await）
- VLLM_ASYNC_KV_OFFLOAD_WORKERS: 线程池工作线程数（默认根据输入长度动态调整）
- VLLM_ASYNC_KV_OFFLOAD_MAX_INFLIGHT: 最大待处理任务数（默认 16，超过则等待）
- VLLM_ASYNC_KV_OFFLOAD_BATCH_SIZE: 批量提交大小（默认 2，同时提交多个 connector）
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Set

logger = logging.getLogger("async_kv_offload")

_enabled = os.environ.get("VLLM_ASYNC_KV_OFFLOAD", "0") == "1"
_async_mode = os.environ.get("VLLM_ASYNC_KV_OFFLOAD_MODE", "nonblocking")

# 最大待处理任务数，超过则等待（16K 输入时任务耗时更长，需要限制堆积）
_max_inflight = int(os.environ.get("VLLM_ASYNC_KV_OFFLOAD_MAX_INFLIGHT", "16"))
# 批量大小（同时提交多个 connector 的保存任务）
_batch_size = int(os.environ.get("VLLM_ASYNC_KV_OFFLOAD_BATCH_SIZE", "2"))
# 默认线程数（会被动态调整覆盖）
_default_workers = int(os.environ.get("VLLM_ASYNC_KV_OFFLOAD_WORKERS", "8"))

# 线程池用于执行阻塞的 IO 操作（动态调整大小）
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

# 任务跟踪
_pending_futures: Set[Future] = set()
_pending_lock = threading.Lock()
_pending_counter = 0  # 用于日志追踪
_counter_lock = threading.Lock()

# 统计信息（用于监控和调优）
_stats = {
    "submitted": 0,
    "completed": 0,
    "errors": 0,
    "rejected": 0,
    "max_inflight": 0,
}
_stats_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """获取或创建线程池（延迟初始化）"""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_default_workers,
                    thread_name_prefix="async_kv_offload"
                )
                logger.info(f"Created thread pool with {_default_workers} workers")
    return _executor


def _adjust_thread_pool_by_config():
    """
    根据环境变量配置动态调整线程池

    环境变量：
    - VLLM_ASYNC_KV_OFFLOAD_WORKERS: 直接指定线程数（覆盖动态调整）
    - VLLM_ASYNC_KV_OFFLOAD_AUTO_TUNE: 是否启用自动调优 (0/1)
    """
    global _default_workers, _executor

    # 如果用户显式指定了 WORKERS，不动态调整
    if os.environ.get("VLLM_ASYNC_KV_OFFLOAD_WORKERS"):
        return

    auto_tune = os.environ.get("VLLM_ASYNC_KV_OFFLOAD_AUTO_TUNE", "1") == "1"
    if not auto_tune:
        return

    # 根据并发度和队列状态自动调整
    # 如果 pending 任务经常达到上限，增加线程
    with _stats_lock:
        max_inflight = _stats.get("max_inflight", 0)
        rejected = _stats.get("rejected", 0)

    current_workers = _default_workers

    # 如果经常有任务被拒绝，增加线程
    if rejected > 10 and current_workers < 32:
        new_workers = min(32, current_workers + 4)
        logger.info(f"Auto-tuning thread pool: {current_workers} -> {new_workers} workers (rejected={rejected})")
        _default_workers = new_workers

        with _executor_lock:
            if _executor is not None:
                old_executor = _executor
                _executor = ThreadPoolExecutor(
                    max_workers=new_workers,
                    thread_name_prefix="async_kv_offload"
                )
                old_executor.shutdown(wait=False)


def _wait_for_slot(timeout: float = 5.0) -> bool:
    """
    等待一个待处理槽位（背压控制）

    Returns:
        True: 获得槽位
        False: 超时放弃（不应该发生）
    """
    global _pending_futures, _max_inflight, _stats

    start = time.time()
    while True:
        with _pending_lock:
            current_inflight = len(_pending_futures)
            _stats["max_inflight"] = max(_stats["max_inflight"], current_inflight)

            if current_inflight < _max_inflight:
                return True

        if time.time() - start > timeout:
            logger.warning(f"Backpressure: waited {timeout}s for slot, current inflight={current_inflight}")
            return False

        time.sleep(0.01)  # 避免忙等待


def _submit_task(func, *args, **kwargs) -> Optional[Future]:
    """
    提交任务到线程池，带背压控制和错误追踪

    Returns:
        Future 对象或 None（如果被拒绝或出错）
    """
    global _pending_futures, _pending_counter, _stats

    # 背压控制：等待槽位
    if not _wait_for_slot(timeout=10.0):
        with _stats_lock:
            _stats["rejected"] += 1
        logger.warning("Task rejected due to backpressure")
        return None

    executor = _get_executor()

    def wrapped_func():
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background wait_for_save failed: {e}")
            with _stats_lock:
                _stats["errors"] += 1
            raise

    try:
        future = executor.submit(wrapped_func)

        def _on_done(f: Future):
            with _pending_lock:
                _pending_futures.discard(f)
            with _stats_lock:
                _stats["completed"] += 1
            if f.exception():
                logger.error(f"wait_for_save exception: {f.exception()}")

        future.add_done_callback(_on_done)

        with _pending_lock:
            _pending_futures.add(future)

        with _counter_lock:
            _pending_counter += 1
            task_id = _pending_counter

        with _stats_lock:
            _stats["submitted"] += 1

        logger.debug(f"Submitted task #{task_id}, current inflight={len(_pending_futures)}")
        return future

    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        with _stats_lock:
            _stats["errors"] += 1
        return None


def get_stats() -> dict:
    """获取统计信息（用于监控）"""
    with _stats_lock:
        with _pending_lock:
            return {
                **_stats,
                "current_inflight": len(_pending_futures),
            }


def reset_stats():
    """重置统计信息"""
    with _stats_lock:
        _stats["submitted"] = 0
        _stats["completed"] = 0
        _stats["errors"] = 0
        _stats["rejected"] = 0
        _stats["max_inflight"] = 0


def shutdown():
    """关闭线程池（程序退出时调用）"""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = None
            logger.info("async_kv_offload executor shutdown")


def apply_patch():
    """应用 patch - 默认使用非阻塞模式"""
    global _enabled, _async_mode
    if not _enabled:
        logger.info("async_kv_offload not enabled")
        return False

    try:
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnector
    except ImportError as e:
        logger.warning(f"MooncakeConnector not found: {e}")
        return False

    # 如果已经 patch 过了，跳过
    if hasattr(MooncakeConnector, '_async_kv_offload_patched'):
        logger.info("async_kv_offload already applied")
        return True

    # 获取原始方法
    original_wait_for_save = MooncakeConnector.wait_for_save
    original_get_finished = MooncakeConnector.get_finished

    # 保存原始方法
    MooncakeConnector._original_wait_for_save = original_wait_for_save
    MooncakeConnector._original_get_finished = original_get_finished
    MooncakeConnector._async_kv_offload_patched = True

    # ========== 根据模式选择实现 ==========
    if _async_mode == "sync":
        # 同步模式：直接调用原始方法
        logger.info("Using sync mode for async_kv_offload")

        def patched_wait_for_save(self):
            return original_wait_for_save(self)

        def patched_get_finished(self, finished_req_ids):
            return original_get_finished(self, finished_req_ids)

    elif _async_mode == "async":
        # 异步模式：需要调用方 await
        logger.info("Using async mode for async_kv_offload")

        async def patched_wait_for_save(self):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                _get_executor(), lambda: original_wait_for_save(self)
            )

        async def patched_get_finished(self, finished_req_ids):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                _get_executor(), lambda: original_get_finished(self, finished_req_ids)
            )

    else:  # nonblocking (默认)
        # 非阻塞模式：
        # - wait_for_save: 非阻塞，后台异步执行（提高 Prefill 吞吐量）
        # - get_finished: 同步返回真实状态（保证 KV 正确释放）
        logger.info("Using nonblocking mode for async_kv_offload (RECOMMENDED)")
        logger.info(f"  - max_inflight: {_max_inflight}")
        logger.info(f"  - batch_size: {_batch_size}")
        logger.info(f"  - default_workers: {_default_workers}")

        def patched_wait_for_save(self):
            """非阻塞等待 KV 保存 - 立即返回，后台异步执行

            16K 输入优化点：
            1. 自动调优线程池（根据背压情况动态增加线程）
            2. 背压控制（限制待处理任务数，防止 OOM）
            3. Future 追踪（错误不会被静默忽略）
            """
            _adjust_thread_pool_by_config()

            logger.debug("nonblocking_wait_for_save: submitting to background")
            _submit_task(original_wait_for_save, self)
            return None

        def patched_get_finished(self, finished_req_ids):
            """同步获取完成请求 - 返回真实状态，保证 KV 正确释放"""
            logger.debug(f"get_finished: checking {len(finished_req_ids)} reqs")
            return original_get_finished(self, finished_req_ids)

    # 替换原方法
    MooncakeConnector.wait_for_save = patched_wait_for_save
    MooncakeConnector.get_finished = patched_get_finished

    # 添加便捷访问方法
    async def async_wait_for_save(self):
        """显式异步等待（用于 async 上下文）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _get_executor(), lambda: original_wait_for_save(self)
        )

    async def async_get_finished(self, finished_req_ids):
        """显式异步获取"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _get_executor(), lambda: original_get_finished(self, finished_req_ids)
        )

    def sync_wait_for_save(self):
        """显式同步等待（强制等待完成）"""
        return original_wait_for_save(self)

    def sync_get_finished(self, finished_req_ids):
        """显式同步获取"""
        return original_get_finished(self, finished_req_ids)

    MooncakeConnector.async_wait_for_save = async_wait_for_save
    MooncakeConnector.async_get_finished = async_get_finished
    MooncakeConnector.sync_wait_for_save = sync_wait_for_save
    MooncakeConnector.sync_get_finished = sync_get_finished

    # 添加统计和工具方法
    def get_async_stats(self):
        """获取统计信息"""
        return get_stats()

    MooncakeConnector.async_kv_offload_stats = property(get_async_stats)
    MooncakeConnector.async_kv_offload_reset_stats = lambda self: reset_stats()
    MooncakeConnector.async_kv_offload_shutdown = lambda self: shutdown()

    logger.info(f"[x] async_kv_offload patched MooncakeConnector (mode={_async_mode})")
    logger.info("    默认方法已替换:")
    logger.info("    - wait_for_save(): 非阻塞，立即返回，后台异步执行")
    logger.info("    - get_finished(): 同步返回真实状态（保证 KV 正确释放）")
    logger.info("    显式调用方法:")
    logger.info("    - async_wait_for_save(): 需要 await")
    logger.info("    - async_get_finished(): 需要 await")
    logger.info("    - sync_wait_for_save(): 强制同步等待")
    logger.info("    - sync_get_finished(): 强制同步获取")
    logger.info("    工具方法:")
    logger.info("    - .async_kv_offload_stats: 获取统计信息")
    logger.info("    - .async_kv_offload_reset_stats(): 重置统计")
    logger.info("    - .async_kv_offload_shutdown(): 关闭线程池")

    return True


# 移除模块级别的自动 apply
# if _enabled:
#     try:
#         apply_patch()
#     except Exception as e:
#         logger.warning(f"Failed to apply async_kv_offload: {e}")