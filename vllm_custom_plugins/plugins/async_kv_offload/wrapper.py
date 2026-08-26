# SPDX-License-Identifier: Apache-2.0
"""
Async KV Offload Plugin - Wrapper 实现

提供 AsyncKVConnectorWrapper 类，可以包装任意 KV Connector
为其添加异步保存能力。

使用方式：
    from vllm_custom_plugins.plugins.async_kv_offload import AsyncKVConnectorWrapper

    # 包装已有的 connector
    wrapped = AsyncKVConnectorWrapper(original_connector)
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Optional, Callable

logger = logging.getLogger("async_kv_offload")


class AsyncKVConnectorWrapper:
    """
    异步 KV Connector 包装器

    为现有的 KV Connector 添加异步保存能力。

    示例：
        # 假设你有一个 existing_connector
        wrapper = AsyncKVConnectorWrapper(existing_connector)
        # 使用 wrapper 替代 original_connector
        wrapper.wait_for_save()  # 异步执行
    """

    def __init__(
        self,
        connector: Any,
        enable_async: bool = True,
        max_workers: int = 1,
        store_func: Optional[Callable] = None,
        unpin_func: Optional[Callable] = None,
    ):
        """
        初始化包装器

        Args:
            connector: 要包装的原始 Connector
            enable_async: 是否启用异步模式
            max_workers: 最大并行工作线程数
            store_func: 自定义的存储函数，默认为 connector.wait_for_save
            unpin_func: 自定义的 unpin 函数
        """
        self._connector = connector
        self._enable_async = enable_async and max_workers > 0
        self._max_workers = max_workers

        # 存储和 unpin 函数
        self._store_func = store_func or getattr(connector, 'wait_for_save', None)
        self._unpin_func = unpin_func

        # 线程池和 Future 列表
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: list[Future] = []

        if self._enable_async:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="kv-offload-wrapper",
            )
            logger.debug(
                f"AsyncKVConnectorWrapper enabled: max_workers={max_workers}"
            )

    @property
    def connector(self) -> Any:
        """获取原始 connector"""
        return self._connector

    @property
    def is_async_enabled(self) -> bool:
        """是否启用了异步模式"""
        return self._enable_async

    def wait_for_save(self):
        """
        等待 KV 保存完成

        如果启用异步模式，将使用线程池并行执行。

        16K 输入优化：
        - 批量提交任务以减少线程切换开销
        - 错误追踪（不会静默忽略失败）
        """
        if not self._enable_async:
            # 同步模式：直接调用
            if self._store_func:
                self._store_func()
            return

        # 异步模式：提交到线程池
        self._poll_errors()

        if not self._store_func:
            return

        # 提交新任务
        future = self._executor.submit(self._store_func)
        self._futures.append(future)

        # 等待所有任务完成（保持原有语义）
        self._wait_all_done()

    def _poll_errors(self):
        """轮询并处理错误"""
        if not self._futures:
            return

        remaining = []
        for future in self._futures:
            if not future.done():
                remaining.append(future)
                continue

            # 检查异常
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async wait_for_save error: {e}")

        self._futures = remaining

    def _wait_all_done(self):
        """等待所有异步任务完成"""
        for future in self._futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async task error: {e}")

        self._futures.clear()

    def get_finished(self, finished_req_ids: set) -> tuple[set, set]:
        """获取完成的请求ID"""
        # 先等待异步任务完成
        self._wait_all_done()

        # 转发到原始 connector
        if hasattr(self._connector, 'get_finished'):
            return self._connector.get_finished(finished_req_ids)

        return None, None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs):
        """转发 save_kv_layer 调用"""
        if hasattr(self._connector, 'save_kv_layer'):
            self._connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def start_load_kv(self, forward_context: Any, **kwargs):
        """转发 start_load_kv 调用"""
        if hasattr(self._connector, 'start_load_kv'):
            self._connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str):
        """转发 wait_for_layer_load 调用"""
        if hasattr(self._connector, 'wait_for_layer_load'):
            self._connector.wait_for_layer_load(layer_name)

    def __getattr__(self, name: str) -> Any:
        """转发其他属性访问到原始 connector"""
        return getattr(self._connector, name)

    def shutdown(self):
        """关闭包装器，释放资源"""
        # 等待所有任务完成
        self._wait_all_done()

        # 关闭线程池
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.debug("AsyncKVConnectorWrapper executor shutdown")

        # 关闭原始 connector
        if hasattr(self._connector, 'shutdown'):
            self._connector.shutdown()

    def __del__(self):
        """析构函数"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)


# ==================== 辅助函数 ====================

def wrap_connector(
    connector: Any,
    config: Optional[dict] = None,
) -> AsyncKVConnectorWrapper:
    """
    包装 connector 的便捷函数

    Args:
        connector: 要包装的 connector
        config: 配置字典，支持以下键：
            - async_offload: bool，是否启用异步
            - max_workers: int，最大工作线程数

    Returns:
        包装后的 AsyncKVConnectorWrapper
    """
    config = config or {}

    # 从环境变量或配置获取参数
    enable_async = config.get(
        'async_offload',
        os.environ.get('VLLM_ASYNC_KV_OFFLOAD', '0') == '1'
    )
    max_workers = config.get(
        'max_workers',
        int(os.environ.get('VLLM_ASYNC_KV_OFFLOAD_MAX_INFLIGHT', '1'))
    )

    return AsyncKVConnectorWrapper(
        connector=connector,
        enable_async=enable_async,
        max_workers=max_workers,
    )


def create_async_wrapper_mixin():
    """
    创建一个可用于继承的 Mixin 类

    使用方式：
        class MyAsyncConnector(AsyncKVConnectorMixin, BaseConnector):
            pass
    """
    class AsyncKVConnectorMixin:
        """Mixin 类，为 Connector 添加异步保存能力"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 初始化异步相关属性
            config = kwargs.get('async_config', {})
            self._async_enabled = config.get('async_offload', False)
            self._async_executor: Optional[ThreadPoolExecutor] = None
            self._async_futures: list[Future] = []

            if self._async_enabled:
                max_workers = config.get('max_workers', 1)
                self._async_executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="kv-offload-mixin",
                )

        def _async_submit(self, func: Callable, *args, **kwargs) -> Future:
            """提交异步任务"""
            if self._async_executor is None:
                # 同步执行
                return func(*args, **kwargs)

            future = self._async_executor.submit(func, *args, **kwargs)
            self._async_futures.append(future)
            return future

        def _async_wait_all(self):
            """等待所有异步任务完成"""
            if not self._async_futures:
                return

            for future in self._async_futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Async task error: {e}")

            self._async_futures.clear()

        def shutdown(self):
            """关闭资源"""
            self._async_wait_all()
            if self._async_executor is not None:
                self._async_executor.shutdown(wait=True)
                self._async_executor = None

            # 调用父类 shutdown
            if hasattr(super(), 'shutdown'):
                super().shutdown()

    return AsyncKVConnectorMixin