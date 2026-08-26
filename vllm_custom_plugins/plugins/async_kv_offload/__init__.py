# SPDX-License-Identifier: Apache-2.0
"""
Async KV Offload Plugin
"""

import os
import sys
import logging
import importlib

logger = logging.getLogger("async_kv_offload")

__version__ = "1.0.0"

_enabled = os.environ.get("VLLM_ASYNC_KV_OFFLOAD", "0") == "1"

logger.info(f"async_kv_offload loaded, _enabled={_enabled}")

# Patch target module path
_TARGET_MODULE = 'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector'


class ImportHook:
    """Import hook - 在 MooncakeConnector 导入时自动 patch"""

    def find_module(self, fullname, path=None):
        if fullname == _TARGET_MODULE:
            logger.info(f"Intercepting import of {fullname}")
            print(f"DEBUG: Intercepting {fullname}!")
            return self
        return None

    def load_module(self, fullname):
        # 先移除自己避免递归
        if fullname == _TARGET_MODULE and _hooks_installed:
            try:
                sys.meta_path.remove(_import_hook_instance)
            except (ValueError, AttributeError):
                pass

        # 检查是否已加载
        if fullname in sys.modules:
            return sys.modules[fullname]

        # 使用原始导入方式
        module = importlib.import_module(fullname)

        # 应用 patch
        if _enabled and fullname == _TARGET_MODULE:
            try:
                from . import patch as _patch_module
                result = _patch_module.apply_patch()
                logger.info(f"apply_patch result: {result}")
                print(f"DEBUG: apply_patch result: {result}")
            except Exception as e:
                logger.exception(f"Failed to apply patch: {e}")
                print(f"DEBUG: Failed to apply patch: {e}")

        return module


_import_hook_instance = None
_hooks_installed = False


def register(manager):
    """注册 import hook"""
    global _hooks_installed, _import_hook_instance

    # logger.info(f"register called, _enabled={_enabled}, _hooks_installed={_hooks_installed}")
    # print(f"DEBUG: register called, _enabled={_enabled}")

    if _enabled and not _hooks_installed:
        # 创建并安装 import hook
        _import_hook_instance = ImportHook()
        sys.meta_path.insert(0, _import_hook_instance)
        _hooks_installed = True
        logger.info("Import hook installed for async_kv_offload")
        print("DEBUG: Import hook installed!")
    elif not _enabled:
        logger.info("async_kv_offload not enabled")
        print("DEBUG: not enabled")