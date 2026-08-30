#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""
vllm-its-plugin: Zero-interruption Inference ITS Plugin

This plugin provides fault tolerance and deployment strategy execution
for vLLM running on Ascend NPUs.

Features:
- Fault Keep: Maintain service during worker failures
- Strategy Execution: Execute deployment strategies from decision center
- State Reporting: Report executor state to decision center
- Smooth Recovery: Recover service after deployment
"""

__version__ = "1.0.0"
PLUGIN_NAME = "vllm-its-plugin"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Zero-interruption Inference ITS Plugin for vLLM on Ascend NPUs"


def init_zero_interrupt():
    """Initialize the ITS plugin.

    This function should be called during vLLM startup to enable
    ITS functionality.
    """
    # Apply EngineCore patch
    from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.engine.engine_core_patch import patch_engine_core
    patch_engine_core()

    # Patch MultiprocExecutor
    import vllm.v1.executor.multiproc_executor as mp_module
    from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.its_multiproc_executor import ITSMultiprocExecutor
    from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor import ITSNPUWorker

    mp_module.MultiprocExecutor = ITSMultiprocExecutor
    mp_module.WorkerProc = ITSNPUWorker

    # Also patch if AscendMultiprocExecutor exists
    try:
        import vllm_ascend.patch.platform.patch_multiproc_executor as ascend_module
        ascend_module.MultiprocExecutor = ITSMultiprocExecutor
        ascend_module.WorkerProc = ITSNPUWorker
    except ImportError:
        pass


def register(manager):
    """Register ITS plugin patch with the PatchManager.

    This function is called by the vLLM custom plugins framework
    to register the ITS plugin patch.

    Args:
        manager: The PatchManager instance
    """
    from .patch import ZeroInterruptPluginPatch
    manager.register('zero_interrupt', ZeroInterruptPluginPatch)