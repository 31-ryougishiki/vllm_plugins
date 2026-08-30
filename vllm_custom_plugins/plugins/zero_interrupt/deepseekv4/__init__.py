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
    ITS functionality.  The DeepSeek-V4 family is defined by
    ``deepseekv4/patch.py:apply()`` (control plane + all heterogeneous
    model/data-plane patches); applying only the executor swap here would
    start a half-patched service, so delegate to the full patch entry point.
    """
    from .patch import apply

    return apply()


def register(manager):
    """Register ITS plugin patch with the PatchManager.

    This function is called by the vLLM custom plugins framework
    to register the ITS plugin patch.

    Args:
        manager: The PatchManager instance
    """
    from .patch import ZeroInterruptPluginPatch
    manager.register('zero_interrupt', ZeroInterruptPluginPatch)