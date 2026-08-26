# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Surgical patch for EngineArgs.create_engine_config.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_original_func = None


def set_original_func(func):
    global _original_func
    _original_func = func


def patched_create_engine_config(self, usage_context=None, headless=False):
    """
    Patched create_engine_config that simply calls the original function.
    The verify_with_parallel_config in patch_model.py already handles
    the case where TP size is not divisible by num_heads with just a warning.
    """
    return _original_func(self, usage_context, headless)
