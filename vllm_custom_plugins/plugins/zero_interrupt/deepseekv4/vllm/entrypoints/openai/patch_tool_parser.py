#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""DeepSeek-V4 tool-call parser registration patch.

Some deployed vLLM builds do not register ``deepseek_v4`` in
``vllm.tool_parsers.ToolParserManager`` (the registration table was added in
newer v0.23.0 builds).  ``vllm serve --tool-call-parser deepseek_v4`` is
validated in ``api_server.validate_api_server_args`` before model loading, so
the parser must be registered while the general plugin entry point runs.

Preference order:
1. real ``vllm.tool_parsers.deepseekv4_tool_parser.DeepSeekV4ToolParser``
2. ``deepseek_v32`` parser (closest DSML-compatible fallback)
3. ``deepseek_v31`` parser
4. ``deepseek_v3`` parser
"""

from __future__ import annotations

import importlib.util

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False

_CANDIDATES = (
    ("vllm.tool_parsers.deepseekv4_tool_parser", "DeepSeekV4ToolParser"),
    ("vllm.tool_parsers.deepseekv32_tool_parser", "DeepSeekV32ToolParser"),
    ("vllm.tool_parsers.deepseekv31_tool_parser", "DeepSeekV31ToolParser"),
    ("vllm.tool_parsers.deepseekv3_tool_parser", "DeepSeekV3ToolParser"),
)


def _registered_tool_parsers(manager) -> set[str]:
    try:
        return set(manager.list_registered())
    except Exception:
        parsers = set(getattr(manager, "tool_parsers", {}).keys())
        parsers.update(getattr(manager, "lazy_parsers", {}).keys())
        return parsers


def _register(manager, name: str, module_path: str, class_name: str) -> None:
    register_lazy = getattr(manager, "register_lazy_module", None)
    if register_lazy is not None:
        register_lazy(name, module_path, class_name)
        return
    # Very old vLLM fallback.
    lazy_parsers = getattr(manager, "lazy_parsers", None)
    if isinstance(lazy_parsers, dict):
        lazy_parsers[name] = (module_path, class_name)
        return
    raise RuntimeError("ToolParserManager has no registration API")


def apply_deepseek_v4_tool_parser_patch():
    global _PATCHED
    if _PATCHED:
        return

    from vllm.tool_parsers import ToolParserManager

    registered = _registered_tool_parsers(ToolParserManager)
    if "deepseek_v4" in registered:
        _PATCHED = True
        return

    selected = None
    for module_path, class_name in _CANDIDATES:
        if importlib.util.find_spec(module_path) is None:
            continue
        selected = (module_path, class_name)
        break

    if selected is None:
        raise ImportError(
            "No DeepSeek tool parser module is importable; cannot register "
            "'deepseek_v4'. Upgrade vllm to a build that ships "
            "vllm.tool_parsers.deepseekv4_tool_parser."
        )

    _register(ToolParserManager, "deepseek_v4", selected[0], selected[1])
    logger.info(
        "Registered tool call parser 'deepseek_v4' -> %s.%s",
        selected[0],
        selected[1],
    )
    _PATCHED = True
