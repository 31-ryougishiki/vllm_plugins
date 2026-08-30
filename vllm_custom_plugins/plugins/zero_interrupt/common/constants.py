#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Constants and configuration for ITS plugin.

This module provides all configuration values for the ITS plugin.
The single source of truth for all settings.
"""

import os
import uuid

# =============================================================================
# Plugin Metadata
# =============================================================================
PLUGIN_NAME = "vllm-its-plugin"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Zero-interruption Inference ITS Plugin for vLLM on Ascend NPUs"

# =============================================================================
# Decision Center Configuration
# =============================================================================
VLLM_ITS_DECISION_CENTER_URL = os.getenv("VLLM_ITS_DECISION_CENTER_URL", "http://127.0.0.1:8080")
VLLM_ITS_DECISION_CENTER_TOKEN = os.getenv("VLLM_ITS_DECISION_CENTER_TOKEN", "")

# =============================================================================
# Executor Configuration
# =============================================================================
# Executor http server 启动端口
VLLM_ITS_HTTP_SERVER_PORT_START = int(os.getenv("VLLM_ITS_HTTP_SERVER_PORT_START", "8001"))
VLLM_ITS_HEALTH_CHECK_INTERVAL = int(os.getenv("VLLM_ITS_HEALTH_CHECK_INTERVAL", "5"))
VLLM_ITS_STRATEGY_TIMEOUT = int(os.getenv("VLLM_ITS_STRATEGY_TIMEOUT", "300"))
VLLM_ITS_MAX_RETRY_COUNT = int(os.getenv("VLLM_ITS_MAX_RETRY_COUNT", "3"))

# =============================================================================
# Feature Flags
# =============================================================================
VLLM_ITS_ENABLE_FAULT_KEEP = os.getenv("VLLM_ITS_ENABLE_FAULT_KEEP", "true").lower() in ("true", "1")
VLLM_ITS_ENABLE_PD_REBUILD = os.getenv("VLLM_ITS_ENABLE_PD_REBUILD", "true").lower() in ("true", "1")

# 从环境变量 VLLM_SERVICE_ID 获取，生成 UUID 作为默认值
VLLM_SERVICE_ID = os.getenv("VLLM_SERVICE_ID", str(uuid.uuid4()))

# =============================================================================
# DeepSeek-V4 Feature Flag
# =============================================================================
# 当 VLLM_ITS_DEEPSEEK_V4=1 时，所有与 0829 分支冲突的 patch/整文件替换
# 均切换到 zero_interrupt/deepseekv4/ 目录下的实现；
# 未设置或为 0 时，默认使用主目录（vllm_plugins_0829 合并过来的实现）。
VLLM_ITS_DEEPSEEK_V4 = (
    os.getenv("VLLM_ITS_DEEPSEEK_V4", "0").strip().lower()
    in ("1", "true", "yes", "on")
)


def is_deepseek_v4_enabled() -> bool:
    """Return True when the DeepSeek-V4 patch family should be selected.

    Read the environment at call time so callers that set
    ``VLLM_ITS_DEEPSEEK_V4`` before invoking the dispatcher work correctly.
    """
    return (
        os.getenv("VLLM_ITS_DEEPSEEK_V4", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )

# =============================================================================
# API Endpoints
# =============================================================================
API_INIT_EXECUTOR_STATE = "/api/v1/decision_center/init_executor_state"
API_DEPLOY = "/api/v1/executor/deploy"
API_REPORT_DEPLOY_STATUS = "/api/v1/decision_center/report_deploy_status"