#!/usr/bin/env python
"""
Apply async_kv_offload patch manually

Usage:
    python apply_async_kv_offload.py
    # Then start vLLM
"""

import os
import sys

# 设置环境变量
os.environ['VLLM_ASYNC_KV_OFFLOAD'] = '1'

# 导入并应用 patch
from vllm_custom_plugins.plugins.async_kv_offload import apply_async_kv_offload

print("Applying async_kv_offload patch...")
result = apply_async_kv_offload()
if result:
    print("Patch applied successfully!")
else:
    print("Patch application skipped or failed")
    sys.exit(1)