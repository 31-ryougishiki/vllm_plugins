#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# [merge-0829] 运行时分发器。
#
# setup.py 会把本文件以及两个实现文件安装到
# vllm_ascend/patch/worker/ 下：
#   patch_qwen3_5.py                  <- 本分发器
#   patch_qwen3_5_deepseek_v4.py      <- DeepSeek-V4 / v0.23 实现
#   patch_qwen3_5_0829.py             <- 0829 非对称实现
#
# import 时根据 VLLM_ITS_DEEPSEEK_V4 选择实现，因此安装阶段不再需要
# 环境变量。
#
import importlib
import os
import sys

_impl_name = (
    "patch_qwen3_5_deepseek_v4"
    if os.environ.get("VLLM_ITS_DEEPSEEK_V4", "0").strip().lower()
    in ("1", "true", "yes", "on")
    else "patch_qwen3_5_0829"
)

_impl = importlib.import_module(
    f"vllm_ascend.patch.worker.{_impl_name}"
)

# 让 `from vllm_ascend.patch.worker.patch_qwen3_5 import X` 取到实现模块。
sys.modules[__name__] = _impl
