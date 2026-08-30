#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Heterogeneous-TP partition helpers for ``vllm.distributed.utils``.

vllm_plugins runs on a stock v0.23.0 tree where these helpers do not exist;
several hetero patches (DSA-CP, linear layers) import them from
``vllm.distributed.utils``.
"""

_PATCHED = False


def _validate_tp_sharding_ratios(tp_size, tp_sharding_ratios):
    if len(tp_sharding_ratios) != tp_size:
        raise ValueError(
            f"tp_sharding_ratios length {len(tp_sharding_ratios)} must "
            f"equal tp_size {tp_size}."
        )
    if any(ratio <= 0 for ratio in tp_sharding_ratios):
        raise ValueError(
            "tp_sharding_ratios entries must be positive integers, got "
            f"{tp_sharding_ratios}."
        )


def get_tp_partition_size(total_size, tp_rank, tp_size, tp_sharding_ratios=None):
    if tp_size == 1:
        return total_size
    if tp_sharding_ratios is None:
        from vllm.distributed.utils import divide

        return divide(total_size, tp_size)
    _validate_tp_sharding_ratios(tp_size, tp_sharding_ratios)
    total_ratio = sum(tp_sharding_ratios)
    sizes = [total_size * r // total_ratio for r in tp_sharding_ratios]
    sizes[-1] += total_size - sum(sizes)
    return sizes[tp_rank]


def get_tp_partition_offset(total_size, tp_rank, tp_size, tp_sharding_ratios=None):
    if tp_size == 1:
        return 0
    if tp_sharding_ratios is None:
        from vllm.distributed.utils import divide

        return tp_rank * divide(total_size, tp_size)
    _validate_tp_sharding_ratios(tp_size, tp_sharding_ratios)
    total_ratio = sum(tp_sharding_ratios)
    return sum(
        total_size * tp_sharding_ratios[i] // total_ratio
        for i in range(tp_rank)
    )


def get_current_tp_sharding_ratios():
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return None
    pc = cfg.parallel_config
    if not getattr(pc, "is_heterogeneous_tp", False):
        return None
    return pc.get_sharding_ratios_for_dp(pc.data_parallel_rank)


def apply_hetero_distributed_utils_patch():
    global _PATCHED
    if _PATCHED:
        return
    import vllm.distributed.utils as mod

    if not hasattr(mod, "get_tp_partition_size"):
        mod.get_tp_partition_size = get_tp_partition_size
    if not hasattr(mod, "get_tp_partition_offset"):
        mod.get_tp_partition_offset = get_tp_partition_offset
    if not hasattr(mod, "get_current_tp_sharding_ratios"):
        mod.get_current_tp_sharding_ratios = get_current_tp_sharding_ratios
    _PATCHED = True
