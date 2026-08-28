#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Ascend forward-context patches for heterogeneous TP.

Port of hetero_cp ``vllm_ascend/ascend_forward_context.py`` changes:
- per-DP padded stream lengths and per-DP TP sizes are published through
  ``_EXTRA_CTX`` so the MoE gather/reduce paths can unpad rank by rank;
- token capacities / MC2 padding are aligned to ``lcm(tp_sizes)``;
- A3 MC2 kernels fall back to ALLGATHER when experts don't divide the EP
  world size (e.g. 256 experts / 15 ranks).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.forward_context import get_forward_context

_PATCHED = False


def _is_hetero(vllm_config: VllmConfig) -> bool:
    return bool(
        getattr(vllm_config.parallel_config, "is_heterogeneous_tp", False)
    )


@contextmanager
def _patched_set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    skip_compiled: bool = False,
    max_tokens_across_pcp: int = 0,
    draft_attn_metadatas=None,
    has_sinks=False,
    input_ids=None,
    eplb_heat_collection_status: bool = False,
):
    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }
    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas
        forward_context.input_ids = input_ids

        import vllm_ascend.ascend_forward_context as afc
        from vllm_ascend.ops.fused_moe.moe_comm_method import (
            get_moe_comm_method,
        )

        max_num_tokens = (
            int(num_tokens_across_dp.max().item())
            if num_tokens_across_dp is not None else num_tokens
        )
        moe_comm_type = afc.select_moe_comm_method(
            max_num_tokens, vllm_config, is_draft_model
        )
        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()
        forward_context.in_profile_run = in_profile_run
        forward_context.capturing = False
        forward_context.sinks = has_sinks

        mmrs_fusion = tp_world_size <= 8

        is_context_moe_model = (
            afc.is_drafter_moe_model(vllm_config)
            if is_draft_model else afc.is_moe_model(vllm_config)
        )
        if is_context_moe_model:
            flash_comm_v1_enabled = (
                afc.enable_sp(vllm_config) and num_tokens is not None
            )
            mmrs_fusion = False
        elif is_draft_model:
            flash_comm_v1_enabled = False
        else:
            flash_comm_v1_enabled = (
                afc.enable_sp(vllm_config)
                and num_tokens is not None
                and num_tokens > 1000
            )
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
        forward_context.flashcomm_v2_enabled = (
            afc.flashcomm2_enable()
            and tp_world_size > 1
            and num_tokens is not None
        )

        forward_context.pad_size = 0
        if (
            forward_context.flash_comm_v1_enabled
            or forward_context.flashcomm_v2_enabled
        ):
            pad_size = (
                tp_world_size - (num_tokens % tp_world_size)
            ) % tp_world_size
            forward_context.pad_size = pad_size

        forward_context.is_first_layer = True
        forward_context.layer_idx = None
        if afc.has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model
        forward_context.is_draft_model_prefill = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        # Under heterogeneous TP the DP group of an orphaned TP rank is a
        # singleton, so get_dp_group().world_size can be 1 even though the
        # logical DP size is > 1. Read the true DP size from the config.
        is_hetero = _is_hetero(vllm_config)
        true_dp_size = (
            vllm_config.parallel_config.data_parallel_size
            if is_hetero else get_dp_group().world_size
        )
        forward_context.per_dp_padded_lengths = None
        forward_context.per_dp_tp_sizes = None
        if true_dp_size > 1 and forward_context.dp_metadata is not None:
            dp_meta = forward_context.dp_metadata
            max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
            if is_hetero:
                pc = vllm_config.parallel_config
                tp_sizes = [
                    pc.get_tp_size_for_dp(i) for i in range(true_dp_size)
                ]
                per_dp = []
                for i in range(true_dp_size):
                    n = int(dp_meta.num_tokens_across_dp_cpu[i].item())
                    per_dp.append(
                        ((n + tp_sizes[i] - 1) // tp_sizes[i]) * tp_sizes[i]
                    )
                forward_context.per_dp_padded_lengths = per_dp
                forward_context.per_dp_tp_sizes = tp_sizes
                if (
                    forward_context.flash_comm_v1_enabled
                    or forward_context.flashcomm_v2_enabled
                ):
                    _pl = max(per_dp)
                    _align = math.lcm(*tp_sizes)
                    if _pl % _align != 0:
                        _pl += _align - (_pl % _align)
                    forward_context.padded_length = _pl
                    forward_context.pad_size = (
                        forward_context.padded_length - num_tokens
                    )
            else:
                if (
                    forward_context.flash_comm_v1_enabled
                    or forward_context.flashcomm_v2_enabled
                ):
                    padded_length = (
                        (max_tokens_across_dp + tp_world_size - 1)
                        // tp_world_size * tp_world_size
                    )
                    pad_size = padded_length - num_tokens
                    forward_context.padded_length = padded_length
                    forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp
        forward_context.eplb_heat_collection_status = (
            eplb_heat_collection_status
        )

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            if is_hetero:
                align = math.lcm(
                    *[
                        vllm_config.parallel_config.get_tp_size_for_dp(i)
                        for i in range(true_dp_size)
                    ]
                )
            else:
                align = tp_world_size
            forward_context.padded_num_tokens = (
                math.ceil(max_tokens_across_dp / align) * align
            )
            reserved_mc2_mask = afc.get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[
                    : forward_context.padded_num_tokens
                ]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask
        try:
            yield
        finally:
            pass


def _patched_set_mc2_tokens_capacity(
    vllm_config, max_num_reqs, uniform_decode_query_len
):
    import vllm_ascend.ascend_forward_context as afc

    if afc._mc2_tokens_capacity is not None:
        return
    if afc.get_ascend_config().enable_prefill_mc2:
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = (
            vllm_config.compilation_config.max_cudagraph_capture_size
        )
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len

    pc = vllm_config.parallel_config
    if getattr(pc, "is_heterogeneous_tp", False):
        tp_sizes = [
            pc.get_tp_size_for_dp(i) for i in range(pc.data_parallel_size)
        ]
        align = math.lcm(*tp_sizes)
    else:
        tp_sizes = None
        align = pc.tensor_parallel_size

    num_tokens_per_tp_rank = (max_num_tokens + align - 1) // align
    if tp_sizes is not None:
        max_safe_units = min((tp_i * 512) // align for tp_i in tp_sizes)
        num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, max_safe_units)
    else:
        num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, 512)
    afc._mc2_tokens_capacity = num_tokens_per_tp_rank * align


def _patched_select_a3_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    quant_type: str | None,
    mc2_tokens_capacity: int,
    enable_fused_mc2: int,
):
    import vllm_ascend.ascend_forward_context as afc

    dispatch_ffn_combine_enable = get_ep_group().world_size <= 32
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = get_ep_group().world_size
    if num_experts % ep_world_size != 0:
        # 256 experts cannot be split evenly across 15 heterogeneous-TP
        # ranks; neither MC2 nor A3/ALLTOALL can shard them evenly.
        return afc.MoECommType.ALLGATHER

    if num_tokens <= mc2_tokens_capacity:
        fused_decode_enable = enable_fused_mc2
        if enable_fused_mc2 == 1:
            fused_decode_enable = (
                enable_fused_mc2 and dispatch_ffn_combine_enable
            )
        elif enable_fused_mc2 == 2:
            fused_decode_enable = (
                enable_fused_mc2
                and afc.speculative_enable_dispatch_gmm_combine_decode(
                    vllm_config
                )
                and quant_type == "w8a8_dynamic"
            )
        return (
            afc.MoECommType.FUSED_MC2
            if fused_decode_enable else afc.MoECommType.MC2
        )

    fused_prefill_enable = enable_fused_mc2
    if enable_fused_mc2 == 1:
        fused_prefill_enable = (
            enable_fused_mc2 and dispatch_ffn_combine_enable
        )
    elif enable_fused_mc2 == 2:
        fused_prefill_enable = False
    return (
        afc.MoECommType.FUSED_MC2
        if fused_prefill_enable else afc.MoECommType.ALLTOALL
    )


def apply_hetero_forward_context_patch():
    global _PATCHED
    if _PATCHED:
        return
    import sys

    import vllm_ascend.ascend_forward_context as afc

    # ``vllm_ascend.worker.model_runner_v1`` (and several spec-decode
    # proposer modules) do ``from vllm_ascend.ascend_forward_context import
    # set_ascend_forward_context`` at module import time.  The zero_interrupt
    # plugin imports ``NPUWorker`` (and therefore ``model_runner_v1``) before
    # this patch runs, so those modules keep a reference to the ORIGINAL
    # function object.  Replacing only the attribute on ``afc`` would leave
    # those aliases stale and the hetero per-DP token metadata
    # (``per_dp_tp_sizes`` / ``per_dp_padded_lengths``) would never be
    # published to ``_EXTRA_CTX`` during profile_run / forward.
    _orig_set_forward_context = afc.set_ascend_forward_context
    _orig_set_mc2_tokens_capacity = afc.set_mc2_tokens_capacity

    afc.set_ascend_forward_context = _patched_set_ascend_forward_context
    afc.set_mc2_tokens_capacity = _patched_set_mc2_tokens_capacity
    afc._select_a3_moe_comm_method = _patched_select_a3_moe_comm_method

    # Refresh every already-imported module that captured the original
    # function objects as module-level names.
    _rebindings = (
        ("set_ascend_forward_context",
         _orig_set_forward_context,
         _patched_set_ascend_forward_context),
        ("set_mc2_tokens_capacity",
         _orig_set_mc2_tokens_capacity,
         _patched_set_mc2_tokens_capacity),
    )
    for module in list(sys.modules.values()):
        if module is None or module is afc:
            continue
        for attr, original, replacement in _rebindings:
            try:
                if getattr(module, attr, None) is original:
                    setattr(module, attr, replacement)
            except Exception:  # noqa: BLE001 - best-effort alias refresh
                continue

    # 允许 _EXTRA_CTX 代理读写异构 TP 的 per-DP token 布局字段。
    _extra_attrs = list(afc._ExtraForwardContextProxy.extra_attrs)
    for _name in ("per_dp_padded_lengths", "per_dp_tp_sizes"):
        if _name not in _extra_attrs:
            _extra_attrs.append(_name)
    afc._ExtraForwardContextProxy.extra_attrs = tuple(_extra_attrs)
    _PATCHED = True
