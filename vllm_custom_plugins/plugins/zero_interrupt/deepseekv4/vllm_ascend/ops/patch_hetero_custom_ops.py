#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Heterogeneous-TP implementation for the registered Ascend MoE custom ops.

``vllm_ascend.ops.register_custom_ops`` registers Python callables with
``direct_register_custom_op`` at import time.  Re-registering an op would fail
with a duplicate-schema error, so this patch installs the heterogeneous
implementation by replacing the ``__code__`` of the already-registered Python
function objects and injecting the helpers into that module's globals.  When
``_EXTRA_CTX.per_dp_tp_sizes`` is None the functions keep the homogeneous
behaviour; the hetero branches only activate for the DeepSeek-V4
DP4TP(3,4,4,4) restart topology.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context

_PATCHED = False


def _hetero_chunks_and_uniform_rank(per_dp, tp_sizes, stream_padded):
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    padded_num = int(getattr(_EXTRA_CTX, "padded_num_tokens", 0) or 0)
    if padded_num <= 0:
        padded_num = max(per_dp)
    chunks = []
    for i in range(len(tp_sizes)):
        stream_len = padded_num if stream_padded else per_dp[i]
        chunks.append(stream_len // tp_sizes[i])
    return chunks, max(chunks)


def _hetero_fake_output_sizes(is_ep_comm):
    from vllm.distributed import get_ep_group
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    if not is_ep_comm:
        return None, None
    per_dp = getattr(_EXTRA_CTX, "per_dp_padded_lengths", None)
    tp_sizes = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None)
    if per_dp is None or tp_sizes is None:
        return None, None
    try:
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return None, None
        actual_tokens = [
            int(v) for v in dp_metadata.num_tokens_across_dp_cpu.tolist()
        ]
    except (AssertionError, AttributeError, IndexError, TypeError, ValueError):
        return None, None

    stream_padded = bool(
        getattr(_EXTRA_CTX, "flash_comm_v1_enabled", False)
        or getattr(_EXTRA_CTX, "flashcomm_v2_enabled", False)
    )
    chunks, uniform_rank = _hetero_chunks_and_uniform_rank(
        per_dp, tp_sizes, stream_padded
    )
    from vllm_ascend.utils import enable_sp_by_pass

    try:
        if enable_sp_by_pass():
            gather_len = get_ep_group().world_size * uniform_rank
            reduce_len = uniform_rank
        else:
            gather_len = sum(actual_tokens)
            ep_rank = get_ep_group().rank_in_group
            i = 0
            while ep_rank >= tp_sizes[i]:
                ep_rank -= tp_sizes[i]
                i += 1
            reduce_len = chunks[i]
    except (AssertionError, AttributeError, IndexError, TypeError, ValueError):
        return None, None
    return gather_len, reduce_len


def _maybe_all_gather_and_maybe_unpad_impl(x, label, is_ep_comm=False):
    from vllm.distributed import get_ep_group
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.utils import enable_sp_by_pass

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    is_hetero = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None
    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (
        enable_sp_by_pass() and is_ep_comm
    )
    if (flash_comm_v1_enabled or (is_hetero and is_ep_comm)) and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
        else:
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            per_dp = getattr(_EXTRA_CTX, "per_dp_padded_lengths", None)
            tp_sizes = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None)
            if per_dp is not None and tp_sizes is not None:
                stream_padded = bool(
                    getattr(_EXTRA_CTX, "flash_comm_v1_enabled", False)
                    or getattr(_EXTRA_CTX, "flashcomm_v2_enabled", False)
                )
                chunks, uniform_rank = _hetero_chunks_and_uniform_rank(
                    per_dp, tp_sizes, stream_padded
                )
                if x.shape[0] < uniform_rank:
                    pad = uniform_rank - x.shape[0]
                    x = F.pad(x, (0, 0) * (x.dim() - 1) + (0, pad))
                x = get_ep_group().all_gather(x, 0)
                if enable_sp_by_pass():
                    return x
                result = torch.empty(
                    (num_tokens_across_dp_cpu.sum(), *x.shape[1:]),
                    device=x.device,
                    dtype=x.dtype,
                )
                result_offset = 0
                x_offset = 0
                for i in range(len(tp_sizes)):
                    actual_i = int(num_tokens_across_dp_cpu[i].item())
                    chunk = chunks[i]
                    for r in range(tp_sizes[i]):
                        start = r * chunk
                        if start < actual_i:
                            n = min(chunk, actual_i - start)
                            result[result_offset:result_offset + n] = (
                                x[x_offset:x_offset + n]
                            )
                            result_offset += n
                        x_offset += uniform_rank
                return result
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():
                return x
            result = torch.empty(
                (num_tokens_across_dp_cpu.sum(), *x.shape[1:]),
                device=x.device,
                dtype=x.dtype,
            )
            from vllm.distributed import get_dp_group

            dp_size = get_dp_group().world_size
            # Homogeneous path keeps the original semantic.
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset:offset + num_tokens_dp] = (
                    x[idx, :num_tokens_dp]
                )
                offset += num_tokens_dp
            x = result
    return x


def _maybe_pad_and_reduce_impl(x, is_ep_comm=False):
    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.utils import enable_sp_by_pass, is_vl_model

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    is_hetero = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None
    flash_comm_v1_enabled = getattr(
        forward_context, "flash_comm_v1_enabled", False
    ) or (enable_sp_by_pass() and is_ep_comm)

    if (
        not flash_comm_v1_enabled
        and not (is_hetero and is_ep_comm)
    ) or (
        forward_context.is_draft_model and is_vl_model() and not is_ep_comm
    ):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        if x.shape[0] % tp_size != 0:
            extra = tp_size - (x.shape[0] % tp_size)
            x = F.pad(x, (0, 0, 0, extra))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        num_tokens_across_dp_cpu = (
            get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        )
        per_dp = getattr(_EXTRA_CTX, "per_dp_padded_lengths", None)
        tp_sizes = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None)
        if per_dp is not None and tp_sizes is not None:
            if enable_sp_by_pass():
                return get_ep_group().reduce_scatter(
                    x.view(-1, *x.shape[1:]), 0
                )
            stream_padded = bool(
                getattr(_EXTRA_CTX, "flash_comm_v1_enabled", False)
                or getattr(_EXTRA_CTX, "flashcomm_v2_enabled", False)
            )
            chunks, uniform_rank = _hetero_chunks_and_uniform_rank(
                per_dp, tp_sizes, stream_padded
            )
            ep_world_size = get_ep_group().world_size
            padded_x = torch.empty(
                (ep_world_size * uniform_rank, *x.shape[1:]),
                device=x.device,
                dtype=x.dtype,
            )
            x_offset = 0
            padded_offset = 0
            for i in range(len(tp_sizes)):
                actual_i = int(num_tokens_across_dp_cpu[i].item())
                chunk = chunks[i]
                for r in range(tp_sizes[i]):
                    start = r * chunk
                    if start < actual_i:
                        n = min(chunk, actual_i - start)
                        padded_x[padded_offset:padded_offset + n] = (
                            x[x_offset:x_offset + n]
                        )
                        x_offset += n
                    padded_offset += uniform_rank
            x = get_ep_group().reduce_scatter(padded_x, 0)
            ep_rank = get_ep_group().rank_in_group
            i = 0
            while ep_rank >= tp_sizes[i]:
                ep_rank -= tp_sizes[i]
                i += 1
            local_dp, local_tp = i, ep_rank
            chunk_local = chunks[local_dp]
            actual_local = int(num_tokens_across_dp_cpu[local_dp].item())
            n_local = max(
                min(chunk_local, actual_local - local_tp * chunk_local), 0
            )
            x = x[:chunk_local]
            if n_local < chunk_local:
                x[n_local:] = 0
            return x
        if enable_sp_by_pass():
            return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        from vllm.distributed import get_dp_group

        dp_size = get_dp_group().world_size
        padded_x = torch.empty(
            (dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset:offset + num_tokens_dp]
            offset += num_tokens_dp
        padded_x = padded_x.view(-1, *x.shape[1:])
        return get_ep_group().reduce_scatter(padded_x, 0)


def _maybe_all_gather_and_maybe_unpad_fake(x, label, is_ep_comm=False):
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    is_hetero = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None
    if (
        _EXTRA_CTX.flash_comm_v1_enabled or (is_hetero and is_ep_comm)
    ) and label:
        output_len = None
        if is_hetero and is_ep_comm:
            output_len, _ = _hetero_fake_output_sizes(is_ep_comm)
        if output_len is None:
            output_len = x.shape[0] * get_tensor_model_parallel_world_size()
        return torch.empty(
            (output_len, *x.shape[1:]), device=x.device, dtype=x.dtype
        )
    return x


def _maybe_pad_and_reduce_fake(x, is_ep_comm=False):
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.utils import enable_sp_by_pass

    is_hetero = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None
    if (
        _EXTRA_CTX.flash_comm_v1_enabled
        or enable_sp_by_pass()
        or (is_hetero and is_ep_comm)
    ):
        output_len = None
        if is_hetero and is_ep_comm:
            _, output_len = _hetero_fake_output_sizes(is_ep_comm)
        if output_len is None:
            output_len = x.shape[0] // get_tensor_model_parallel_world_size()
        return torch.empty(
            (output_len, *x.shape[1:]), device=x.device, dtype=x.dtype
        )
    return x


def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states):
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

    moe_comm_type = _EXTRA_CTX.moe_comm_type
    is_hetero = getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None
    if (
        moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        }
        or _EXTRA_CTX.flash_comm_v1_enabled
        or is_hetero
    ):
        return final_hidden_states
    return tensor_model_parallel_all_reduce(final_hidden_states)


def apply_hetero_custom_ops_patch():
    global _PATCHED
    if _PATCHED:
        return
    import vllm_ascend.ops.register_custom_ops as mod

    replacements = {
        "_maybe_all_gather_and_maybe_unpad_impl":
            _maybe_all_gather_and_maybe_unpad_impl,
        "_maybe_pad_and_reduce_impl": _maybe_pad_and_reduce_impl,
        "_maybe_all_gather_and_maybe_unpad_fake":
            _maybe_all_gather_and_maybe_unpad_fake,
        "_maybe_pad_and_reduce_fake": _maybe_pad_and_reduce_fake,
        "_maybe_all_reduce_tensor_model_parallel_impl":
            _maybe_all_reduce_tensor_model_parallel_impl,
    }
    # Inject helpers referenced by the replaced code objects.
    mod.__dict__["_hetero_chunks_and_uniform_rank"] = (
        _hetero_chunks_and_uniform_rank
    )
    mod.__dict__["_hetero_fake_output_sizes"] = _hetero_fake_output_sizes

    for name, new_func in replacements.items():
        target = getattr(mod, name)
        target.__code__ = new_func.__code__
    _PATCHED = True
