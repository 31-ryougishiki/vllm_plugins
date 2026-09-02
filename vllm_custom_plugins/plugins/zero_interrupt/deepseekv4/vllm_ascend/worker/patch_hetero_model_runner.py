#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""NPUModelRunner patches for heterogeneous TP.

Port of hetero_cp ``vllm_ascend/worker/model_runner_v1.py`` changes:
- DP metadata sync must use the EP group because orphaned TP ranks have
  singleton DP groups; each DP slot is divided by that DP rank's tp_size.
- profile_run max_num_tokens is aligned to lcm(all per-DP tp sizes).
"""

from __future__ import annotations

import math

import torch

from vllm.config import CUDAGraphMode

_PATCHED = False
_SAMPLE_DUMP_COUNT = 0
_SAMPLE_DUMP_CAP = 2048
_DP_META_DUMP_COUNT = 0
_DP_META_DUMP_CAP = 2048
_IN_DUMMY_RUN = False


def _patched_sync_metadata_across_dp(
    self,
    num_tokens: int,
    is_draft_model: bool = False,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    allow_dp_padding: bool = False,
):
    from vllm.distributed import get_dp_group, get_ep_group
    import torch.distributed as dist
    from vllm.config import CUDAGraphMode as _CUDAGraphMode
    from vllm.logger import init_logger as _init_logger

    _diag_packed_tokens = None
    _log_dp_meta_once = False

    if self.dp_size == 1:
        return num_tokens, None, cudagraph_mode

    import vllm_ascend.utils as u

    if u.should_skip_allreduce_across_dp_group(
        self.vllm_config, is_draft_model
    ):
        num_tokens_after_padding = torch.tensor(
            [num_tokens] * self.dp_size, device="cpu", dtype=torch.int32
        )
        return num_tokens, num_tokens_after_padding, cudagraph_mode

    is_hetero = bool(
        getattr(
            self.vllm_config.parallel_config, "is_heterogeneous_tp", False
        )
    )
    # Older vllm-ascend wheels predating #10046 have no
    # AscendConfig.dp_allreduce_on_npu attribute, so never read it directly.
    dp_allreduce_on_npu = bool(
        getattr(self.ascend_config, "dp_allreduce_on_npu", False)
    )
    # NOTE: this function body is injected via __code__ replacement, so it
    # executes with the globals of vllm_ascend.worker.model_runner_v1.
    # Module-level helpers from this patch file are NOT visible here; keep
    # every new dependency local to this function.
    indivisible_on_a3 = False
    try:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() == AscendDeviceType.A3:
            num_experts = self.vllm_config.model_config.get_num_experts()
            indivisible_on_a3 = (
                isinstance(num_experts, int)
                and num_experts > 0
                and num_experts % self.dp_size != 0
            )
    except Exception:
        indivisible_on_a3 = False

    if is_hetero:
        group = (
            get_ep_group().device_group
            if dp_allreduce_on_npu else get_ep_group().cpu_group
        )
        device_str = "npu" if dp_allreduce_on_npu else "cpu"
    else:
        use_npu = dp_allreduce_on_npu or indivisible_on_a3
        group = (
            get_dp_group().device_group
            if use_npu else get_dp_group().cpu_group
        )
        device_str = "npu" if use_npu else "cpu"
        if use_npu and not dp_allreduce_on_npu:
            _init_logger(__name__).warning_once(
                "A3 remainder fallback: DP metadata all_reduce uses the "
                "NPU device group (num_experts=%s, dp_size=%s).",
                self.vllm_config.model_config.get_num_experts(),
                self.dp_size,
            )
    packed_tensor = torch.zeros(
        2, self.dp_size, device=device_str, dtype=torch.int32
    )
    packed_tensor[0][self.dp_rank] = num_tokens
    packed_tensor[1][self.dp_rank] = cudagraph_mode.value
    dist.all_reduce(packed_tensor, group=group)
    if not is_hetero and not getattr(self, "_dp_meta_diag_logged", False):
        # One-time NPU -> CPU sync for triage only; the instance flag keeps
        # every later call on the cheap path.
        _diag_packed_tokens = packed_tensor[0].tolist()
        _log_dp_meta_once = True
        self._dp_meta_diag_logged = True
    if is_hetero:
        tp_sizes = [
            self.vllm_config.parallel_config.get_tp_size_for_dp(i)
            for i in range(self.dp_size)
        ]
        for i in range(self.dp_size):
            packed_tensor[0, i] //= tp_sizes[i]
            packed_tensor[1, i] //= tp_sizes[i]
    if device_str == "npu":
        packed_tensor = packed_tensor.cpu()

    num_tokens_across_dp = packed_tensor[0, :]
    max_tokens_across_dp = int(num_tokens_across_dp.max().item())
    import vllm_ascend.worker.model_runner_v1 as mrm

    synced_cudagraph_mode = _CUDAGraphMode(
        mrm._post_process_cudagraph_mode(packed_tensor)
    )

    # The ALLGATHER dummy-zero fix in PrepareAndFinalizeWithAllGather.prepare
    # contains a Python conditional.  During FULL-graph capture that branch is
    # evaluated once with is_dummy_run=True and the zeroing is baked into the
    # captured graph; replaying that graph for a REAL request would zero real
    # MoE inputs.  For the A3 indivisible-expert fallback, keep dummy/graph
    # capture runs on FULL graphs but force REAL forwards to eager mode.
    try:
        from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.worker import (  # noqa: E501
            patch_hetero_model_runner as _mrm_patch,
        )

        _is_dummy = bool(getattr(_mrm_patch, "_IN_DUMMY_RUN", False))
    except Exception:  # noqa: BLE001 - patch module not loaded yet
        _is_dummy = False
    if indivisible_on_a3 and not is_hetero and not _is_dummy:
        synced_cudagraph_mode = _CUDAGraphMode.NONE
        _init_logger(__name__).warning_once(
            "A3 remainder fallback: forcing real forward to "
            "cudagraph_mode=NONE (dummy-zero ALLGATHER graph is not "
            "safe to replay for real requests)."
        )

    if allow_dp_padding or is_draft_model:
        num_tokens_after_padding = torch.tensor(
            [max_tokens_across_dp] * self.dp_size,
            device="cpu",
            dtype=torch.int32,
        )
    else:
        num_tokens_after_padding = num_tokens_across_dp.cpu()

    if _log_dp_meta_once:
        _init_logger(__name__).warning_once(
            "[hetero-moe diag] DP metadata sync: num_tokens=%s "
            "allow_dp_padding=%s is_draft_model=%s cudagraph_mode=%s "
            "packed_tokens=%s max_tokens_across_dp=%s "
            "num_tokens_after_padding=%s",
            num_tokens,
            allow_dp_padding,
            is_draft_model,
            cudagraph_mode,
            str(_diag_packed_tokens),
            max_tokens_across_dp,
            str(num_tokens_after_padding.tolist()),
        )

    import os as _os
    if _os.getenv("VLLM_ITS_DUMP_DP_META_EVERY", "0") == "1":
        from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.worker import (  # noqa: E501
            patch_hetero_model_runner as _diag_mod,
        )

        _diag_mod._dp_meta_dump_every(
            _init_logger(__name__),
            num_tokens=num_tokens,
            allow_dp_padding=allow_dp_padding,
            is_draft_model=is_draft_model,
            cudagraph_mode=cudagraph_mode,
            packed_tokens=packed_tensor,
            max_tokens_across_dp=max_tokens_across_dp,
            num_tokens_after_padding=num_tokens_after_padding,
            device_str=device_str,
        )

    return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode


def _patched_profile_run(self):
    origin_max_num_tokens = self.max_num_tokens
    if self.pcp_size > 1:
        # The saved original profile_run already applies the PCP token
        # adjustment once; do NOT pre-apply it here or max_num_tokens gets
        # rounded down twice (e.g. 100 -> 50 -> 26 instead of 100 -> 50).
        pass
    elif bool(
        getattr(self.vllm_config.parallel_config, "is_heterogeneous_tp", False)
    ):
        pc = self.vllm_config.parallel_config
        align = math.lcm(
            *[pc.get_tp_size_for_dp(i) for i in range(pc.data_parallel_size)]
        )
        self.max_num_tokens = (self.max_num_tokens // align) * align

    # Call the original profile_run without recursion.  The original is a
    # function object saved in the class attribute by apply_*.
    try:
        _ORIGINAL_PROFILE_RUN(self)
    finally:
        self.max_num_tokens = origin_max_num_tokens


def _patched_sample_tokens(self, grammar_output):
    """Optional per-step sampled-token dump for DP15-vs-DP16 triage.

    Gated by VLLM_ITS_DUMP_SAMPLED_TOKENS=1.  The original ``sample_tokens``
    is called unchanged; this wrapper only reads the returned sampled token
    ids / draft ids afterwards so three identical requests can be compared
    step by step without touching the sampler math.
    """
    import os as _os

    _result = _ORIGINAL_SAMPLE_TOKENS(self, grammar_output)
    if _os.getenv("VLLM_ITS_DUMP_SAMPLED_TOKENS", "0") != "1":
        return _result

    global _SAMPLE_DUMP_COUNT
    if _SAMPLE_DUMP_COUNT >= _SAMPLE_DUMP_CAP:
        return _result
    _SAMPLE_DUMP_COUNT += 1

    # Debug-only: the async sampler output is copied to CPU on a separate
    # stream.  Wait for that copy before reading the values, otherwise the
    # dump shows the previous/uninitialized tensor contents.
    try:
        _event = getattr(_result, "async_copy_ready_event", None)
        if _event is not None:
            _event.synchronize()
    except Exception:  # noqa: BLE001 - logging must not break sampling
        pass

    from vllm.logger import init_logger as _init_logger

    _logger = _init_logger(__name__)

    def _to_cpu_list(value, limit):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            _flat = []
            for _v in value:
                if isinstance(_v, (list, tuple)):
                    _flat.extend(_v)
                else:
                    _flat.append(_v)
            value = _flat
        return list(value)[:limit]

    _sampled = _draft = None
    try:
        _cpu_ids = getattr(_result, "sampled_token_ids_cpu", None)
        _sampled = _to_cpu_list(
            _cpu_ids if _cpu_ids is not None else getattr(
                _result, "sampled_token_ids", None
            ),
            64,
        )
    except Exception as _exc:  # noqa: BLE001
        _sampled = f"<unavailable: {type(_exc).__name__}>"
    try:
        _draft_ids = getattr(self, "_draft_token_ids", None)
        _draft = _to_cpu_list(
            _draft_ids if hasattr(_draft_ids, "cpu") else None,
            64,
        )
    except Exception as _exc:  # noqa: BLE001
        _draft = f"<unavailable: {type(_exc).__name__}>"
    _logger.warning(
        "[hetero-moe diag] sampled tokens #%s: num_reqs=%s sampled=%s draft=%s",
        _SAMPLE_DUMP_COUNT,
        getattr(getattr(self, "input_batch", None), "num_reqs", None),
        str(_sampled),
        str(_draft),
    )
    return _result


def _dp_meta_dump_every(
    _logger,
    *,
    num_tokens,
    allow_dp_padding,
    is_draft_model,
    cudagraph_mode,
    packed_tokens,
    max_tokens_across_dp,
    num_tokens_after_padding,
    device_str,
):
    """Per-call DP metadata dump for cross-request shape triage.

    The caller is a __code__-swapped function, so this helper lives in the
    plugin module and is reached via an explicit module import.  Gated by
    VLLM_ITS_DUMP_DP_META_EVERY=1 and capped so a long decode does not
    flood dp logs.
    """
    import os as _os

    global _DP_META_DUMP_COUNT
    if (
        _os.getenv("VLLM_ITS_DUMP_DP_META_EVERY", "0") != "1"
        or _DP_META_DUMP_COUNT >= _DP_META_DUMP_CAP
    ):
        return
    _DP_META_DUMP_COUNT += 1
    _packed = packed_tokens.tolist()
    _after = num_tokens_after_padding.tolist()
    _logger.warning(
        "[hetero-moe diag] dp-meta every#%s: num_tokens=%s allow_dp_padding=%s "
        "is_draft_model=%s cudagraph_mode=%s device=%s packed=%s "
        "max_across=%s after_padding=%s",
        _DP_META_DUMP_COUNT,
        num_tokens,
        allow_dp_padding,
        is_draft_model,
        getattr(cudagraph_mode, "value", cudagraph_mode),
        device_str,
        str(_packed),
        max_tokens_across_dp,
        str(_after),
    )


_ORIGINAL_SAMPLE_TOKENS = None
_ORIGINAL_DUMMY_RUN = None


def _patched_dummy_run(self, *args, **kwargs):
    """Flag ``_dummy_run`` executions for the ALLGATHER MoE path.

    Idle DP engines keep the collective world alive by continuously running
    dummy batches.  MC2/FUSED_MC2 dispatch isolates those dummy rows per
    source rank, but the ALLGATHER fallback pads every rank to
    ``max_tokens_across_dp`` and then reduce-scatters: dummy rows from idle
    ranks are summed into the real request's rows.  The flag published here
    lets ``PrepareAndFinalizeWithAllGather.prepare`` zero the dummy
    hidden/router contribution before the DP all-gather.
    """
    global _IN_DUMMY_RUN
    _prev = _IN_DUMMY_RUN
    _IN_DUMMY_RUN = True
    try:
        return _ORIGINAL_DUMMY_RUN(self, *args, **kwargs)
    finally:
        _IN_DUMMY_RUN = _prev
_ORIGINAL_PROFILE_RUN = None


def apply_hetero_model_runner_patch():
    global _PATCHED, _ORIGINAL_PROFILE_RUN, _ORIGINAL_SAMPLE_TOKENS
    global _ORIGINAL_DUMMY_RUN
    if _PATCHED:
        return
    import vllm_ascend.worker.model_runner_v1 as mod

    _ORIGINAL_PROFILE_RUN = mod.NPUModelRunner.profile_run
    _ORIGINAL_SAMPLE_TOKENS = mod.NPUModelRunner.sample_tokens
    _ORIGINAL_DUMMY_RUN = mod.NPUModelRunner._dummy_run
    mod.NPUModelRunner._sync_metadata_across_dp.__code__ = (
        _patched_sync_metadata_across_dp.__code__
    )
    mod.__dict__["_ORIGINAL_PROFILE_RUN"] = _ORIGINAL_PROFILE_RUN
    mod.NPUModelRunner.profile_run = _patched_profile_run
    mod.NPUModelRunner.sample_tokens = _patched_sample_tokens
    mod.NPUModelRunner._dummy_run = _patched_dummy_run
    _PATCHED = True
