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
    if is_hetero:
        device_str = "npu" if self.ascend_config.dp_allreduce_on_npu else "cpu"
        group = (
            get_ep_group().device_group
            if device_str == "npu" else get_ep_group().cpu_group
        )
    else:
        device_str, group = (
            ("npu", get_dp_group().device_group)
            if self.ascend_config.dp_allreduce_on_npu
            else ("cpu", get_dp_group().cpu_group)
        )
    packed_tensor = torch.zeros(
        2, self.dp_size, device=device_str, dtype=torch.int32
    )
    packed_tensor[0][self.dp_rank] = num_tokens
    packed_tensor[1][self.dp_rank] = cudagraph_mode.value
    dist.all_reduce(packed_tensor, group=group)
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

    if allow_dp_padding or is_draft_model:
        num_tokens_after_padding = torch.tensor(
            [max_tokens_across_dp] * self.dp_size,
            device="cpu",
            dtype=torch.int32,
        )
    else:
        num_tokens_after_padding = num_tokens_across_dp.cpu()

    return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode


def _patched_profile_run(self):
    origin_max_num_tokens = self.max_num_tokens
    if self.pcp_size > 1:
        self.max_num_tokens = (
            math.ceil(self.max_num_tokens / (self.pcp_size * 2)) * 2
        )
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


_ORIGINAL_PROFILE_RUN = None


def apply_hetero_model_runner_patch():
    global _PATCHED, _ORIGINAL_PROFILE_RUN
    if _PATCHED:
        return
    import vllm_ascend.worker.model_runner_v1 as mod

    _ORIGINAL_PROFILE_RUN = mod.NPUModelRunner.profile_run
    mod.NPUModelRunner._sync_metadata_across_dp.__code__ = (
        _patched_sync_metadata_across_dp.__code__
    )
    mod.__dict__["_ORIGINAL_PROFILE_RUN"] = _ORIGINAL_PROFILE_RUN
    mod.NPUModelRunner.profile_run = _patched_profile_run
    _PATCHED = True
