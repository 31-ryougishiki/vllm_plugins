#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""MTP proposer patch for heterogeneous TP.

Under heterogeneous TP the speculative config is built once from the DP0
snapshot (draft_tensor_parallel_size == DP0 tp_size, e.g. 3), while
EngineCore has already rewritten the target parallel_config to the current
DP rank's tp_size (e.g. 4).  Do not patch the global TP group to a singleton
for the MTP draft model in that case; the MTP weights are sharded with the
target model's per-DP TP group.
"""

from __future__ import annotations

from contextlib import nullcontext

_PATCHED = False
_ORIG_INIT = None


def _patched_proposer_init(
    self,
    vllm_config,
    device,
    pass_hidden_states_to_model,
    runner=None,
):
    _ORIG_INIT(
        self,
        vllm_config,
        device,
        pass_hidden_states_to_model,
        runner=runner,
    )
    pc = vllm_config.parallel_config
    if getattr(pc, "is_heterogeneous_tp", False):
        draft_tp_size = getattr(
            self.speculative_config, "draft_tensor_parallel_size", 1
        )
        if draft_tp_size != 1:
            # Reuse the target model's per-DP TP group.
            self.tp_group_context = nullcontext()


def apply_hetero_spec_decode_patch():
    global _PATCHED, _ORIG_INIT
    if _PATCHED:
        return
    import vllm_ascend.spec_decode.llm_base_proposer as mod

    _ORIG_INIT = mod.AscendSpecDecodeBaseProposer.__init__
    mod.AscendSpecDecodeBaseProposer.__init__ = _patched_proposer_init
    _PATCHED = True
