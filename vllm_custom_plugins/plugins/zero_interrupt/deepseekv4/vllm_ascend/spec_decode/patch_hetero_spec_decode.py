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
_ORIG_INIT_ATTN_BACKEND = None


def _mark_dsa_cp_draft_builders(self) -> None:
    """Mark DSA-CP metadata builders that belong to the draft model.

    Their forward runs with ``flash_comm_v1_enabled=False`` (only real
    tokens), so the hetero DSA-CP patch must not pad draft metadata to
    ``lcm(all per-DP tp sizes)``.  Without this marker the draft builder
    pads e.g. 1 decode token to 12 rows while the hidden-state buffer has
    1 row, and the RoPE / output-write shapes diverge.
    """
    if not getattr(self.vllm_config.parallel_config, "is_heterogeneous_tp", False):
        return
    try:
        from vllm_ascend.attention.context_parallel.dsa_cp import (
            AscendDSACPMetadataBuilder,
        )

        for attn_group in getattr(self, "draft_attn_groups", []) or []:
            for builder in getattr(attn_group, "metadata_builders", []) or []:
                if isinstance(builder, AscendDSACPMetadataBuilder):
                    builder._is_dsa_cp_draft_builder = True
    except Exception:
        # Marker is an optimization for DSA-CP metadata padding; other
        # draft backends (or unit-test mocks without draft_attn_groups)
        # must not fail proposer construction.
        pass


def _patched_initialize_attn_backend(
    self,
    kv_cache_config,
    kernel_block_sizes=None,
):
    _ORIG_INIT_ATTN_BACKEND(self, kv_cache_config, kernel_block_sizes)
    # ``draft_attn_groups`` is created here (initialize_attn_backend), not in
    # ``__init__``, so the draft-builder marker must be applied after this call.
    _mark_dsa_cp_draft_builders(self)


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
    global _PATCHED, _ORIG_INIT, _ORIG_INIT_ATTN_BACKEND
    if _PATCHED:
        return
    import vllm_ascend.spec_decode.llm_base_proposer as mod

    _ORIG_INIT = mod.AscendSpecDecodeBaseProposer.__init__
    _ORIG_INIT_ATTN_BACKEND = (
        mod.AscendSpecDecodeBaseProposer.initialize_attn_backend
    )
    mod.AscendSpecDecodeBaseProposer.__init__ = _patched_proposer_init
    mod.AscendSpecDecodeBaseProposer.initialize_attn_backend = (
        _patched_initialize_attn_backend
    )
    _PATCHED = True
