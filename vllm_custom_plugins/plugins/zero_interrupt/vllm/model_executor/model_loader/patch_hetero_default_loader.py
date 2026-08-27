#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""DefaultModelLoader EP weight-filter patch for heterogeneous TP."""

_PATCHED = False


def _patched_init_ep_weight_filter(self, model_config):
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader.ep_weight_filter import (
        compute_local_expert_ids,
    )

    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config

    if not (
        model_config.is_moe
        and parallel_config.enable_expert_parallel
        and parallel_config.enable_ep_weight_filter
    ):
        return
    if parallel_config.enable_eplb:
        return

    num_experts = model_config.get_num_experts()
    if num_experts <= 0:
        return

    from vllm.distributed import (
        get_dp_group,
        get_pcp_group,
        get_tensor_model_parallel_rank,
    )

    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0

    if getattr(parallel_config, "is_heterogeneous_tp", False):
        dp_rank = parallel_config.data_parallel_rank
        tp_sizes = [
            parallel_config.get_tp_size_for_dp(i) for i in range(dp_size)
        ]
        ep_size = sum(tp_sizes) * pcp_size
        ep_rank = (
            sum(tp_sizes[i] * pcp_size for i in range(dp_rank))
            + pcp_rank * tp_sizes[dp_rank]
            + tp_rank
        )
    else:
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

    self.local_expert_ids = compute_local_expert_ids(
        num_experts,
        ep_size,
        ep_rank,
        placement=parallel_config.expert_placement_strategy,
    )


def apply_hetero_default_loader_patch():
    global _PATCHED
    if _PATCHED:
        return
    import vllm.model_executor.model_loader.default_loader as mod

    mod.DefaultModelLoader._init_ep_weight_filter = (
        _patched_init_ep_weight_filter
    )
    _PATCHED = True
