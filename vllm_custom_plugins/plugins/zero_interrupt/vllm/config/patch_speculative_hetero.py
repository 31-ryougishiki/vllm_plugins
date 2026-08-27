#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""SpeculativeConfig draft ParallelConfig patch for heterogeneous TP."""

_PATCHED = False


def _patched_create_draft_parallel_config(
    target_parallel_config,
    speculative_draft_tensor_parallel_size,
):
    from vllm.config import ParallelConfig

    kwargs = dict(
        pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
        tensor_parallel_size=speculative_draft_tensor_parallel_size,
        heterogeneous_dp_config=target_parallel_config.heterogeneous_dp_config,
        data_parallel_size=target_parallel_config.data_parallel_size,
        data_parallel_rank=target_parallel_config.data_parallel_rank,
        enable_expert_parallel=target_parallel_config.enable_expert_parallel,
        distributed_executor_backend=(
            target_parallel_config.distributed_executor_backend
        ),
        max_parallel_loading_workers=(
            target_parallel_config.max_parallel_loading_workers
        ),
        disable_custom_all_reduce=(
            target_parallel_config.disable_custom_all_reduce
        ),
        ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
        placement_group=target_parallel_config.placement_group,
    )
    return ParallelConfig(**kwargs)


def apply_speculative_hetero_patch():
    global _PATCHED
    if _PATCHED:
        return
    import vllm.config.speculative as mod

    mod.SpeculativeConfig.create_draft_parallel_config = staticmethod(
        _patched_create_draft_parallel_config
    )
    _PATCHED = True
