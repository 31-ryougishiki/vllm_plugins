# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asymmetric-TP support for the v2 ``BasevLLMParameter`` weight loaders.

The plugin's ``ColumnParallelLinearAsymmetric`` /
``MergedColumnParallelLinearAsymmetric`` / ``RowParallelLinearAsymmetric``
allocate parameters with per-rank asymmetric shapes, but the stock v0.23
``vllm.model_executor.parameter`` loaders always narrow the checkpoint with
``tp_rank * shard_size``.  For DP4TP(3,4,4,4) that offsets rank 2 by two
uniform chunks instead of the cumulative ratio offset (2,1,1) and loads wrong
weight rows.

These wrappers use ``get_current_tp_sharding_ratios()`` (installed by
``patch_hetero_utils.py``) and fall back to the untouched originals outside
heterogeneous TP.
"""

from __future__ import annotations

from vllm_custom_plugins.plugins.zero_interrupt.vllm.distributed.patch_hetero_utils import (
    get_current_tp_sharding_ratios,
    get_tp_partition_offset,
)

_PATCHED = False
_ORIG_LOAD_COLUMN = None
_ORIG_LOAD_MERGED_COLUMN = None
_ORIG_LOAD_ROW = None


def _patched_load_column_parallel_weight(self, loaded_weight):
    shard_size = self.data.shape[self.output_dim]
    ratios = get_current_tp_sharding_ratios()
    if ratios is not None:
        start_idx = get_tp_partition_offset(
            total_size=loaded_weight.shape[self.output_dim],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_sharding_ratios=ratios,
        )
        loaded_weight = loaded_weight.narrow(
            self.output_dim, start_idx, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)
        return
    return _ORIG_LOAD_COLUMN(self, loaded_weight)


def _patched_load_merged_column_weight(self, loaded_weight, **kwargs):
    ratios = get_current_tp_sharding_ratios()
    if ratios is None:
        return _ORIG_LOAD_MERGED_COLUMN(self, loaded_weight, **kwargs)

    shard_offset = kwargs["shard_offset"]
    shard_size = kwargs["shard_size"]

    import vllm.model_executor.parameter as mod

    # Preserve the stock packed-parameter adjustment before narrowing.
    if (
        isinstance(
            self, (mod.PackedColumnParameter, mod.PackedvLLMParameter)
        )
        and self.packed_dim == self.output_dim
    ):
        shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
            shard_offset=shard_offset, shard_size=shard_size
        )

    param_data = self.data.narrow(
        self.output_dim, shard_offset, shard_size
    )
    start_idx = get_tp_partition_offset(
        total_size=loaded_weight.shape[self.output_dim],
        tp_rank=self.tp_rank,
        tp_size=self.tp_size,
        tp_sharding_ratios=ratios,
    )
    loaded_weight = loaded_weight.narrow(
        self.output_dim, start_idx, shard_size
    )
    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)


def _patched_load_row_parallel_weight(self, loaded_weight):
    shard_size = self.data.shape[self.input_dim]
    ratios = get_current_tp_sharding_ratios()
    if ratios is not None:
        start_idx = get_tp_partition_offset(
            total_size=loaded_weight.shape[self.input_dim],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_sharding_ratios=ratios,
        )
        loaded_weight = loaded_weight.narrow(
            self.input_dim, start_idx, shard_size
        )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)
        return
    return _ORIG_LOAD_ROW(self, loaded_weight)


def apply_hetero_parameter_patch():
    """Patch v2 parameter loaders for asymmetric TP."""
    global _PATCHED
    global _ORIG_LOAD_COLUMN, _ORIG_LOAD_MERGED_COLUMN, _ORIG_LOAD_ROW
    if _PATCHED:
        return

    import vllm.model_executor.parameter as mod

    _ORIG_LOAD_COLUMN = mod._ColumnvLLMParameter.load_column_parallel_weight
    _ORIG_LOAD_MERGED_COLUMN = (
        mod._ColumnvLLMParameter.load_merged_column_weight
    )
    _ORIG_LOAD_ROW = mod.RowvLLMParameter.load_row_parallel_weight

    mod._ColumnvLLMParameter.load_column_parallel_weight = (
        _patched_load_column_parallel_weight
    )
    mod._ColumnvLLMParameter.load_merged_column_weight = (
        _patched_load_merged_column_weight
    )
    mod.RowvLLMParameter.load_row_parallel_weight = (
        _patched_load_row_parallel_weight
    )
    _PATCHED = True
