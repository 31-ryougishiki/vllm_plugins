#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Runtime patch for heterogeneous-TP support in vllm_ascend.ops.linear.

``vllm_ascend/ops/linear.py`` contains the Ascend custom-op linear classes
(AscendColumn/MergedColumn/RowParallelLinear).  hetero_cp modified them so
that their per-rank parameter partitions honor
``tp_asymmetric_shardings`` / ``heterogeneous_dp_config``.  This module
installs the same behaviour without replacing the source file.

The wrappers run the original initializer first (it creates weights with
uniform partitions), then rebuild the quant-method weights with the
asymmetric partition sizes when ratios are active.  This mirrors the
double-create approach already used by the plugin's vLLM asymmetric linear
classes.
"""

from __future__ import annotations

import itertools
import logging

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.model_executor.utils import set_weight_attrs

logger = logging.getLogger(__name__)

_PATCHED = False
_ORIG_COL_INIT = None
_ORIG_MERGED_INIT = None
_ORIG_ROW_INIT = None
_ORIG_COL_WEIGHT_LOADER = None
_ORIG_ROW_WEIGHT_LOADER = None
_ORIG_MERGED_WEIGHT_LOADER = None
_ORIG_MERGED_WEIGHT_LOADER_V2 = None
_ORIG_ASCEND_UNQUANT_PROCESS = None
_ORIG_VLLM_UNQUANT_PROCESS = None


def _reshape_wo_a_for_dsa(layer) -> None:
    """Reshape an unquantized DeepSeek-V4 ``wo_a`` weight to 3-D.

    ModelSlim W8A8 recipes intentionally leave ``wo_a`` unquantized, so the
    FP8 ``ds_linear`` post-load reshape never runs.  Both DSA-CP and DSA v1
    feed ``wo_a.weight`` directly into ``npu_transpose_batchmatmul`` with
    ``perm_x2=(0, 1, 2)``; the op requires a 3-D weight
    ``[n_local_groups, input_dim, o_lora_rank]`` and raises
    ``IndexError: Dimension out of range (expected [-2, 1], got 2)`` for a
    2-D parameter.
    """
    prefix = getattr(layer, "prefix", "")
    if not prefix.endswith("wo_a"):
        return
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2:
        return

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    try:
        cfg = get_current_vllm_config()
        hf_config = cfg.model_config.hf_text_config
        parallel_config = cfg.parallel_config
    except Exception:  # noqa: BLE001
        return
    o_groups = getattr(hf_config, "o_groups", 0)
    o_lora_rank = getattr(hf_config, "o_lora_rank", 0)
    if o_groups <= 0 or o_lora_rank <= 0:
        return
    tp_size = getattr(layer, "tp_size", None) or (
        get_tensor_model_parallel_world_size()
    )
    tp_rank = getattr(layer, "tp_rank", None)
    if tp_rank is None:
        tp_rank = get_tensor_model_parallel_rank()

    # Under heterogeneous TP the wo_a groups follow tp_asymmetric_shardings
    # (e.g. tp=3 with [2,1,1] -> 4/2/2 groups).  ``o_groups // tp_size`` is
    # wrong there and would make rank0's reshape check fail (expecting
    # 2*1024 rows instead of the real 4*1024), leaving wo_a 2-D for the DSA
    # batchmatmul kernels.
    ratios = getattr(layer, "_tp_sharding_ratios", None)
    if ratios is None and getattr(parallel_config, "is_heterogeneous_tp", False):
        ratios = parallel_config.get_sharding_ratios_for_dp(
            parallel_config.data_parallel_rank
        )
    if ratios:
        from vllm.distributed.utils import get_tp_partition_size

        n_local_groups = get_tp_partition_size(
            o_groups, tp_rank, tp_size, ratios
        )
    else:
        n_local_groups = o_groups // tp_size
    if n_local_groups <= 0 or weight.shape[0] != n_local_groups * o_lora_rank:
        return

    weight.data = (
        weight.data.view(n_local_groups, o_lora_rank, -1)
        .transpose(1, 2)
        .contiguous()
    )


def _patched_ascend_unquant_process_weights_after_loading(self, layer):
    _ORIG_ASCEND_UNQUANT_PROCESS(self, layer)
    _reshape_wo_a_for_dsa(layer)


def _patched_vllm_unquant_process_weights_after_loading(self, layer):
    _ORIG_VLLM_UNQUANT_PROCESS(self, layer)
    _reshape_wo_a_for_dsa(layer)


def _ratios_for(disable_tp: bool):
    from vllm.distributed.utils import get_current_tp_sharding_ratios

    return get_current_tp_sharding_ratios() if not disable_tp else None


def _ceil_divisible(value: int, divisor: int) -> int:
    """Smallest multiple of ``divisor`` that is >= ``value``."""
    return ((value + divisor - 1) // divisor) * divisor


def _rebuild_col_weights(self, output_size: int, ratios: list[int]) -> None:
    from vllm.distributed.utils import get_tp_partition_size

    output_sizes = list(getattr(self, "output_sizes", [output_size]) or [output_size])
    self.output_size_per_partition = get_tp_partition_size(
        output_size, self.tp_rank, self.tp_size, ratios
    )
    self.output_partition_sizes = [
        get_tp_partition_size(size, self.tp_rank, self.tp_size, ratios)
        for size in output_sizes
    ]

    assert self.quant_method is not None
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2
            if self.quant_method.__class__.__name__
            in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        ),
    )
    if self.bias is not None:
        self.bias = Parameter(
            torch.empty(
                self.output_size_per_partition, dtype=self.params_dtype
            )
        )
        set_weight_attrs(
            self.bias,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )

    if "wo_a" in self.prefix:
        hf_config = get_current_vllm_config().model_config.hf_text_config
        o_groups = getattr(hf_config, "o_groups", 0)
        self.n_local_groups = get_tp_partition_size(
            o_groups, self.tp_rank, self.tp_size, ratios
        )
        self.o_lora_rank = getattr(hf_config, "o_lora_rank", 0)

    if self.custom_op is not None:
        self.custom_op.update_attrs()


def _patched_col_init(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config=None,
    output_sizes: list[int] | None = None,
    prefix: str = "",
    *,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    # Preserve the original __init__'s `hasattr(self, "output_sizes")`
    # behaviour and make the Merged subclass path deterministic.
    real_output_sizes = (
        list(output_sizes) if output_sizes is not None else None
    )
    self.output_sizes = real_output_sizes or [output_size]
    ratios = _ratios_for(disable_tp)
    self._tp_sharding_ratios = ratios

    # The stock AscendColumnParallelLinear.__init__ requires
    # ``output_size % tp_size == 0``.  Under heterogeneous TP that is no
    # longer true (e.g. wo_a output 8192 with tp=3).  Call the original with
    # the smallest divisible size so it can build the layer scaffolding, then
    # restore the real output sizes and rebuild the quant weights with the
    # asymmetric partition below.
    if ratios is not None:
        import vllm_ascend.ops.linear as mod

        self.custom_op, self.tp_rank, self.tp_size = mod.get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        uniform_output_size = _ceil_divisible(output_size, self.tp_size)
        # The stock init does ``if hasattr(self, "output_sizes")`` and
        # divides ``self.output_sizes``, *not* the ``output_sizes`` argument.
        # We set the attribute above (for the Merged subclass path), so it
        # must temporarily hold the divisible scaffolding size here;
        # otherwise stock init still divides the real 8192 by tp=3 and
        # raises before the rebuild below can run.
        self.output_sizes = [uniform_output_size]
        try:
            _ORIG_COL_INIT(
                self,
                input_size=input_size,
                output_size=uniform_output_size,
                bias=bias,
                gather_output=gather_output,
                skip_bias_add=skip_bias_add,
                params_dtype=params_dtype,
                quant_config=quant_config,
                output_sizes=None,
                prefix=prefix,
                return_bias=return_bias,
                disable_tp=disable_tp,
            )
        finally:
            # Restore the real geometry even if stock init raised.
            self.output_size = output_size
            self.output_sizes = real_output_sizes or [output_size]
        _rebuild_col_weights(self, output_size, ratios)
    else:
        _ORIG_COL_INIT(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            output_sizes=output_sizes,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )


def _patched_merged_init(
    self,
    input_size: int,
    output_sizes: list[int],
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config=None,
    prefix: str = "",
    *,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    import vllm_ascend.ops.linear as mod

    self.custom_op, self.tp_rank, self.tp_size = mod.get_parallel_op(
        disable_tp, prefix, self, "column"
    )
    self.output_sizes = list(output_sizes)
    ratios = _ratios_for(disable_tp)
    self._tp_sharding_ratios = ratios
    if ratios is None:
        assert all(
            output_size % self.tp_size == 0 for output_size in output_sizes
        )

    # Call the (patched) column initializer.  It will run the original
    # uniform init first and then rebuild weights with asymmetric sizes.
    mod.AscendColumnParallelLinear.__init__(
        self,
        input_size=input_size,
        output_size=sum(output_sizes),
        bias=bias,
        gather_output=gather_output,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        output_sizes=output_sizes,
        prefix=prefix,
        return_bias=return_bias,
        disable_tp=disable_tp,
    )


def _patched_row_init(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    input_is_parallel: bool = True,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    reduce_results: bool = True,
    quant_config=None,
    prefix: str = "",
    *,
    return_bias: bool = True,
    disable_tp: bool = False,
):
    ratios = _ratios_for(disable_tp)
    self._tp_sharding_ratios = ratios

    # Same divisibility issue as the column path: stock AscendRowParallelLinear
    # requires ``input_size % tp_size == 0``, but heterogeneous ratios make
    # wo_b/down_proj input 8192 with tp=3.  Build the scaffolding with a
    # divisible input size, restore the real input size, then rebuild.
    if ratios is not None:
        import vllm_ascend.ops.linear as mod

        self.custom_op, self.tp_rank, self.tp_size = mod.get_parallel_op(
            disable_tp, prefix, self, "row"
        )
        uniform_input_size = _ceil_divisible(input_size, self.tp_size)
        _ORIG_ROW_INIT(
            self,
            input_size=uniform_input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            out_dtype=out_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        self.input_size = input_size
    else:
        _ORIG_ROW_INIT(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            out_dtype=out_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    if ratios is None:
        return

    from vllm.distributed.utils import get_tp_partition_size

    self.input_size_per_partition = get_tp_partition_size(
        input_size, self.tp_rank, self.tp_size, ratios
    )
    self.output_size_per_partition = output_size
    self.output_partition_sizes = [output_size]

    assert self.quant_method is not None
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2
            if self.quant_method.__class__.__name__
            in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        ),
    )
    if self.custom_op is not None:
        self.custom_op.update_attrs()


def _patched_col_weight_loader_ratio(
    self, param: Parameter, loaded_weight: torch.Tensor
):
    """Ratio-aware port of ``ColumnParallelLinear.weight_loader``.

    Used for Ascend column-parallel weights other than ``wo_a`` under
    heterogeneous TP.  ``wo_a`` keeps its 2-D/3-D reshape handling in
    ``_patched_col_weight_loader``.
    """
    output_dim = getattr(param, "output_dim", None)

    is_sharded_weight = getattr(param, "is_sharded_weight", False)
    use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
    is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()

    if is_gguf_weight and isinstance(param, UninitializedParameter):
        from vllm.distributed.utils import get_tp_partition_size

        final_shape = list(loaded_weight.shape)
        if output_dim is not None:
            final_shape[output_dim] = get_tp_partition_size(
                final_shape[output_dim],
                self.tp_rank,
                self.tp_size,
                self._tp_sharding_ratios,
            )
        param.materialize(final_shape, dtype=loaded_weight.dtype)

    param_data = param.data
    if output_dim is not None and not is_sharded_weight:
        from vllm.distributed.utils import get_tp_partition_offset

        shard_size = param_data.shape[output_dim]
        start_idx = get_tp_partition_offset(
            total_size=loaded_weight.shape[output_dim],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_sharding_ratios=self._tp_sharding_ratios,
        )
        loaded_weight = loaded_weight.narrow(
            output_dim, start_idx, shard_size
        )

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)


def _patched_col_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    """AscendColumnParallelLinear.weight_loader with asymmetric wo_a slicing."""
    ratios = getattr(self, "_tp_sharding_ratios", None)
    if ratios is None:
        return _ORIG_COL_WEIGHT_LOADER(self, param, loaded_weight)

    prefix = getattr(self, "prefix", "")
    if "wo_a" not in prefix:
        return _patched_col_weight_loader_ratio(self, param, loaded_weight)

    from vllm.distributed.utils import (
        get_tp_partition_offset,
        get_tp_partition_size,
    )

    if self.weight.ndim == 2:
        # Delegate the asymmetric column narrowing to the plugin's vLLM
        # asymmetric ColumnParallelLinear weight loader (it handles packed /
        # quantized parameters), then apply the same wo_a reshape as stock.
        self.tp_asymmetric_shardings = ratios
        from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.layers.patch_linear import (
            ColumnParallelLinearAsymmetric,
        )

        ColumnParallelLinearAsymmetric.weight_loader(
            self, param, loaded_weight
        )
        self.weight.data = (
            self.weight.data.view(
                self.n_local_groups, self.o_lora_rank, -1
            )
            .transpose(2, 1)
            .contiguous()
        )
        return

    # RL update flows: wo_a may already be transformed into
    # [n_local_groups, hidden_size, o_lora_rank].
    shard_size = self.n_local_groups * self.o_lora_rank
    start_idx = get_tp_partition_offset(
        loaded_weight.shape[0], self.tp_rank, self.tp_size, ratios
    )
    if loaded_weight.shape[0] != shard_size:
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
    loaded_weight = (
        loaded_weight.view(self.n_local_groups, self.o_lora_rank, -1)
        .transpose(2, 1)
        .contiguous()
    )
    if loaded_weight.shape != self.weight.shape:
        raise ValueError(
            f"Unexpected wo_a weight shape {tuple(loaded_weight.shape)}, "
            f"expected {tuple(self.weight.shape)}"
        )
    self.weight.data.copy_(loaded_weight)


def _patched_row_weight_loader(
    self, param: Parameter, loaded_weight: torch.Tensor
):
    """Ratio-aware port of ``RowParallelLinear.weight_loader``.

    hetero_cp patched this method in ``vllm/model_executor/layers/linear.py``.
    vllm_plugins keeps the stock vLLM file and patches the Ascend subclass
    here, so DP0's ``[2,1,1]`` partitions must use cumulative offsets rather
    than ``tp_rank * shard_size``.
    """
    ratios = getattr(self, "_tp_sharding_ratios", None)
    if ratios is None:
        return _ORIG_ROW_WEIGHT_LOADER(self, param, loaded_weight)

    input_dim = getattr(param, "input_dim", None)
    use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
    is_sharded_weight = getattr(param, "is_sharded_weight", False)
    is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()

    if is_gguf_weight and isinstance(param, UninitializedParameter):
        from vllm.distributed.utils import get_tp_partition_size

        weight_shape = list(loaded_weight.shape)
        if input_dim:
            weight_shape[input_dim] = get_tp_partition_size(
                weight_shape[input_dim],
                self.tp_rank,
                self.tp_size,
                ratios,
            )
        param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

    param_data = param.data
    if input_dim is not None and not is_sharded_weight:
        from vllm.distributed.utils import get_tp_partition_offset

        shard_size = param_data.shape[input_dim]
        start_idx = get_tp_partition_offset(
            total_size=loaded_weight.shape[input_dim],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_sharding_ratios=ratios,
        )
        loaded_weight = loaded_weight.narrow(
            input_dim, start_idx, shard_size
        )

    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)


def _patched_merged_weight_loader(
    self,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id=None,
):
    """Ratio-aware port of ``MergedColumnParallelLinear.weight_loader``.

    Both the rank-local shard offset inside the merged parameter and the
    checkpoint start index must follow ``tp_sharding_ratios``.
    """
    ratios = getattr(self, "_tp_sharding_ratios", None)
    if ratios is None:
        return _ORIG_MERGED_WEIGHT_LOADER(
            self, param, loaded_weight, loaded_shard_id
        )

    from vllm.distributed.utils import (
        get_tp_partition_offset,
        get_tp_partition_size,
    )

    self.validate_shard_id(loaded_shard_id)

    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if isinstance(loaded_shard_id, tuple) and (
        is_gguf_weight or is_gguf_weight_type
    ):
        raise NotImplementedError(
            "Shard id with multiple indices is not supported for GGUF."
        )
    if is_gguf_weight_type:
        if loaded_shard_id is not None:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
        else:
            param.shard_weight_type = {
                i: loaded_weight.item()
                for i, _ in enumerate(self.output_sizes)
            }
        return

    if is_gguf_weight:
        output_dim = getattr(param, "output_dim", None)
        shard_size = loaded_weight.size(output_dim) // self.tp_size
        start_idx = self.tp_rank * shard_size
        if loaded_shard_id is not None:
            loaded_weight = loaded_weight.narrow(
                output_dim, start_idx, shard_size
            )
            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

    param_data = param.data
    output_dim = getattr(param, "output_dim", None)
    needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

    if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
        if output_dim is None:
            if needs_scalar_to_array:
                from vllm.model_executor.layers.linear import (
                    adjust_scalar_to_fused_array,
                )

                param_data, loaded_weight = adjust_scalar_to_fused_array(
                    param_data, loaded_weight, 0
                )

            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return

        output_sizes = (
            self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
            if loaded_shard_id is not None
            else self.output_sizes
        )
        current_shard_offset = 0
        use_bitsandbytes_4bit = getattr(
            param, "use_bitsandbytes_4bit", False
        )
        if (
            use_bitsandbytes_4bit
            and isinstance(loaded_shard_id, tuple)
            and self.tp_size > 1
        ):
            raise NotImplementedError(
                "Shard id with multiple indices is not supported "
                "for BNB quantization with TP yet."
            )
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        packed_dim = getattr(param, "packed_dim", None)
        for shard_id, shard_offset, shard_size in shard_offsets:
            from vllm.model_executor.layers.linear import (
                adjust_bitsandbytes_4bit_shard,
                adjust_block_scale_shard,
                adjust_marlin_shard,
            )
            from vllm.model_executor.parameter import (
                BlockQuantScaleParameter,
            )

            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )

            if packed_dim == output_dim:
                shard_size = shard_size // param.packed_factor
                shard_offset = shard_offset // param.packed_factor
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            if use_bitsandbytes_4bit:
                index = list(itertools.accumulate([0] + self.output_sizes))
                orig_offsets = {
                    str(i): (index[i], size)
                    for i, size in enumerate(self.output_sizes)
                }
                orig_offsets["total"] = (self.output_size, 0)
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_offsets, str(shard_id)
                )

            loaded_weight_shard = loaded_weight.narrow(
                output_dim, shard_offset, shard_size
            )
            self.weight_loader(param, loaded_weight_shard, shard_id)
        return

    assert loaded_shard_id < len(self.output_sizes)
    if output_dim is not None:
        total_ratio = sum(ratios)
        local_offset = 0
        for i in range(loaded_shard_id):
            s = self.output_sizes[i]
            local_offset += s * ratios[self.tp_rank] // total_ratio
        shard_offset = local_offset
        shard_size = get_tp_partition_size(
            self.output_sizes[loaded_shard_id],
            self.tp_rank,
            self.tp_size,
            ratios,
        )

        from vllm.model_executor.layers.linear import (
            adjust_bitsandbytes_4bit_shard,
            adjust_block_scale_shard,
            adjust_marlin_shard,
        )
        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
        )

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        packed_dim = getattr(param, "packed_dim", None)
        if packed_dim == output_dim:
            shard_size = round(shard_size // param.packed_factor)
            shard_offset = round(shard_offset // param.packed_factor)
            shard_size, shard_offset = adjust_marlin_shard(
                param, shard_size, shard_offset
            )

        use_bitsandbytes_4bit = getattr(
            param, "use_bitsandbytes_4bit", False
        )
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        if use_bitsandbytes_4bit:
            index = list(itertools.accumulate([0] + self.output_sizes))
            orig_offsets = {
                str(i): (index[i], size)
                for i, size in enumerate(self.output_sizes)
            }
            orig_offsets["total"] = (self.output_size, 0)
            shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                param, orig_offsets, str(loaded_shard_id)
            )
        param_data = param_data.narrow(
            output_dim, shard_offset, shard_size
        )
        _full_sz = self.output_sizes[loaded_shard_id]
        start_idx = get_tp_partition_offset(
            _full_sz, self.tp_rank, self.tp_size, ratios
        )
        if not is_sharded_weight:
            loaded_weight = loaded_weight.narrow(
                output_dim, start_idx, shard_size
            )
    elif needs_scalar_to_array:
        from vllm.model_executor.layers.linear import (
            adjust_scalar_to_fused_array,
        )

        param_data, loaded_weight = adjust_scalar_to_fused_array(
            param_data, loaded_weight, loaded_shard_id
        )
    else:
        ignore_warning = getattr(param, "ignore_warning", False)
        if not ignore_warning:
            logger.warning(
                "Loading a weight without `output_dim` attribute in "
                "MergedColumnParallelLinear, assume the weight is "
                "the same for all partitions."
            )

    assert param_data.shape == loaded_weight.shape
    param_data.copy_(loaded_weight)


def _patched_merged_weight_loader_v2(
    self,
    param,
    loaded_weight: torch.Tensor,
    loaded_shard_id=None,
):
    """Ratio-aware port of ``MergedColumnParallelLinear.weight_loader_v2``.

    ``patch_hetero_parameter`` already makes the v2 parameter loader ratio
    aware, but the merged layer still has to pass the ratio-correct local
    ``shard_offset`` / ``shard_size`` to that loader.
    """
    ratios = getattr(self, "_tp_sharding_ratios", None)
    if ratios is None:
        return _ORIG_MERGED_WEIGHT_LOADER_V2(
            self, param, loaded_weight, loaded_shard_id
        )

    from vllm.distributed.utils import get_tp_partition_size
    from vllm.model_executor.layers.linear import adjust_block_scale_shard
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        BlockQuantScaleParameter,
        PerTensorScaleParameter,
        RowvLLMParameter,
    )

    self.validate_shard_id(loaded_shard_id)
    if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
        if isinstance(param, PerTensorScaleParameter):
            if isinstance(loaded_shard_id, tuple):
                for idx in loaded_shard_id:
                    param.load_merged_column_weight(
                        loaded_weight=loaded_weight, shard_id=idx
                    )
            else:
                for idx in range(param.data.shape[0]):
                    param.load_merged_column_weight(
                        loaded_weight=loaded_weight, shard_id=idx
                    )
            return
        elif type(param) in (RowvLLMParameter, BasevLLMParameter):
            param.load_merged_column_weight(loaded_weight=loaded_weight)
            return

        output_sizes = (
            [self.output_sizes[idx] for idx in loaded_shard_id]
            if loaded_shard_id
            else None
        )
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            output_sizes = [
                adjust_block_scale_shard(weight_block_size, size, 0)[0]
                for size in (output_sizes or self.output_sizes)
            ]
        self._load_fused_module_from_checkpoint(
            param, loaded_weight, output_sizes=output_sizes
        )
        return

    assert loaded_shard_id < len(self.output_sizes)

    total_ratio = sum(ratios)
    local_offset = 0
    for i in range(loaded_shard_id):
        s = self.output_sizes[i]
        local_offset += s * ratios[self.tp_rank] // total_ratio
    shard_offset = local_offset
    shard_size = get_tp_partition_size(
        self.output_sizes[loaded_shard_id],
        self.tp_rank,
        self.tp_size,
        ratios,
    )

    if isinstance(param, BlockQuantScaleParameter):
        weight_block_size = getattr(self, "weight_block_size", None)
        shard_size, shard_offset = adjust_block_scale_shard(
            weight_block_size, shard_size, shard_offset
        )

    param.load_merged_column_weight(
        loaded_weight=loaded_weight,
        shard_id=loaded_shard_id,
        shard_offset=shard_offset,
        shard_size=shard_size,
        tp_rank=self.tp_rank,
    )


def apply_hetero_ascend_linear_patch():
    """Patch Ascend custom-op linear classes for heterogeneous TP."""
    global _PATCHED
    global _ORIG_COL_INIT, _ORIG_MERGED_INIT, _ORIG_ROW_INIT
    global _ORIG_COL_WEIGHT_LOADER, _ORIG_ROW_WEIGHT_LOADER
    global _ORIG_MERGED_WEIGHT_LOADER, _ORIG_MERGED_WEIGHT_LOADER_V2
    global _ORIG_ASCEND_UNQUANT_PROCESS, _ORIG_VLLM_UNQUANT_PROCESS

    if _PATCHED:
        return

    import vllm_ascend.ops.linear as mod

    _ORIG_COL_INIT = mod.AscendColumnParallelLinear.__init__
    _ORIG_MERGED_INIT = mod.AscendMergedColumnParallelLinear.__init__
    _ORIG_ROW_INIT = mod.AscendRowParallelLinear.__init__
    _ORIG_COL_WEIGHT_LOADER = mod.AscendColumnParallelLinear.weight_loader
    _ORIG_ROW_WEIGHT_LOADER = mod.AscendRowParallelLinear.weight_loader
    _ORIG_MERGED_WEIGHT_LOADER = (
        mod.AscendMergedColumnParallelLinear.weight_loader
    )
    _ORIG_MERGED_WEIGHT_LOADER_V2 = (
        mod.AscendMergedColumnParallelLinear.weight_loader_v2
    )

    mod.AscendColumnParallelLinear.__init__ = _patched_col_init
    mod.AscendMergedColumnParallelLinear.__init__ = _patched_merged_init
    mod.AscendRowParallelLinear.__init__ = _patched_row_init
    mod.AscendColumnParallelLinear.weight_loader = _patched_col_weight_loader
    mod.AscendRowParallelLinear.weight_loader = _patched_row_weight_loader
    mod.AscendMergedColumnParallelLinear.weight_loader = (
        _patched_merged_weight_loader
    )
    mod.AscendMergedColumnParallelLinear.weight_loader_v2 = (
        _patched_merged_weight_loader_v2
    )

    # W8A8 modelslim keeps DeepSeek-V4 wo_a unquantized; reshape it for the
    # DSA batchmatmul kernels after the original post-load processing.
    if _ORIG_ASCEND_UNQUANT_PROCESS is None:
        _ORIG_ASCEND_UNQUANT_PROCESS = (
            mod.AscendUnquantizedLinearMethod.process_weights_after_loading
        )
        mod.AscendUnquantizedLinearMethod.process_weights_after_loading = (
            _patched_ascend_unquant_process_weights_after_loading
        )

    # Same guard for environments where the layer falls back to the stock
    # vLLM UnquantizedLinearMethod instead of the Ascend subclass.
    try:
        from vllm.model_executor.layers.linear import (
            UnquantizedLinearMethod as VllmUnquantizedLinearMethod,
        )

        if _ORIG_VLLM_UNQUANT_PROCESS is None:
            _ORIG_VLLM_UNQUANT_PROCESS = (
                VllmUnquantizedLinearMethod.process_weights_after_loading
            )
            VllmUnquantizedLinearMethod.process_weights_after_loading = (
                _patched_vllm_unquant_process_weights_after_loading
            )
    except ImportError:
        pass

    _PATCHED = True
