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

import torch
from torch.nn.parameter import Parameter

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.model_executor.utils import set_weight_attrs

_PATCHED = False
_ORIG_COL_INIT = None
_ORIG_MERGED_INIT = None
_ORIG_ROW_INIT = None
_ORIG_COL_WEIGHT_LOADER = None
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
        # Restore the real geometry before rebuilding weights.
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


def _patched_col_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    """AscendColumnParallelLinear.weight_loader with asymmetric wo_a slicing."""
    ratios = getattr(self, "_tp_sharding_ratios", None)
    if not (
        ratios is not None
        and "wo_a" in getattr(self, "prefix", "")
    ):
        return _ORIG_COL_WEIGHT_LOADER(self, param, loaded_weight)

    from vllm.distributed.utils import (
        get_tp_partition_offset,
        get_tp_partition_size,
    )

    if self.weight.ndim == 2:
        # Delegate the asymmetric column narrowing to the plugin's vLLM
        # asymmetric ColumnParallelLinear weight loader (it handles packed /
        # quantized parameters), then apply the same wo_a reshape as stock.
        self.tp_asymmetric_shardings = ratios
        from vllm_custom_plugins.plugins.zero_interrupt.vllm.model_executor.layers.patch_linear import (
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


def apply_hetero_ascend_linear_patch():
    """Patch Ascend custom-op linear classes for heterogeneous TP."""
    global _PATCHED
    global _ORIG_COL_INIT, _ORIG_MERGED_INIT, _ORIG_ROW_INIT
    global _ORIG_COL_WEIGHT_LOADER
    global _ORIG_ASCEND_UNQUANT_PROCESS, _ORIG_VLLM_UNQUANT_PROCESS

    if _PATCHED:
        return

    import vllm_ascend.ops.linear as mod

    _ORIG_COL_INIT = mod.AscendColumnParallelLinear.__init__
    _ORIG_MERGED_INIT = mod.AscendMergedColumnParallelLinear.__init__
    _ORIG_ROW_INIT = mod.AscendRowParallelLinear.__init__
    _ORIG_COL_WEIGHT_LOADER = mod.AscendColumnParallelLinear.weight_loader

    mod.AscendColumnParallelLinear.__init__ = _patched_col_init
    mod.AscendMergedColumnParallelLinear.__init__ = _patched_merged_init
    mod.AscendRowParallelLinear.__init__ = _patched_row_init
    mod.AscendColumnParallelLinear.weight_loader = _patched_col_weight_loader

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
