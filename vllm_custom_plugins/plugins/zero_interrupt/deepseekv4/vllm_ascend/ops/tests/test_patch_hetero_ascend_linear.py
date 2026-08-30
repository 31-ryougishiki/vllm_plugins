# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Regression tests for the hetero Ascend linear scaffolding patch.

The patch file is loaded directly (like the executor utils tests) so the test
does not need a vllm / vllm-ascend installation; minimal stand-ins are
registered in ``sys.modules``.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

_PATCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "patch_hetero_ascend_linear.py")
)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _install_standins():
    _package("vllm")
    _package("vllm.model_executor")
    _package("vllm.model_executor.layers")
    _module(
        "vllm.config",
        get_current_vllm_config=lambda: types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_text_config=types.SimpleNamespace(
                    o_groups=8, o_lora_rank=1024
                )
            )
        ),
        get_current_vllm_config_or_none=lambda: None,
    )
    _module(
        "vllm.model_executor.layers.linear",
        WEIGHT_LOADER_V2_SUPPORTED=frozenset(),
        UnquantizedLinearMethod=type(
            "UnquantizedLinearMethod",
            (),
            {"process_weights_after_loading": lambda self, layer: None},
        ),
        adjust_block_scale_shard=lambda *a: (_ for _ in ()).throw(
            NotImplementedError
        ),
        adjust_marlin_shard=lambda *a: (_ for _ in ()).throw(
            NotImplementedError
        ),
        adjust_bitsandbytes_4bit_shard=lambda *a: (_ for _ in ()).throw(
            NotImplementedError
        ),
        adjust_scalar_to_fused_array=lambda *a: (_ for _ in ()).throw(
            NotImplementedError
        ),
    )
    _module("vllm.model_executor.utils", set_weight_attrs=lambda *a, **kw: None)
    _module(
        "vllm.model_executor.parameter",
        BlockQuantScaleParameter=type("BlockQuantScaleParameter", (), {}),
        PerTensorScaleParameter=type("PerTensorScaleParameter", (), {}),
        RowvLLMParameter=type("RowvLLMParameter", (), {}),
        BasevLLMParameter=type("BasevLLMParameter", (), {}),
    )
    _package("vllm.distributed")
    _module(
        "vllm.distributed.utils",
        divide=_divide,
        get_current_tp_sharding_ratios=lambda: _RATIOS,
        get_tp_partition_size=_get_tp_partition_size,
        get_tp_partition_offset=_get_tp_partition_offset,
    )

    _package("vllm_ascend")
    _package("vllm_ascend.ops")
    return _module(
        "vllm_ascend.ops.linear",
        AscendUnquantizedLinearMethod=type(
            "AscendUnquantizedLinearMethod",
            (),
            {"process_weights_after_loading": lambda self, layer: None},
        ),
        AscendColumnParallelLinear=None,
        AscendMergedColumnParallelLinear=None,
        AscendRowParallelLinear=None,
    )


def _divide(numerator, denominator):
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


_RATIOS = [2, 1, 1]
_TP_SIZE = 3


def _get_tp_partition_size(total_size, tp_rank, tp_size, ratios=None):
    if ratios is None:
        return _divide(total_size, tp_size)
    total_ratio = sum(ratios)
    sizes = [total_size * ratio // total_ratio for ratio in ratios]
    sizes[-1] += total_size - sum(sizes)
    return sizes[tp_rank]


def _get_tp_partition_offset(
    total_size, tp_rank, tp_size, ratios=None, tp_sharding_ratios=None
):
    if ratios is None:
        ratios = tp_sharding_ratios
    if ratios is None:
        return tp_rank * _divide(total_size, tp_size)
    return sum(
        total_size * ratios[index] // sum(ratios)
        for index in range(tp_rank)
    )


class _FakeCustomOp:
    def update_attrs(self):
        return None


def _fake_get_parallel_op(disable_tp, prefix, layer, direct):
    del disable_tp, prefix, layer, direct
    return _FakeCustomOp(), 0, _TP_SIZE


class _RecordingQuantMethod:
    def __init__(self):
        self.create_calls = []

    def create_weights(
        self,
        layer=None,
        input_size_per_partition=None,
        output_partition_sizes=None,
        input_size=None,
        output_size=None,
        params_dtype=None,
        weight_loader=None,
    ):
        del layer, input_size, output_size
        del params_dtype, weight_loader
        self.create_calls.append(
            (input_size_per_partition, tuple(output_partition_sizes))
        )


class _FakeLayer:
    def __init__(self):
        self.quant_method = _RecordingQuantMethod()
        self.custom_op = None
        self.bias = None
        self.params_dtype = None
        self.weight_loader = lambda *args: None
        self.weight_loader_v2 = lambda *args: None
        self.prefix = "model.layers.0.self_attn.wo_a"

    def register_parameter(self, name, value):
        setattr(self, name, value)


def _orig_col_init(
    self,
    input_size,
    output_size,
    bias=True,
    gather_output=False,
    skip_bias_add=False,
    params_dtype=None,
    quant_config=None,
    output_sizes=None,
    prefix="",
    *,
    return_bias=True,
    disable_tp=False,
):
    """Minimal copy of stock AscendColumnParallelLinear.__init__ checks."""
    del bias, gather_output, skip_bias_add, quant_config
    del return_bias, disable_tp
    self.input_size_per_partition = input_size
    self.output_size_per_partition = _divide(output_size, self.tp_size)
    self.output_partition_sizes = [self.output_size_per_partition]
    if hasattr(self, "output_sizes"):
        self.output_partition_sizes = [
            _divide(size, self.tp_size) for size in self.output_sizes
        ]
    self.input_size = input_size
    self.output_size = output_size
    self.prefix = prefix
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=params_dtype,
        weight_loader=self.weight_loader,
    )


def _orig_row_init(
    self,
    input_size,
    output_size,
    bias=True,
    input_is_parallel=True,
    skip_bias_add=False,
    params_dtype=None,
    out_dtype=None,
    reduce_results=True,
    quant_config=None,
    prefix="",
    *,
    return_bias=True,
    disable_tp=False,
):
    """Minimal copy of stock AscendRowParallelLinear.__init__ checks."""
    del bias, input_is_parallel, skip_bias_add, out_dtype, reduce_results
    del quant_config, return_bias, disable_tp
    self.input_size_per_partition = _divide(input_size, self.tp_size)
    self.output_size_per_partition = output_size
    self.output_partition_sizes = [output_size]
    self.input_size = input_size
    self.output_size = output_size
    self.prefix = prefix
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=params_dtype,
        weight_loader=self.weight_loader,
    )


def _load_patch(linear_module):
    linear_module.get_parallel_op = _fake_get_parallel_op
    col_cls = type("AscendColumnParallelLinear", (), {})
    col_cls.__init__ = _orig_col_init
    col_cls.weight_loader = lambda self, param, loaded: None
    row_cls = type("AscendRowParallelLinear", (), {})
    row_cls.__init__ = _orig_row_init
    row_cls.weight_loader = lambda self, param, loaded: None
    linear_module.AscendColumnParallelLinear = col_cls
    merged_cls = type("AscendMergedColumnParallelLinear", (), {})
    merged_cls.weight_loader = lambda self, param, loaded, shard_id=None: None
    merged_cls.weight_loader_v2 = (
        lambda self, param, loaded, shard_id=None: None
    )
    linear_module.AscendMergedColumnParallelLinear = merged_cls
    linear_module.AscendRowParallelLinear = row_cls

    spec = importlib.util.spec_from_file_location(
        "patch_hetero_ascend_linear_under_test", _PATCH_PATH
    )
    patch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    patch_module.apply_hetero_ascend_linear_patch()
    return patch_module


class TestHeteroAscendLinearScaffolding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.config",
                "vllm.model_executor",
                "vllm.model_executor.layers",
                "vllm.model_executor.layers.linear",
                "vllm.model_executor.utils",
                "vllm.model_executor.parameter",
                "vllm.distributed",
                "vllm.distributed.utils",
                "vllm_ascend",
                "vllm_ascend.ops",
                "vllm_ascend.ops.linear",
            )
        }
        cls._linear_module = _install_standins()
        cls._patch = _load_patch(cls._linear_module)
        cls._col_cls = cls._linear_module.AscendColumnParallelLinear
        cls._row_cls = cls._linear_module.AscendRowParallelLinear

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def test_hetero_col_init_scaffolds_divisible_output_sizes(self):
        layer = _FakeLayer()
        self._col_cls.__init__(
            layer, input_size=4096, output_size=8192, bias=False,
            prefix=layer.prefix,
        )
        self.assertEqual(layer.output_size, 8192)
        self.assertEqual(layer.output_sizes, [8192])
        self.assertEqual(layer.output_size_per_partition, 4096)
        self.assertEqual(layer.output_partition_sizes, [4096])
        # First call scaffolds with 8193 // 3, second rebuilds with the
        # ratio-aware rank-0 partition 8192 * 2 / 4.
        self.assertEqual(
            layer.quant_method.create_calls,
            [(4096, (2731,)), (4096, (4096,))],
        )
        self.assertEqual(layer.n_local_groups, 4)

    def test_hetero_merged_col_init_scaffolds_output_sizes(self):
        layer = _FakeLayer()
        layer.output_sizes = [8192, 2048]
        self._col_cls.__init__(
            layer,
            input_size=4096,
            output_size=10240,
            bias=False,
            output_sizes=[8192, 2048],
            prefix=layer.prefix,
        )
        self.assertEqual(layer.output_sizes, [8192, 2048])
        self.assertEqual(
            layer.output_partition_sizes,
            [4096, 1024],
        )
        self.assertEqual(
            layer.quant_method.create_calls,
            [(4096, (3414,)), (4096, (4096, 1024))],
        )

    def test_hetero_col_init_preserves_preexisting_output_sizes(self):
        # AscendQKVParallelLinear sets its q/k/v split before calling
        # AscendColumnParallelLinear.__init__ WITHOUT output_sizes; the
        # patched initializer must keep that split instead of collapsing it.
        layer = _FakeLayer()
        layer.output_sizes = [1024, 512, 512]
        self._col_cls.__init__(
            layer,
            input_size=4096,
            output_size=2048,
            bias=False,
            prefix=layer.prefix,
            output_sizes=None,
        )
        self.assertEqual(layer.output_sizes, [1024, 512, 512])

    def test_hetero_row_init_scaffolds_divisible_input_size(self):
        layer = _FakeLayer()
        self._row_cls.__init__(
            layer,
            input_size=8192,
            output_size=4096,
            bias=False,
            prefix=layer.prefix,
        )
        self.assertEqual(layer.input_size, 8192)
        self.assertEqual(layer.input_size_per_partition, 4096)
        self.assertEqual(
            layer.quant_method.create_calls,
            [(2731, (4096,)), (4096, (4096,))],
        )

    def test_hetero_row_weight_loader_uses_cumulative_offsets(self):
        hidden = 8
        full_rows = 8192
        loaded = torch.arange(
            full_rows * hidden, dtype=torch.float32
        ).reshape(full_rows, hidden)
        shard_sizes = {0: 4096, 1: 2048, 2: 2048}
        offsets = {0: 0, 1: 4096, 2: 6144}
        for rank in range(3):
            param = _SimpleParam(
                torch.empty(shard_sizes[rank], hidden), input_dim=0
            )
            layer = _SimpleRowLayer(rank)
            self._patch._patched_row_weight_loader(layer, param, loaded)
            expected = loaded[
                offsets[rank] : offsets[rank] + shard_sizes[rank]
            ]
            self.assertTrue(
                torch.equal(param.data, expected),
                f"row rank {rank}: cumulative offset not used",
            )

    def test_hetero_merged_weight_loader_uses_ratio_local_offsets(self):
        hidden = 8
        output_sizes = [2048, 1024]
        full = torch.arange(
            sum(output_sizes) * hidden, dtype=torch.float32
        ).reshape(sum(output_sizes), hidden)

        # ratios [2,1,1] -> rank local partitions 1024/512, 512/256, 512/256
        expected_partitions = {
            0: [1024, 512],
            1: [512, 256],
            2: [512, 256],
        }
        expected_starts = {
            0: [0, 0],
            1: [1024, 512],
            2: [1536, 768],
        }
        # ``weight_loader`` with an int ``loaded_shard_id`` receives the
        # checkpoint tensor for THAT column only (fused tensors use
        # ``loaded_shard_id=None``/a tuple in stock vLLM).  Feed one column
        # tensor per shard so the assertion matches the real loader contract.
        columns = [
            full[: output_sizes[0]],
            full[output_sizes[0] :],
        ]
        for rank in range(3):
            partitions = expected_partitions[rank]
            param = _SimpleParam(
                torch.empty(sum(partitions), hidden), output_dim=0
            )
            layer = _SimpleMergedLayer(rank, output_sizes)
            for shard_id in range(2):
                loaded = columns[shard_id]
                self._patch._patched_merged_weight_loader(
                    layer, param, loaded, loaded_shard_id=shard_id
                )
                local_offset = sum(partitions[:shard_id])
                start = expected_starts[rank][shard_id]
                expected = loaded[start : start + partitions[shard_id]]
                self.assertTrue(
                    torch.equal(
                        param.data[
                            local_offset : local_offset + partitions[shard_id]
                        ],
                        expected,
                    ),
                    f"merged rank {rank} shard {shard_id}: "
                    "ratio local/global offsets not used",
                )


class _SimpleParam:
    def __init__(self, data, input_dim=None, output_dim=None):
        self.data = data
        self.input_dim = input_dim
        self.output_dim = output_dim


class _SimpleRowLayer:
    def __init__(self, rank):
        self.tp_rank = rank
        self.tp_size = 3
        self._tp_sharding_ratios = [2, 1, 1]
        self.prefix = "model.layers.0.self_attn.wo_b"


class _SimpleMergedLayer:
    def __init__(self, rank, output_sizes):
        self.tp_rank = rank
        self.tp_size = 3
        self._tp_sharding_ratios = [2, 1, 1]
        self.output_sizes = output_sizes
        self.output_size = sum(output_sizes)

    def validate_shard_id(self, loaded_shard_id):
        if loaded_shard_id is not None and not 0 <= loaded_shard_id < len(
            self.output_sizes
        ):
            raise ValueError(loaded_shard_id)


if __name__ == "__main__":
    unittest.main()
