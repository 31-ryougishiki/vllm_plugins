# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Regression tests for the runtime patch-family dispatch in
``vllm/model_executor/layers/fused_moe/config.py``.

The unified replacement file must not route DeepSeek-V4 pure-DP shrink
strategies through the 0829 legacy asymmetric helpers: those helpers derive
the flattened TP rank from the ORIGINAL ``data_parallel_rank`` in
``zero_interrupt_config``, while the surviving-DP barrier and the executor
have already renumbered the active executors. The test loads the source file
directly with minimal stand-ins, so it does not need a vllm installation.
"""

import importlib.util
import os
import sys
import types
import unittest

_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "config.py",
    )
)


def _package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _cdiv(a, b):
    return (a + b - 1) // b


def _install_standins():
    _package("vllm")
    _package("vllm.config")
    _package("vllm.utils")
    _package("vllm.model_executor")
    _package("vllm.model_executor.layers")
    _package("vllm.model_executor.layers.fused_moe")
    _package("vllm.model_executor.layers.quantization")
    _package("vllm.model_executor.layers.quantization.utils")

    _module(
        "vllm.config",
        ParallelConfig=object,
        SchedulerConfig=type(
            "SchedulerConfig",
            (),
            {"DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP": 8192},
        ),
        get_current_vllm_config_or_none=lambda: None,
    )
    _module("vllm.config.kernel", MoEBackend=object)
    _module(
        "vllm.distributed",
        get_dp_group=lambda: None,
        get_pcp_group=lambda: None,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    _module(
        "vllm.logger",
        init_logger=lambda name: __import__("logging").getLogger(name),
    )
    _module(
        "vllm.model_executor.layers.fused_moe.activation",
        MoEActivation=type("MoEActivation", (), {}),
    )
    _module(
        "vllm.model_executor.layers.quantization.utils.ocp_mx_utils",
        OCP_MX_DTYPES=set(),
        OCP_MX_Scheme=type("OCP_MX_Scheme", (), {}),
    )
    _module(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        GroupShape=type(
            "GroupShape",
            (),
            {"PER_TOKEN": object(), "PER_TENSOR": object()},
        ),
    )
    _module(
        "vllm.platforms",
        current_platform=types.SimpleNamespace(fp8_dtype=lambda: None),
    )
    _module("vllm.utils.import_utils", has_triton_kernels=lambda: False)
    _module("vllm.utils.math_utils", cdiv=_cdiv)

    # Lazy helpers used only by the 0829 legacy branch.
    _package("vllm.distributed.parallel_state")
    _module(
        "vllm.distributed.parallel_state",
        asym_world_size=lambda zero_interrupt_config: sum(
            int(c["new_tp"])
            for c in zero_interrupt_config["engine_parallel_config"]
        ),
        get_global_rank_asym=lambda rank: int(rank) + 5,
    )


def _load_config():
    spec = importlib.util.spec_from_file_location(
        "fused_moe_config_under_test", _CONFIG_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeParallelConfig:
    is_heterogeneous_tp = False

    def __init__(self, cfg):
        self._cfg = cfg


class _FakeVllmConfig:
    def __init__(self, additional_config):
        self.additional_config = additional_config
        self.parallel_config = _FakeParallelConfig(additional_config)


_DEEPSEEK_STRATEGY = {
    "executor_id": "1",
    "engine_parallel_config": [
        {"executor_id": "0", "dp": 16, "tp": 1,
         "data_parallel_rank": 0, "new_dp": 15, "new_tp": 1},
        {"executor_id": "1", "dp": 16, "tp": 1,
         "data_parallel_rank": 1, "new_dp": 15, "new_tp": 1},
    ],
}


class TestFusedMoeConfigDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.config",
                "vllm.config.kernel",
                "vllm.distributed",
                "vllm.distributed.parallel_state",
                "vllm.logger",
                "vllm.model_executor",
                "vllm.model_executor.layers",
                "vllm.model_executor.layers.fused_moe",
                "vllm.model_executor.layers.fused_moe.activation",
                "vllm.model_executor.layers.quantization",
                "vllm.model_executor.layers.quantization.utils",
                "vllm.model_executor.layers.quantization.utils.ocp_mx_utils",
                "vllm.model_executor.layers.quantization.utils.quant_utils",
                "vllm.platforms",
                "vllm.utils",
                "vllm.utils.import_utils",
                "vllm.utils.math_utils",
            )
        }
        cls._saved_env = os.environ.get("VLLM_ITS_DEEPSEEK_V4")
        _install_standins()
        cls._config = _load_config()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        if cls._saved_env is None:
            os.environ.pop("VLLM_ITS_DEEPSEEK_V4", None)
        else:
            os.environ["VLLM_ITS_DEEPSEEK_V4"] = cls._saved_env

    def _set_current_config(self, additional_config):
        import vllm.config as vllm_config

        vllm_config.get_current_vllm_config_or_none = lambda: _FakeVllmConfig(
            additional_config
        )

    def test_deepseek_pure_dp_uses_uniform_renumbered_rank(self):
        os.environ["VLLM_ITS_DEEPSEEK_V4"] = "1"
        self._set_current_config(
            {"zero_interrupt_config": dict(_DEEPSEEK_STRATEGY)}
        )
        # Executor 1 was renumbered to survivor rank 0 after executor 0 was
        # scaled to zero; the uniform formula must use the passed dp_rank.
        size, rank = self._config.FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=1, dp_size=15, dp_rank=0, pcp_size=1, pcp_rank=0
        )
        self.assertEqual((size, rank), (15, 0))

    def test_0829_family_keeps_legacy_asym_helpers(self):
        os.environ["VLLM_ITS_DEEPSEEK_V4"] = "0"
        self._set_current_config(
            {"zero_interrupt_config": dict(_DEEPSEEK_STRATEGY)}
        )
        size, rank = self._config.FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=1, dp_size=15, dp_rank=0, pcp_size=1, pcp_rank=0
        )
        # Stand-in asym_world_size sums new_tp; stand-in get_global_rank_asym
        # returns tp_rank + 5.
        self.assertEqual((size, rank), (2, 5))


if __name__ == "__main__":
    unittest.main()
