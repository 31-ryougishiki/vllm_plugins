# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Regression tests for DSA-CP draft metadata LCM alignment."""

import importlib.util
import os
import sys
import types
import unittest

_PATCH_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "patch_deepseek_v4_attention_hetero.py",
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


def _install_standins():
    _package("vllm")
    _module(
        "vllm.config",
        VllmConfig=object,
        get_current_vllm_config=lambda: None,
    )
    _package("vllm.distributed")
    _module(
        "vllm.distributed",
        get_tensor_model_parallel_rank=lambda: 0,
    )
    _module(
        "vllm.distributed.utils",
        get_tp_partition_offset=lambda *args: 0,
        get_tp_partition_size=lambda *args: 0,
    )
    _package("vllm_ascend")
    _module(
        "vllm_ascend.utils",
        is_drafter_moe_model=lambda vllm_config: True,
    )


class _FakeParallelConfig:
    is_heterogeneous_tp = True
    data_parallel_size = 4

    def get_tp_size_for_dp(self, dp_rank):
        return [3, 4, 4, 4][dp_rank]


class _FakeBuilder:
    def __init__(self, is_draft):
        self.vllm_config = types.SimpleNamespace(
            parallel_config=_FakeParallelConfig()
        )
        self._is_dsa_cp_draft_builder = is_draft


class TestBuildLocalTokenMetadataAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.config",
                "vllm.distributed",
                "vllm.distributed.utils",
                "vllm_ascend",
                "vllm_ascend.utils",
            )
        }
        _install_standins()
        spec = importlib.util.spec_from_file_location(
            "patch_deepseek_v4_attention_hetero_under_test",
            _PATCH_PATH,
        )
        cls._patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls._patch)

        def original(self, num_reqs, num_input_tokens, *args, **kwargs):
            del self, num_reqs, args, kwargs
            return num_input_tokens

        cls._patch._ORIG_BUILD_LOCAL_TOKEN_METADATA = original

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _run(self, is_draft):
        builder = _FakeBuilder(is_draft)
        return self._patch._patched_build_local_token_metadata(
            builder,
            num_reqs=2,
            num_input_tokens=5,
            input_positions=None,
            query_start_loc=None,
            seq_lens=None,
            use_cache=False,
        )

    def test_main_model_uses_lcm(self):
        self.assertEqual(self._run(is_draft=False), 12)

    def test_moe_drafter_uses_lcm(self):
        # DeepSeek-V4 MTP is an MoE drafter, so flash_comm stays enabled and
        # the reduced hidden stream is LCM-padded; metadata must match.
        self.assertEqual(self._run(is_draft=True), 12)

    def test_dense_drafter_skips_lcm(self):
        sys.modules["vllm_ascend.utils"].is_drafter_moe_model = (
            lambda vllm_config: False
        )
        try:
            self.assertEqual(self._run(is_draft=True), 5)
        finally:
            sys.modules["vllm_ascend.utils"].is_drafter_moe_model = (
                lambda vllm_config: True
            )


if __name__ == "__main__":
    unittest.main()
