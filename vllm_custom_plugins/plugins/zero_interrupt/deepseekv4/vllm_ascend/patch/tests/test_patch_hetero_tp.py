# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Regression test for forward-context patch alias rebinding."""

import importlib.util
import os
import sys
import types
import unittest

_PATCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "patch_hetero_tp.py")
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


def _original_set_ascend_forward_context(*args, **kwargs):
    del args, kwargs


def _original_set_mc2_tokens_capacity(*args, **kwargs):
    del args, kwargs


def _install_standins():
    _package("vllm")
    _module(
        "vllm.config",
        CUDAGraphMode=type("CUDAGraphMode", (), {"NONE": object()}),
        VllmConfig=object,
    )
    _package("vllm.distributed")
    _module(
        "vllm.distributed",
        get_dp_group=lambda: None,
        get_ep_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 4,
    )
    _package("vllm.forward_context")
    _module(
        "vllm.forward_context",
        BatchDescriptor=object,
        set_forward_context=lambda **kwargs: None,
        get_forward_context=lambda: None,
    )

    _package("vllm_ascend")
    _package("vllm_ascend.patch")
    return _module(
        "vllm_ascend.ascend_forward_context",
        set_ascend_forward_context=_original_set_ascend_forward_context,
        set_mc2_tokens_capacity=_original_set_mc2_tokens_capacity,
        _select_a3_moe_comm_method=lambda *args: None,
        _ExtraForwardContextProxy=type(
            "_ExtraForwardContextProxy", (), {"extra_attrs": []}
        ),
    )


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_hetero_tp_under_test", _PATCH_PATH
    )
    patch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    return patch_module


class TestForwardContextAliasRebinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.config",
                "vllm.distributed",
                "vllm.forward_context",
                "vllm_ascend",
                "vllm_ascend.patch",
                "vllm_ascend.ascend_forward_context",
                "fake_consumer_module",
            )
        }
        cls._afc = _install_standins()
        cls._patch = _load_patch()

        # Simulate model_runner_v1/spec_decode modules that imported the
        # original function objects before the patch runs.
        cls._consumer = _module(
            "fake_consumer_module",
            set_ascend_forward_context=_original_set_ascend_forward_context,
            set_mc2_tokens_capacity=_original_set_mc2_tokens_capacity,
        )
        cls._unrelated_value = object()
        cls._unrelated = _module(
            "fake_unrelated_module",
            set_ascend_forward_context=cls._unrelated_value,
        )

        cls._patch.apply_hetero_forward_context_patch()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def test_afc_module_is_patched(self):
        self.assertIs(
            self._afc.set_ascend_forward_context,
            self._patch._patched_set_ascend_forward_context,
        )
        self.assertIs(
            self._afc.set_mc2_tokens_capacity,
            self._patch._patched_set_mc2_tokens_capacity,
        )

    def test_previously_imported_aliases_are_refreshed(self):
        self.assertIs(
            self._consumer.set_ascend_forward_context,
            self._patch._patched_set_ascend_forward_context,
        )
        self.assertIs(
            self._consumer.set_mc2_tokens_capacity,
            self._patch._patched_set_mc2_tokens_capacity,
        )

    def test_unrelated_attributes_are_not_overwritten(self):
        self.assertIs(
            self._unrelated.set_ascend_forward_context,
            self._unrelated_value,
        )

    def test_extra_ctx_proxy_gets_per_dp_fields(self):
        self.assertIn(
            "per_dp_tp_sizes",
            self._afc._ExtraForwardContextProxy.extra_attrs,
        )
        self.assertIn(
            "per_dp_padded_lengths",
            self._afc._ExtraForwardContextProxy.extra_attrs,
        )


if __name__ == "__main__":
    unittest.main()
