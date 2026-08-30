# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Regression tests for KV extra-config validation during degrade/hetero restart."""

import importlib.util
import os
import types
import unittest

_PATCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "patch_hetero_ascend_config.py")
)


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_hetero_ascend_config_under_test", _PATCH_PATH
    )
    patch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    return patch_module


class _ParallelConfig:
    def __init__(self, dp, tp, is_hetero=False, tp_sizes=None):
        self.data_parallel_size = dp
        self.tensor_parallel_size = tp
        self.is_heterogeneous_tp = is_hetero
        self._tp_sizes = tp_sizes or [tp] * dp

    def get_tp_size_for_dp(self, dp_rank):
        return self._tp_sizes[dp_rank]


class _KVTransferConfig:
    def __init__(self, role, extra):
        self._role = role
        self._extra = extra

    @property
    def is_kv_producer(self):
        return self._role == "producer"

    @property
    def is_kv_consumer(self):
        return self._role == "consumer"

    def get_from_extra_config(self, name, default):
        return self._extra.get(name, default)


class _VllmConfig:
    def __init__(self, parallel_config, kv_transfer_config, additional_config=None):
        self.parallel_config = parallel_config
        self.kv_transfer_config = kv_transfer_config
        self.additional_config = additional_config or {}


def _decode_degrade_vllm_config():
    return _VllmConfig(
        _ParallelConfig(dp=15, tp=1),
        _KVTransferConfig("consumer", {"decode": {"dp_size": 16, "tp_size": 1}}),
        {
            "zero_interrupt_config": {
                "engine_parallel_config": [
                    {"dp": 16, "tp": 1, "new_dp": 15, "new_tp": 1},
                    {"dp": 16, "tp": 1, "new_dp": 0, "new_tp": 0},
                ]
            }
        },
    )


class TestPatchHeteroAscendConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch()

    def test_decode_dp_degrade_accepts_original_extra_dp(self):
        # DP16 -> DP15 后，远端 pool 描述仍是 dp16，应通过校验。
        self.patch._patched_check_kv_extra_config(
            _decode_degrade_vllm_config()
        )

    def test_decode_dp_mismatch_without_strategy_still_rejected(self):
        with self.assertRaises(ValueError):
            self.patch._patched_check_kv_extra_config(
                _VllmConfig(
                    _ParallelConfig(dp=15, tp=1),
                    _KVTransferConfig(
                        "consumer", {"decode": {"dp_size": 16, "tp_size": 1}}
                    ),
                )
            )

    def test_hetero_prefill_tp_accepts_pool_tp(self):
        vllm_config = _VllmConfig(
            _ParallelConfig(dp=4, tp=3, is_hetero=True, tp_sizes=[3, 4, 4, 4]),
            _KVTransferConfig(
                "producer", {"prefill": {"dp_size": 4, "tp_size": 4}}
            ),
        )
        self.patch._patched_check_kv_extra_config(vllm_config)


if __name__ == "__main__":
    unittest.main()
