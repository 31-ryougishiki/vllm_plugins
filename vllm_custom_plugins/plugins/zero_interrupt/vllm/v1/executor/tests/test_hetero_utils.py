import importlib.util
import os
import unittest

# Load utils.py directly so the test does not need a vllm/vllm-ascend
# environment (the package __init__ chain imports vllm).
_UTILS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "utils.py")
)
_spec = importlib.util.spec_from_file_location(
    "zero_interrupt_executor_utils_under_test", _UTILS_PATH
)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

get_global_start_rank = _utils.get_global_start_rank
get_global_world_size = _utils.get_global_world_size
get_heterogeneous_dp_config = _utils.get_heterogeneous_dp_config
get_scale_to_zero_dp_ranks = _utils.get_scale_to_zero_dp_ranks
get_surviving_dp_barrier_geometry = _utils.get_surviving_dp_barrier_geometry
get_tp_asymmetric_shardings = _utils.get_tp_asymmetric_shardings
is_heterogeneous_restart = _utils.is_heterogeneous_restart


class TestHeteroUtils(unittest.TestCase):
    def _strategy(self):
        return {
            "deploy_type": "PD_REBUILD",
            "executor_id": "2",
            "engine_parallel_config": [
                {
                    "executor_id": "0",
                    "dp": 4,
                    "tp": 4,
                    "data_parallel_rank": 0,
                    "new_dp": 4,
                    "new_tp": 3,
                    "tp_asymmetric_shardings": [2, 1, 1],
                },
                {
                    "executor_id": "1",
                    "dp": 4,
                    "tp": 4,
                    "data_parallel_rank": 1,
                    "new_dp": 4,
                    "new_tp": 4,
                    "tp_asymmetric_shardings": None,
                },
                {
                    "executor_id": "2",
                    "dp": 4,
                    "tp": 4,
                    "data_parallel_rank": 2,
                    "new_dp": 4,
                    "new_tp": 4,
                    "tp_asymmetric_shardings": None,
                },
                {
                    "executor_id": "3",
                    "dp": 4,
                    "tp": 4,
                    "data_parallel_rank": 3,
                    "new_dp": 4,
                    "new_tp": 4,
                    "tp_asymmetric_shardings": None,
                },
            ],
        }

    def test_explicit_shardings_are_preferred(self):
        cfg = self._strategy()
        cfg["executor_id"] = "0"
        self.assertEqual(
            get_tp_asymmetric_shardings(cfg), [2, 1, 1]
        )

    def test_legacy_shardings(self):
        cfg = self._strategy()
        cfg["executor_id"] = "0"
        cfg["engine_parallel_config"][0]["tp_asymmetric_shardings"] = None
        # 4 heads across 3 ranks: [1, 1, 2]
        self.assertEqual(
            get_tp_asymmetric_shardings(cfg), [1, 1, 2]
        )

    def test_uniform_when_same_tp(self):
        cfg = self._strategy()
        cfg["executor_id"] = "1"
        self.assertEqual(
            get_tp_asymmetric_shardings(cfg), [1, 1, 1, 1]
        )

    def test_heterogeneous_dp_config(self):
        cfg = self._strategy()
        configs = get_heterogeneous_dp_config(cfg)
        self.assertEqual(
            configs,
            [
                {
                    "dp_rank": 0,
                    "tp_size": 3,
                    "tp_sharding_ratios": [2, 1, 1],
                },
                {"dp_rank": 1, "tp_size": 4, "tp_sharding_ratios": None},
                {"dp_rank": 2, "tp_size": 4, "tp_sharding_ratios": None},
                {"dp_rank": 3, "tp_size": 4, "tp_sharding_ratios": None},
            ],
        )

    def test_global_world_size_and_rank(self):
        cfg = self._strategy()
        self.assertEqual(get_global_world_size(cfg), 15)
        cfg["executor_id"] = "0"
        self.assertEqual(get_global_start_rank(cfg), 0)
        cfg["executor_id"] = "1"
        self.assertEqual(get_global_start_rank(cfg), 3)
        cfg["executor_id"] = "2"
        self.assertEqual(get_global_start_rank(cfg), 7)
        cfg["executor_id"] = "3"
        self.assertEqual(get_global_start_rank(cfg), 11)

    def test_global_start_rank_ignores_strategy_list_order(self):
        cfg = self._strategy()
        cfg["engine_parallel_config"].reverse()
        cfg["executor_id"] = "2"
        # Sorted by data_parallel_rank: dp0 (3 ranks) + dp1 (4 ranks) = 7.
        self.assertEqual(get_global_start_rank(cfg), 7)

    def test_is_heterogeneous_restart(self):
        cfg = self._strategy()
        self.assertTrue(is_heterogeneous_restart(cfg))
        for conf in cfg["engine_parallel_config"]:
            conf["new_tp"] = conf["tp"]
            conf["tp_asymmetric_shardings"] = None
        self.assertFalse(is_heterogeneous_restart(cfg))

    def test_renumber_after_scale_to_zero(self):
        cfg = self._strategy()
        cfg["engine_parallel_config"][0]["new_tp"] = 0
        cfg["engine_parallel_config"][0]["new_dp"] = 0
        configs = get_heterogeneous_dp_config(cfg)
        self.assertEqual([c["dp_rank"] for c in configs], [0, 1, 2])
        self.assertEqual([c["tp_size"] for c in configs], [4, 4, 4])
        self.assertEqual(get_global_world_size(cfg), 12)
        cfg["executor_id"] = "3"
        self.assertEqual(get_global_start_rank(cfg), 8)

    def test_pure_dp_scale_to_zero_is_not_heterogeneous(self):
        cfg = {
            "executor_id": "15",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(15)
            ] + [
                {"executor_id": "15", "dp": 16, "tp": 1,
                 "data_parallel_rank": 15, "new_dp": 0, "new_tp": 0}
            ],
        }
        self.assertFalse(is_heterogeneous_restart(cfg))
        self.assertEqual(get_tp_asymmetric_shardings(cfg), [])

    def test_surviving_dp_barrier_geometry_excludes_scale_to_zero(self):
        cfg = {
            "executor_id": "14",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(15)
            ] + [
                {"executor_id": "15", "dp": 16, "tp": 1,
                 "data_parallel_rank": 15, "new_dp": 0, "new_tp": 0}
            ],
        }
        self.assertEqual(get_scale_to_zero_dp_ranks(cfg), {15})
        self.assertEqual(
            get_surviving_dp_barrier_geometry(cfg),
            (15, 14, {15}),
        )

    def test_surviving_dp_barrier_geometry_renumbers_after_removed_rank(self):
        cfg = {
            "executor_id": "9",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(16)
            ],
        }
        # Remove original DP rank 4: ranks 5..15 shift down by one.
        cfg["engine_parallel_config"][4]["new_tp"] = 0
        cfg["engine_parallel_config"][4]["new_dp"] = 0
        self.assertEqual(get_scale_to_zero_dp_ranks(cfg), {4})
        self.assertEqual(
            get_surviving_dp_barrier_geometry(cfg),
            (15, 8, {4}),
        )

    def test_surviving_dp_barrier_geometry_for_zero_executor(self):
        cfg = {
            "executor_id": "15",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(15)
            ] + [
                {"executor_id": "15", "dp": 16, "tp": 1,
                 "data_parallel_rank": 15, "new_dp": 0, "new_tp": 0}
            ],
        }
        self.assertEqual(
            get_surviving_dp_barrier_geometry(cfg),
            (0, 0, {15}),
        )

    def test_surviving_dp_barrier_geometry_rejects_inconsistent_new_dp(self):
        cfg = {
            "executor_id": "0",
            "engine_parallel_config": [
                {"executor_id": "0", "dp": 16, "tp": 1,
                 "data_parallel_rank": 0, "new_dp": 14, "new_tp": 1},
                {"executor_id": "1", "dp": 16, "tp": 1,
                 "data_parallel_rank": 1, "new_dp": 15, "new_tp": 1},
                {"executor_id": "2", "dp": 16, "tp": 1,
                 "data_parallel_rank": 2, "new_dp": 0, "new_tp": 0},
            ],
        }
        with self.assertRaises(ValueError):
            get_surviving_dp_barrier_geometry(cfg)


if __name__ == "__main__":
    unittest.main()
