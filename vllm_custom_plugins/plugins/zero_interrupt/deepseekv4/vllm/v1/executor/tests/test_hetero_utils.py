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

BarrierPortPool = _utils.BarrierPortPool
get_global_start_rank = _utils.get_global_start_rank
get_global_world_size = _utils.get_global_world_size
get_heterogeneous_dp_config = _utils.get_heterogeneous_dp_config
get_scale_to_zero_dp_ranks = _utils.get_scale_to_zero_dp_ranks
get_surviving_dp_barrier_geometry = _utils.get_surviving_dp_barrier_geometry
get_renumbered_dp_rank = _utils.get_renumbered_dp_rank
get_pd_scheduler_connector_topology = _utils.get_pd_scheduler_connector_topology
get_tp_asymmetric_shardings = _utils.get_tp_asymmetric_shardings
is_heterogeneous_restart = _utils.is_heterogeneous_restart
pin_worker_world_port = _utils.pin_worker_world_port
recover_requires_full_restart = _utils.recover_requires_full_restart
reserve_restart_ports = _utils.reserve_restart_ports


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
        # 4 heads across 3 ranks; remainder goes to the lowest rank to match
        # the DeepSeek-V4 golden topology (DecisionMakingCenter sends no
        # explicit ratios).
        self.assertEqual(
            get_tp_asymmetric_shardings(cfg), [2, 1, 1]
        )

    def test_heterogeneous_dp_config_legacy_shardings_match_top_level(self):
        cfg = self._strategy()
        cfg["executor_id"] = "0"
        cfg["engine_parallel_config"][0]["tp_asymmetric_shardings"] = None
        configs = get_heterogeneous_dp_config(cfg)
        # The per-DP heterogeneous config and the top-level sharding fallback
        # must agree on where the remainder goes. Both derive [2, 1, 1] from
        # old_tp=4 -> new_tp=3; [1, 1, 2] would load different weight/head
        # partitions than the hetero weight loaders expect.
        self.assertEqual(configs[0]["tp_sharding_ratios"], [2, 1, 1])
        self.assertEqual(get_tp_asymmetric_shardings(cfg), [2, 1, 1])

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

    def test_renumbered_dp_rank_matches_barrier_for_middle_fault(self):
        cfg = {
            "executor_id": "9",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(16)
            ],
        }
        cfg["engine_parallel_config"][4]["new_tp"] = 0
        cfg["engine_parallel_config"][4]["new_dp"] = 0
        # Barrier survivors are renumbered 0..14; original rank 9 becomes 8.
        self.assertEqual(
            get_surviving_dp_barrier_geometry(cfg)[1],
            get_renumbered_dp_rank(cfg),
        )
        self.assertEqual(get_renumbered_dp_rank(cfg), 8)

    def test_renumbered_dp_rank_for_scale_to_zero_executor_is_none(self):
        cfg = {
            "executor_id": "0",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 15, "new_tp": 1}
                for i in range(16)
            ],
        }
        cfg["engine_parallel_config"][0]["new_tp"] = 0
        cfg["engine_parallel_config"][0]["new_dp"] = 0
        self.assertIsNone(get_renumbered_dp_rank(cfg))
        cfg["executor_id"] = "1"
        self.assertEqual(get_renumbered_dp_rank(cfg), 0)

    def test_renumbered_dp_rank_rejects_missing_executor(self):
        cfg = {
            "executor_id": "99",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 4, "tp": 4,
                 "data_parallel_rank": i, "new_dp": 4, "new_tp": 4}
                for i in range(4)
            ],
        }
        with self.assertRaises(ValueError):
            get_renumbered_dp_rank(cfg)

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

    def test_recover_barrier_geometry_includes_restored_executor(self):
        cfg = {
            "executor_id": "15",
            "engine_parallel_config": [
                {"executor_id": str(i), "dp": 16, "tp": 1,
                 "data_parallel_rank": i, "new_dp": 16, "new_tp": 1}
                for i in range(16)
            ],
        }
        self.assertEqual(get_scale_to_zero_dp_ranks(cfg), set())
        self.assertEqual(
            get_surviving_dp_barrier_geometry(cfg),
            (16, 15, set()),
        )

    def test_recover_requires_full_restart_for_pure_dp_recovery(self):
        # D 端场景 2 后健康 executor：dp16 -> dp15，tp 不变。
        backup = {"tensor_parallel_size": 1, "data_parallel_size": 16}
        self.assertTrue(
            recover_requires_full_restart(
                backup=backup,
                current_tp=1,
                current_dp=15,
                current_is_heterogeneous=False,
                target_dp=16,
            )
        )
        # 空转 executor：dp15 阶段被缩到 0。
        self.assertTrue(
            recover_requires_full_restart(
                backup=backup,
                current_tp=0,
                current_dp=0,
                current_is_heterogeneous=False,
                target_dp=16,
            )
        )

    def test_recover_requires_full_restart_for_heterogeneous_tp_recovery(self):
        # P 端场景 1 后：异构 TP 恢复为对称 TP，dp 不变。
        backup = {"tensor_parallel_size": 4, "data_parallel_size": 4}
        self.assertTrue(
            recover_requires_full_restart(
                backup=backup,
                current_tp=3,
                current_dp=4,
                current_is_heterogeneous=True,
                target_dp=4,
            )
        )

    def test_recover_requires_full_restart_skips_unchanged_topology(self):
        backup = {"tensor_parallel_size": 1, "data_parallel_size": 16}
        self.assertFalse(
            recover_requires_full_restart(
                backup=backup,
                current_tp=1,
                current_dp=16,
                current_is_heterogeneous=False,
                target_dp=16,
            )
        )

    def test_pd_scheduler_topology_uses_cumulative_offset_for_hetero(self):
        pc = _HeteroParallelConfig(dp_rank=2)
        self.assertEqual(
            get_pd_scheduler_connector_topology(pc, kv_port=36000),
            (36007, 4, 15),
        )

    def test_pd_scheduler_topology_restores_symmetric_after_recover(self):
        pc = _SymmetricParallelConfig(dp_rank=2, tp=4, world_size_across_dp=16)
        self.assertEqual(
            get_pd_scheduler_connector_topology(pc, kv_port=36000),
            (36008, 4, 16),
        )

    def test_pd_scheduler_topology_uses_renumbered_dp_rank(self):
        pc = _SymmetricParallelConfig(dp_rank=0, tp=1, world_size_across_dp=15)
        self.assertEqual(
            get_pd_scheduler_connector_topology(pc, kv_port=36200),
            (36200, 1, 15),
        )


class _HeteroParallelConfig:
    is_heterogeneous_tp = True
    tensor_parallel_size = 4
    pipeline_parallel_size = 1
    prefill_context_parallel_size = 1
    world_size_across_dp = 15

    def __init__(self, dp_rank):
        self.data_parallel_rank = dp_rank

    def get_rank_offset_for_dp(self, dp_rank):
        return [0, 3, 7, 11][dp_rank]


class _SymmetricParallelConfig:
    is_heterogeneous_tp = False
    pipeline_parallel_size = 1
    prefill_context_parallel_size = 1

    def __init__(self, dp_rank, tp, world_size_across_dp):
        self.data_parallel_rank = dp_rank
        self.tensor_parallel_size = tp
        self.world_size_across_dp = world_size_across_dp


class _FakeDPPortConfig:
    """Minimal stand-in for the ParallelConfig DP port source."""

    def __init__(self, dp_size, ports):
        self.data_parallel_size = dp_size
        self._data_parallel_master_port_list = list(ports)
        # Mimic ParallelConfig.__post_init__: the next pop fills
        # data_parallel_master_port first.
        self.data_parallel_master_port = (
            self._data_parallel_master_port_list.pop()
            if self._data_parallel_master_port_list
            else 29500
        )

    def get_next_dp_init_port(self):
        if self._data_parallel_master_port_list:
            return self._data_parallel_master_port_list.pop()
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += 1
        return answer


class TestRestartPortReservation(unittest.TestCase):
    def test_reservation_pins_one_shared_worker_world_port(self):
        # Mirror ParallelConfig.__post_init__ (one pop) + EngineCore
        # dp_group init (one pop): after both, an executor sees three
        # remaining ports; reserve 2 barrier + 1 worker.
        pc = _FakeDPPortConfig(
            dp_size=16, ports=[20001, 20002, 20003, 20004, 20005]
        )
        pc.get_next_dp_init_port()  # engine-core dp_group
        pool, worker_port = reserve_restart_ports(pc)
        self.assertEqual(worker_port, 20001)
        self.assertEqual(pc._data_parallel_master_port_list, [])
        self.assertEqual(pc.data_parallel_master_port, 20001)
        self.assertEqual(
            [pool.next_port() for _ in range(3)],
            [20003, 20002, 20003],
        )

    def test_every_restart_generation_reuses_the_same_port(self):
        pc = _FakeDPPortConfig(
            dp_size=16, ports=[20001, 20002, 20003, 20004, 20005]
        )
        pc.get_next_dp_init_port()
        _, worker_port = reserve_restart_ports(pc)
        # A scale-to-zero executor skips spawning workers during DP shrink;
        # pinning is idempotent and keeps its copy aligned with survivors.
        pin_worker_world_port(pc, worker_port)
        for _ in range(3):
            self.assertEqual(pc.get_next_dp_init_port(), worker_port)
            pin_worker_world_port(pc, worker_port)

    def test_repin_after_recover_post_init_regeneration(self):
        pc = _FakeDPPortConfig(
            dp_size=16, ports=[20001, 20002, 20003, 20004, 20005]
        )
        pc.get_next_dp_init_port()
        _, worker_port = reserve_restart_ports(pc)
        # RECOVER calls parallel_config.__post_init__(), which sees an empty
        # port list and regenerates a fresh (per-executor random) source.
        pc._data_parallel_master_port_list = [29999]
        pc.data_parallel_master_port = 29998
        pin_worker_world_port(pc, worker_port)
        self.assertEqual(pc._data_parallel_master_port_list, [])
        self.assertEqual(pc.data_parallel_master_port, worker_port)
        self.assertEqual(pc.get_next_dp_init_port(), worker_port)

    def test_single_dp_keeps_default_port_behaviour(self):
        pc = _FakeDPPortConfig(dp_size=1, ports=[])
        pool, worker_port = reserve_restart_ports(pc)
        self.assertIsNone(worker_port)
        self.assertEqual(len(pool), 0)


class TestBarrierPortPool(unittest.TestCase):
    def test_two_slot_pool_alternates(self):
        pool = BarrierPortPool([31001, 31002])
        # Degrade barrier uses the first port while the old dp_group is still
        # alive; RECOVER uses the second; the following degrade can reuse the
        # first because its group was destroyed when RECOVER was adopted.
        self.assertEqual(
            [pool.next_port() for _ in range(5)],
            [31001, 31002, 31001, 31002, 31001],
        )

    def test_empty_pool_fails_closed(self):
        with self.assertRaises(RuntimeError):
            BarrierPortPool().next_port()

    def test_ports_are_normalized_to_int(self):
        pool = BarrierPortPool(["31001", 31002])
        self.assertEqual(pool.next_port(), 31001)
        self.assertEqual(pool.next_port(), 31002)


if __name__ == "__main__":
    unittest.main()
