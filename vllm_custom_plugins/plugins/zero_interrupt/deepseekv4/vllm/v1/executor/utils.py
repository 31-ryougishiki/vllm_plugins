import logging
from collections import deque
from collections.abc import Sequence

logger = logging.getLogger("vllm_custom_plugins")


def reserve_restart_ports(parallel_config):
    """Reserve the barrier ports and one fixed worker-world port.

    Returns ``(BarrierPortPool, worker_world_port)``.  The two barrier ports
    and the worker-world port are popped from ``parallel_config`` *in the
    executor process*, before any worker inherits a config copy.  Every DP
    executor runs this identical code path, so all of them reserve the same
    numeric ports.

    The worker-world port is then pinned back onto the config (see
    :func:`pin_worker_world_port`).  It must NOT be popped again inside each
    worker process: the per-worker config copies can drift apart across
    generations (a scale-to-zero executor spawns no worker during DP shrink,
    while the survivors consume one port), and a later RECOVER would make the
    restored executor join a different TCPStore port than its peers.
    """
    if parallel_config.data_parallel_size <= 1:
        return BarrierPortPool(), None

    barrier_ports = [
        parallel_config.get_next_dp_init_port() for _ in range(2)
    ]
    worker_world_port = parallel_config.get_next_dp_init_port()
    pin_worker_world_port(parallel_config, worker_world_port)
    return BarrierPortPool(barrier_ports), worker_world_port


def pin_worker_world_port(parallel_config, worker_world_port) -> None:
    """Point ``parallel_config``'s DP init-port source at one fixed port.

    Worker processes call ``get_next_dp_init_port()`` while building their
    global distributed world.  Clearing the port list and setting
    ``data_parallel_master_port`` to the reserved value makes every worker
    copy (and every restart generation) select the same TCPStore port.  Old
    workers are fully terminated before the next generation starts, so the
    port is free for reuse.
    """
    parallel_config._data_parallel_master_port_list = []
    parallel_config.data_parallel_master_port = int(worker_world_port)


class BarrierPortPool:
    """Round-robin pool of pre-reserved barrier rendezvous ports.

    Every full-restart barrier needs a fresh TCPStore port while the current
    ``engine_core.dp_group`` (which occupies the port used by the previous
    barrier) is still alive.  A two-slot pool is sufficient for sequential
    degrade/recover cycles: each successful barrier destroys the group built
    one generation earlier, so alternating between two ports guarantees the
    selected port is free.

    All DP executors, including scale-to-zero executors that skip the
    barrier itself, must call :meth:`next_port` once per full-restart
    generation.  This keeps every executor aligned on the same port for the
    next RECOVER barrier.
    """

    def __init__(self, ports: Sequence[int] | None = None) -> None:
        self._ports = deque(int(port) for port in (ports or []))

    def __len__(self) -> int:
        return len(self._ports)

    def next_port(self) -> int:
        """Return the next port and rotate it to the end of the queue."""
        if not self._ports:
            raise RuntimeError(
                "No reserved barrier ports left; initialize the pool before "
                "the first full-restart barrier."
            )
        port = int(self._ports[0])
        self._ports.rotate(-1)
        return port


def _find_engine_parallel_config(zero_interrupt_config):
    """Find the engine parallel config that belongs to this executor."""
    engine_parallel_config_list = zero_interrupt_config.get(
        "engine_parallel_config", None
    )
    if not engine_parallel_config_list:
        return None

    executor_id = zero_interrupt_config.get("executor_id", "0")
    config = None
    for engine_parallel_config in engine_parallel_config_list:
        if str(executor_id) == str(engine_parallel_config.get("executor_id", None)):
            config = engine_parallel_config
            break
    return config


def get_tp_asymmetric_shardings(zero_interrupt_config):
    """Get ``tp_asymmetric_shardings`` for the current executor/DP rank.

    Resolution order:
    1. Explicit ``tp_asymmetric_shardings`` from the strategy (preferred,
       e.g. DeepSeek-V4 DP4TP4 -> DP4TP(3,4,4,4) uses ``[2, 1, 1]`` on DP0).
    2. Derived evenly from old_tp/new_tp (legacy behaviour).
    3. Uniform ``[1] * tp_size`` when no asymmetric strategy is present.
    """
    config = _find_engine_parallel_config(zero_interrupt_config)
    if config is None:
        return []

    ori_tp = int(config.get("tp", 1))
    asym_tp = int(config.get("new_tp", config.get("tp", 1)))
    if asym_tp <= 0:
        # Scale-to-zero executor owns no TP ranks.
        return []

    explicit = config.get("tp_asymmetric_shardings", None)
    if explicit is None:
        # 兼容 strategy 顶层注入（ITSMultiprocExecutor._update_vllm_config）。
        explicit = zero_interrupt_config.get("tp_asymmetric_shardings", None)
    if explicit:
        if len(explicit) != asym_tp or any(int(r) <= 0 for r in explicit):
            raise ValueError(
                "tp_asymmetric_shardings must contain exactly one positive "
                f"entry per new_tp rank. new_tp={asym_tp}, got {explicit}."
            )
        return [int(r) for r in explicit]

    if asym_tp == ori_tp:
        return [1] * asym_tp

    # Legacy fallback: distribute old_tp heads evenly across new_tp ranks.
    # DecisionMakingCenter does not send tp_asymmetric_shardings, so the
    # plugin has to derive it. DeepSeek-V4 golden topology is
    # old_tp=4 -> new_tp=3 => [2, 1, 1] (rank0 takes the remainder); assign
    # the remainder to the LOWEST ranks to match hetero_cp.
    base = ori_tp // asym_tp
    remainder = ori_tp % asym_tp
    tp_asymmetric_shardings = [base] * asym_tp
    for i in range(remainder):
        tp_asymmetric_shardings[i] += 1
    return tp_asymmetric_shardings


def is_heterogeneous_restart(zero_interrupt_config) -> bool:
    """True when the strategy changes per-DP TP topology.

    A pure DP shrink/scale-to-zero (all active ranks keep the same tp_size)
    is NOT a heterogeneous TP restart and keeps using the legacy flow.
    """
    for conf in zero_interrupt_config.get("engine_parallel_config", []):
        new_tp = conf.get("new_tp", None)
        if new_tp is None:
            continue
        new_tp = int(new_tp)
        if new_tp <= 0:
            # Scale-to-zero executor: it owns no ranks and does not make the
            # restart heterogeneous (e.g. decode DP16TP1 -> DP15TP1).
            continue
        if new_tp != int(conf.get("tp", new_tp)):
            return True
        if conf.get("tp_asymmetric_shardings"):
            return True
    return False


def get_heterogeneous_dp_config(zero_interrupt_config):
    """Build the full per-DP-rank heterogeneous TP config list.

    Returns a list of dicts sorted by data_parallel_rank::

        [
            {"dp_rank": 0, "tp_size": 3, "tp_sharding_ratios": [2, 1, 1]},
            {"dp_rank": 1, "tp_size": 4},
            ...
        ]
    """
    engine_parallel_config_list = zero_interrupt_config.get(
        "engine_parallel_config", []
    )
    configs = []
    for conf in engine_parallel_config_list:
        new_tp = conf.get("new_tp", None)
        tp_size = new_tp if new_tp is not None else conf.get("tp", 1)
        tp_size = int(tp_size)
        if tp_size <= 0:
            # Scale-to-zero executor; it owns no ranks.
            continue
        ratios = conf.get("tp_asymmetric_shardings", None)
        if ratios is None and new_tp is not None and new_tp != conf.get("tp"):
            # Delegate to the same fallback used by the weight loaders so the
            # remainder direction can never drift apart again.
            ratios = get_tp_asymmetric_shardings(
                {
                    "executor_id": conf.get("executor_id"),
                    "engine_parallel_config": [conf],
                }
            )
        configs.append(
            {
                "dp_rank": int(conf.get("data_parallel_rank", 0)),
                "tp_size": tp_size,
                "tp_sharding_ratios": (
                    [int(r) for r in ratios] if ratios else None
                ),
            }
        )

    configs.sort(key=lambda c: c["dp_rank"])
    # After scale-to-zero executors are skipped, renumber the surviving DP
    # ranks contiguously from 0. This is also the global-rank layout used by
    # torch.distributed init and by get_rank_offset_for_dp.
    for new_rank, cfg in enumerate(configs):
        cfg["dp_rank"] = new_rank
    if not configs:
        raise ValueError(
            "engine_parallel_config has no active executor for "
            "heterogeneous restart."
        )
    return configs


def get_global_world_size(zero_interrupt_config):
    """Total global world size across all heterogeneous DP ranks."""
    configs = get_heterogeneous_dp_config(zero_interrupt_config)
    return sum(c["tp_size"] for c in configs)


def get_global_start_rank(zero_interrupt_config):
    """Global starting rank of the current executor's workers."""
    # ``engine_parallel_config`` may arrive in any order.  Global offsets are
    # cumulative in DP-rank order, so sort by data_parallel_rank first
    # (scale-to-zero executors contribute zero ranks).
    executor_id = zero_interrupt_config.get("executor_id", "0")
    ordered = sorted(
        zero_interrupt_config.get("engine_parallel_config", []),
        key=lambda conf: int(conf.get("data_parallel_rank", 0) or 0),
    )
    offset = 0
    for conf in ordered:
        if str(conf.get("executor_id", None)) == str(executor_id):
            return offset
        new_tp = conf.get("new_tp", None)
        tp_size = new_tp if new_tp is not None else conf.get("tp", 1)
        offset += max(int(tp_size), 0)
    raise ValueError(
        f"Cannot find executor_id={executor_id} in engine_parallel_config"
    )


def _original_dp_rank(conf: dict) -> int:
    """Return the pre-restart DP rank of one engine_parallel_config entry."""
    rank = conf.get("data_parallel_rank", None)
    if rank is None:
        rank = conf.get("executor_id", None)
    try:
        return int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"engine_parallel_config entry has no usable data_parallel_rank: "
            f"{conf!r}"
        ) from exc


def get_scale_to_zero_dp_ranks(zero_interrupt_config) -> set[int]:
    """Return the pre-restart DP ranks that a strategy shrinks to zero.

    A scale-to-zero executor is one whose ``new_tp`` and ``new_dp`` are both
    zero (e.g. decode DP16TP1 -> DP15TP1 executor 15). It owns no worker
    ranks after the restart and therefore must not participate in the
    pre-restart full-restart barrier.
    """
    scale_to_zero_ranks = set()
    for conf in zero_interrupt_config.get("engine_parallel_config", []):
        new_tp = conf.get("new_tp", None)
        new_dp = conf.get("new_dp", None)
        if new_tp is None:
            new_tp = conf.get("tp", 1)
        if new_dp is None:
            new_dp = conf.get("dp", 1)
        if int(new_tp) == 0 and int(new_dp) == 0:
            scale_to_zero_ranks.add(_original_dp_rank(conf))
    return scale_to_zero_ranks


def strategy_matches_current_topology(
    zero_interrupt_config,
    current_tp: int,
    current_dp: int,
    current_heterogeneous_dp_config=None,
) -> bool:
    """Return True when a strategy would not change THIS executor's topology.

    Used as the idempotency boundary for repeated DEGRADE/PD_REBUILD
    strategies.  A repeated full-restart strategy must not restart workers
    again without a barrier, but must also not silently drop a strategy whose
    target topology differs from the executor's current one.
    """
    executor_id = str(zero_interrupt_config.get("executor_id", ""))
    conf = _find_engine_parallel_config(
        {"executor_id": executor_id,
         "engine_parallel_config": zero_interrupt_config.get(
             "engine_parallel_config", [])}
    )
    if conf is None:
        return False

    new_tp = conf.get("new_tp", None)
    new_dp = conf.get("new_dp", None)
    if new_tp is None:
        new_tp = conf.get("tp", 1)
    if new_dp is None:
        new_dp = conf.get("dp", 1)
    if int(current_tp) != int(new_tp) or int(current_dp) != int(new_dp):
        return False

    target_is_hetero = is_heterogeneous_restart(zero_interrupt_config)
    current_is_hetero = current_heterogeneous_dp_config is not None
    if current_is_hetero != target_is_hetero:
        return False
    if not target_is_hetero:
        return True

    target_configs = get_heterogeneous_dp_config(zero_interrupt_config)
    current_configs = [
        {
            "dp_rank": int(cfg.dp_rank),
            "tp_size": int(cfg.tp_size),
            "tp_sharding_ratios": (
                [int(r) for r in cfg.tp_sharding_ratios]
                if cfg.tp_sharding_ratios is not None
                else None
            ),
        }
        for cfg in current_heterogeneous_dp_config
    ]
    return current_configs == target_configs


def recover_requires_full_restart(
    backup: dict,
    current_tp: int,
    current_dp: int,
    current_is_heterogeneous: bool,
    target_dp: int,
) -> bool:
    """Decide whether a RECOVER strategy needs the all-executor barrier.

    Pure DP recovery (DP15TP1 -> DP16TP1) has no TP change. Only comparing
    the backed-up TP with the current TP would mark it as a single-executor
    restart: the healthy executors would kill the old 15-rank worker world
    while the recovered executor is still waiting on the old 16-rank
    barrier, and the new 16-rank ``init_process_group`` would wait forever.
    """
    backup_tp = backup.get("tensor_parallel_size")
    backup_dp = backup.get("data_parallel_size")
    return bool(
        current_is_heterogeneous
        or (
            backup_tp is not None
            and int(backup_tp) != int(current_tp)
        )
        or (
            backup_dp is not None
            and int(backup_dp) != int(current_dp)
        )
        or int(target_dp or 0) != int(current_dp or 0)
    )


def get_renumbered_dp_rank(zero_interrupt_config) -> int | None:
    """Return this executor's contiguous rank among surviving DP executors.

    Mirrors the survivor renumbering used by
    :func:`get_surviving_dp_barrier_geometry`, but is usable by
    ``_get_engine_parallel_config`` for pure DP shrink strategies as well
    (where ``heterogeneous_dp_config`` stays None but the barrier still
    renumbers the survivors).  A scale-to-zero executor returns ``None`` so
    the caller can keep its original rank while it has no workers.
    """
    executor_id = str(zero_interrupt_config.get("executor_id", ""))
    ordered = sorted(
        zero_interrupt_config.get("engine_parallel_config", []),
        key=_original_dp_rank,
    )
    active_rank = 0
    for conf in ordered:
        new_tp = conf.get("new_tp", None)
        new_dp = conf.get("new_dp", None)
        if new_tp is None:
            new_tp = conf.get("tp", 1)
        if new_dp is None:
            new_dp = conf.get("dp", 1)
        if int(new_tp) == 0 and int(new_dp) == 0:
            if str(conf.get("executor_id", None)) == executor_id:
                return None
            continue
        if str(conf.get("executor_id", None)) == executor_id:
            return active_rank
        active_rank += 1

    raise ValueError(
        f"executor_id={executor_id!r} was not found in "
        "engine_parallel_config."
    )


def get_pd_scheduler_connector_topology(
    parallel_config,
    kv_port: int,
) -> tuple[int, int, int]:
    """Return the PD scheduler-connector topology after a restart.

    The scheduler connector survives worker restarts, so every deployment
    must refresh ``(side_channel_port, tp_size, max_device_id)`` from the
    post-restart ``parallel_config``.  In particular, RECOVER from
    DP4TP(3,4,4,4) back to DP4TP4 flips ``is_heterogeneous_tp`` to False;
    only refreshing inside the hetero branch would leave the connector
    advertising the old tp=3 / 15-device layout to the decode pool.
    """
    if getattr(parallel_config, "is_heterogeneous_tp", False):
        port_offset = parallel_config.get_rank_offset_for_dp(
            parallel_config.data_parallel_rank
        )
    else:
        pcp_size = parallel_config.prefill_context_parallel_size
        port_offset = (
            parallel_config.data_parallel_rank
            * parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
            * pcp_size
        )
    return (
        kv_port + port_offset,
        parallel_config.tensor_parallel_size,
        parallel_config.world_size_across_dp,
    )


def get_surviving_dp_barrier_geometry(
    zero_interrupt_config,
) -> tuple[int, int, set[int]]:
    """Compute the pre-restart rendezvous geometry for the surviving ranks.

    Returns ``(surviving_dp_size, surviving_dp_rank, scale_to_zero_ranks)``:

    - ``surviving_dp_size`` is the number of executors that keep workers;
    - ``surviving_dp_rank`` is this executor's rank in the contiguous
      renumbered survivor group (used by the stateless gloo barrier and by
      the replacement engine-core ``dp_group``);
    - ``scale_to_zero_ranks`` are the original DP ranks excluded from the
      barrier.

    For the scale-to-zero executor itself the first two values are ``0`` so
    the caller can skip the barrier.
    """
    scale_to_zero_ranks = get_scale_to_zero_dp_ranks(zero_interrupt_config)
    executor_id = str(zero_interrupt_config.get("executor_id", ""))

    ordered = sorted(
        zero_interrupt_config.get("engine_parallel_config", []),
        key=_original_dp_rank,
    )
    survivors = []
    for conf in ordered:
        new_tp = conf.get("new_tp", None)
        new_dp = conf.get("new_dp", None)
        if new_tp is None:
            new_tp = conf.get("tp", 1)
        if new_dp is None:
            new_dp = conf.get("dp", 1)
        if int(new_tp) == 0 and int(new_dp) == 0:
            continue
        survivors.append(conf)

    # Validate that the strategy is internally consistent: every active
    # executor must agree on the new DP size and it must match the number of
    # active entries. Otherwise the survivor group would hang waiting for a
    # rank that the decision center did not intend to keep.
    expected_new_dp = {
        int(conf["new_dp"])
        for conf in survivors
        if conf.get("new_dp", None) is not None
    }
    if expected_new_dp and expected_new_dp != {len(survivors)}:
        raise ValueError(
            "engine_parallel_config is inconsistent: active executors "
            f"declare new_dp={sorted(expected_new_dp)} but only "
            f"{len(survivors)} active entries were found."
        )

    for new_rank, conf in enumerate(survivors):
        if str(conf.get("executor_id", None)) == executor_id:
            return len(survivors), new_rank, scale_to_zero_ranks

    # The current executor was not among the survivors. Accept that only
    # when it is one of the scale-to-zero entries; otherwise the strategy is
    # missing this executor entirely.
    if any(
        str(conf.get("executor_id", None)) == executor_id
        for conf in zero_interrupt_config.get("engine_parallel_config", [])
        if _original_dp_rank(conf) in scale_to_zero_ranks
    ):
        return 0, 0, scale_to_zero_ranks

    raise ValueError(
        f"executor_id={executor_id!r} was not found in "
        "engine_parallel_config."
    )
