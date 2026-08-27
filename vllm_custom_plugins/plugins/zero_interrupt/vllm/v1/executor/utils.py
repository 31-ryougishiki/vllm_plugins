import logging

logger = logging.getLogger("vllm_custom_plugins")


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

    # Legacy fallback: distribute old_tp heads evenly across new_tp ranks,
    # with the remainder assigned to the highest ranks.
    base = ori_tp // asym_tp
    remainder = ori_tp % asym_tp
    tp_asymmetric_shardings = [base] * asym_tp
    for i in range(remainder):
        tp_asymmetric_shardings[asym_tp - 1 - i] += 1
    return tp_asymmetric_shardings


def is_heterogeneous_restart(zero_interrupt_config) -> bool:
    """True when the strategy changes per-DP TP topology.

    A pure DP shrink/scale-to-zero (all active ranks keep the same tp_size)
    is NOT a heterogeneous TP restart and keeps using the legacy flow.
    """
    for conf in zero_interrupt_config.get("engine_parallel_config", []):
        new_tp = conf.get("new_tp", None)
        if new_tp is not None and int(new_tp) != int(conf.get("tp", new_tp)):
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
            ratios = _legacy_ratios(int(conf.get("tp")), tp_size)
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


def _legacy_ratios(ori_tp: int, asym_tp: int) -> list[int]:
    base = ori_tp // asym_tp
    remainder = ori_tp % asym_tp
    ratios = [base] * asym_tp
    for i in range(remainder):
        ratios[asym_tp - 1 - i] += 1
    return ratios
