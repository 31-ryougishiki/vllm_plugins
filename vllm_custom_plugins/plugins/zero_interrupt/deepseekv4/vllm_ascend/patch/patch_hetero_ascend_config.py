#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""AscendConfig/enable_sp patch for heterogeneous TP.

``AscendConfig`` asserts ``enable_sp(..., enable_shared_expert_dp=True)``.
That helper force-enables FlashComm1 globally.  Under heterogeneous TP the
MoE EP path already implements shared-expert data parallelism, so do not
change the global SP state; only satisfy the assertion when it is reached.
"""

_PATCHED = False
_ORIG_ENABLE_SP = None


def _patched_check_kv_extra_config(vllm_config):
    parallel_config = vllm_config.parallel_config

    def _allowed_parallel_values(key: str) -> set[int]:
        """Collect acceptable pool sizes for a restart/degrade scenario.

        KV extra config describes the ORIGINAL remote pool layout.  After a
        heterogeneous restart (P TP 4->3) or a decode degradation
        (D DP 16->15) the local parallel config intentionally differs from
        that original layout, so the check must accept the current value,
        every per-DP value and every pre/post value carried by the
        zero-interrupt strategy.
        """
        attr = {
            "tp_size": "tensor_parallel_size",
            "dp_size": "data_parallel_size",
        }[key]
        allowed = set()
        current = getattr(parallel_config, attr, None)
        if current is not None:
            allowed.add(int(current))

        if key == "tp_size" and getattr(
            parallel_config, "is_heterogeneous_tp", False
        ):
            for dp_rank in range(parallel_config.data_parallel_size):
                allowed.add(
                    int(parallel_config.get_tp_size_for_dp(dp_rank))
                )

        additional = getattr(vllm_config, "additional_config", None) or {}
        zi_config = additional.get("zero_interrupt_config", {}) or {}
        cfg_key = key.removesuffix("_size")  # tp_size -> tp, dp_size -> dp
        for cfg in zi_config.get("engine_parallel_config", []) or []:
            for field in (cfg_key, f"new_{cfg_key}"):
                value = cfg.get(field)
                if value is not None and int(value) > 0:
                    allowed.add(int(value))
        return allowed

    def _check(name: str, config: dict):
        for key, attr in (
            ("tp_size", "tensor_parallel_size"),
            ("dp_size", "data_parallel_size"),
        ):
            if key not in config:
                continue
            config_value = int(config[key])
            allowed = _allowed_parallel_values(key)
            if config_value in allowed:
                continue
            raise ValueError(
                f"KV transfer '{name}' config has a conflicting "
                f"{attr}. Expected one of {sorted(allowed)}, "
                f"but got {config_value}."
            )

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return
    if kv_transfer_config.is_kv_producer:
        _check(
            "prefill",
            kv_transfer_config.get_from_extra_config("prefill", {}),
        )
    if kv_transfer_config.is_kv_consumer:
        _check(
            "decode",
            kv_transfer_config.get_from_extra_config("decode", {}),
        )


def _patched_enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False):
    if enable_shared_expert_dp:
        # The original helper force-enables the module-level FlashComm1 flag
        # as a side effect.  Under heterogeneous TP the MoE EP path already
        # implements shared-expert data parallelism; satisfy the caller
        # (AscendConfig's assertion) WITHOUT mutating the global SP state.
        cfg = vllm_config
        if cfg is None:
            try:
                from vllm.config import get_current_vllm_config_or_none

                cfg = get_current_vllm_config_or_none()
            except AssertionError:
                cfg = None
        if (
            cfg is not None
            and getattr(cfg.parallel_config, "is_heterogeneous_tp", False)
        ):
            return True

    return _ORIG_ENABLE_SP(
        vllm_config=vllm_config,
        enable_shared_expert_dp=enable_shared_expert_dp,
    )


def apply_hetero_ascend_config_patch():
    global _PATCHED, _ORIG_ENABLE_SP
    if _PATCHED:
        return
    import vllm_ascend.utils as mod

    _ORIG_ENABLE_SP = mod.enable_sp
    mod.enable_sp = _patched_enable_sp
    mod.check_kv_extra_config = _patched_check_kv_extra_config

    # ``vllm_ascend.platform`` imports ``check_kv_extra_config`` by name at
    # module load and calls its own reference.  Patch that reference as well,
    # otherwise the strict homogeneous TP check still runs and rejects the
    # DP0 tp=3 prefill after a heterogeneous restart.
    try:
        import vllm_ascend.platform as platform_mod

        platform_mod.check_kv_extra_config = _patched_check_kv_extra_config
    except Exception:
        pass
    _PATCHED = True
