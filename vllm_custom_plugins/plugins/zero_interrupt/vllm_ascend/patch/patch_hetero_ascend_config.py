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

    def _check(name: str, config: dict):
        tp_key = "tp_size"
        dp_key = "dp_size"
        if tp_key in config:
            config_tp = config[tp_key]
            if getattr(parallel_config, "is_heterogeneous_tp", False):
                # The extra config describes the logical remote pool layout
                # (e.g. prefill dp4/tp4).  The local instance is heterogeneous
                # (dp0 tp=3, dp1..3 tp=4), so accept the pool tp_size as long
                # as it matches one of the per-DP tp sizes; rank selection and
                # side-channel ports are derived from the real per-DP sizes.
                local_tp_sizes = sorted(
                    {
                        parallel_config.get_tp_size_for_dp(i)
                        for i in range(parallel_config.data_parallel_size)
                    }
                )
                if config_tp not in local_tp_sizes:
                    raise ValueError(
                        f"KV transfer '{name}' config has an incompatible "
                        f"tensor parallel size. Expected one of the "
                        f"heterogeneous per-DP tp sizes {local_tp_sizes}, "
                        f"but got {config_tp}."
                    )
            else:
                vllm_tp = parallel_config.tensor_parallel_size
                if config_tp != vllm_tp:
                    raise ValueError(
                        f"KV transfer '{name}' config has a conflicting tensor parallel size. "
                        f"Expected {vllm_tp}, but got {config_tp}."
                    )
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}."
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
    result = _ORIG_ENABLE_SP(
        vllm_config=vllm_config,
        enable_shared_expert_dp=enable_shared_expert_dp,
    )
    if not enable_shared_expert_dp:
        return result
    if result:
        return True

    cfg = vllm_config
    if cfg is None:
        try:
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
        except AssertionError:
            cfg = None
    if (
        cfg is not None
        and getattr(cfg.parallel_config, "is_heterogeneous_tp", False)
    ):
        # AscendConfig only uses this return value for the assertion.  The
        # real per-forward SP decision keeps using the unforced _ENABLE_SP.
        return True
    return result


def apply_hetero_ascend_config_patch():
    global _PATCHED, _ORIG_ENABLE_SP
    if _PATCHED:
        return
    import vllm_ascend.utils as mod

    _ORIG_ENABLE_SP = mod.enable_sp
    mod.enable_sp = _patched_enable_sp
    mod.check_kv_extra_config = _patched_check_kv_extra_config
    _PATCHED = True
