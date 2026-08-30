"""Patch for Mamba layer to support asymmetric TP.

This patch modifies the mamba_v2_sharded_weight_loader function to correctly
handle TP degradation scenarios (e.g., TP=4 -> TP=3).
"""
from vllm.logger import logger
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    LoaderFunction,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
    get_tp_asymmetric_shardings,
)




def mamba_v2_sharded_weight_loader_asymmetric(
    shard_spec,
    tp_size,
    tp_rank,
) -> LoaderFunction:
    """Patched weight loader that supports asymmetric TP."""

    # Get asymmetric TP config from vllm_config (computed lazily at loader creation time)
    tp_asymmetric_shardings = None
    original_tp_size = None

    try:
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        if zero_interrupt_config:
            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
            # Get original_tp_size from the same config location as get_tp_asymmetric_shardings
            engine_parallel_config_list = zero_interrupt_config.get('engine_parallel_config', None)
            executor_id = zero_interrupt_config.get('executor_id', '0')
            if engine_parallel_config_list:
                for config in engine_parallel_config_list:
                    if executor_id == config.get('executor_id', None):
                        original_tp_size = config.get("tp")
                        break
    except Exception as e:
        logger.warning(f"[mzm] Failed to get asymmetric TP config for Mamba: {e}")

    # Check if we have asymmetric TP config
    # For TP=4->TP=3 with shardings=[1,1,2]: sum=4, original_tp_size=4 ✓
    has_asymmetric = (
        tp_asymmetric_shardings is not None
        and original_tp_size is not None
        and len(tp_asymmetric_shardings) == tp_size
        and sum(tp_asymmetric_shardings) == original_tp_size
    )


    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        boundary, loaded_boundary = 0, 0

        for full_dim, extra, duplicate_groups in shard_spec:
            rank = 0 if duplicate_groups else tp_rank

            if has_asymmetric:
                # Asymmetric TP: use cumulative offsets
                base = full_dim // original_tp_size
                split_size = tp_asymmetric_shardings[rank]
                shard_size = split_size * base
                loaded_skip = sum(tp_asymmetric_shardings[:rank]) * base
            else:
                # Original symmetric TP logic
                shard_size = full_dim // tp_size
                loaded_skip = rank * shard_size

            loaded_start_idx = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)


            param.data[
                boundary : (boundary + take), ...
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)
            ]

            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader