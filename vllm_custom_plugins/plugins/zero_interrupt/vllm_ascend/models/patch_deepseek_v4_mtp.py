#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""DeepSeek-V4 MTP heterogeneous-TP patch.

The MTP block reuses ``DeepseekV2DecoderLayer`` from
``vllm_ascend.models.deepseek_v4`` (which is patched separately), so only the
MTP ``load_weights`` attention-sink slicing needs the asymmetric head
partition.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

_ORIG_MTP_LOAD_WEIGHTS = None
_PATCHED = False


def _get_ratios() -> list[int] | None:
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return None
    pc = cfg.parallel_config
    if not getattr(pc, "is_heterogeneous_tp", False):
        return None
    ratios = pc.get_sharding_ratios_for_dp(pc.data_parallel_rank)
    return [int(r) for r in ratios] if ratios else None


def _partition_size(total: int, rank: int, size: int,
                    ratios: list[int] | None) -> int:
    if size == 1:
        return total
    if ratios is None:
        return total // size
    total_ratio = sum(ratios)
    sizes = [total * r // total_ratio for r in ratios]
    sizes[-1] += total - sum(sizes)
    return sizes[rank]


def _partition_offset(total: int, rank: int, size: int,
                      ratios: list[int] | None) -> int:
    if size == 1:
        return 0
    if ratios is None:
        return rank * (total // size)
    total_ratio = sum(ratios)
    return sum(total * ratios[i] // total_ratio for i in range(rank))


def _patched_mtp_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    import vllm_ascend.models.deepseek_v4_mtp as m

    rocm_aiter_moe_shared_expert_enabled = (
        m.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
    )
    rocm_aiter_moe_shared_expert_enabled = getattr(
        m.get_ascend_config(), "mix_placement", False
    )
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    if m.vllm_version_is("0.23.0"):
        expert_params_mapping = m.FusedMoE.make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )
    else:
        expert_params_mapping = m.fused_moe_make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    ratios = _get_ratios()
    if ratios is not None:
        heads_per_rank = _partition_size(
            self.config.num_attention_heads, tp_rank, tp_size, ratios
        )
        head_start = _partition_offset(
            self.config.num_attention_heads, tp_rank, tp_size, ratios
        )
    else:
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "fp8"
        ):
            if name == "embed.weight":
                name = "mtp.0.emb.tok_emb.weight"
            if name == "head.weight":
                name = "mtp.0.head.weight"

        spec_layer = m.get_spec_layer_idx_from_weight_name(
            self.config, name
        )
        if spec_layer is None:
            continue

        assert "mtp.0." in name
        if ".emb.tok_emb." in name:
            name = name.replace("mtp.0.", "model.")
        elif self.no_mtp_block_in_name(name):
            name = name.replace("mtp.0.", "model.layers.0.")
        else:
            name = name.replace("mtp.0.", "model.layers.0.mtp_block.")

        if ".w1." in name:
            name = name.replace(".w1.", ".gate_proj.")
        if ".w2." in name:
            name = name.replace(".w2.", ".down_proj.")
        if ".w3." in name:
            name = name.replace(".w3.", ".up_proj.")
        if name.endswith(".scale"):
            name = name.replace(".scale", ".weight_scale")
        if ".head." in name:
            name = name.replace(".head.", ".shared_head.head.")
        if ".norm." in name:
            name = name.replace(".norm.", ".shared_head.norm.")
        if ".emb.tok_emb." in name:
            name = name.replace(".emb.tok_emb.", ".embed_tokens.")
        if "attn" in name and "self_attn" not in name:
            name = name.replace(".attn.", ".self_attn.")
        if ".ffn." in name:
            name = name.replace(".ffn.", ".mlp.")
        if ".ffn_norm." in name:
            name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        if ".attn_norm." in name:
            name = name.replace(".attn_norm.", ".input_layernorm.")
        if ".gate.bias" in name:
            name = name.replace(
                ".gate.bias", ".gate.e_score_correction_bias"
            )

        if "sink" in name:
            param = params_dict[name]
            if m.enable_dsa_cp():
                param.data.copy_(loaded_weight)
            else:
                narrow_weight = loaded_weight.narrow(
                    0, head_start, heads_per_rank
                )
                param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue

        is_fusion_moe_shared_experts_layer = (
            rocm_aiter_moe_shared_expert_enabled
            and ("mlp.shared_experts" in name)
        )
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if is_fusion_moe_shared_experts_layer:
                continue
            name_mapped = name.replace(weight_name, param_name)
            if (
                param_name == "fused_qkv_a_proj"
                and name_mapped not in params_dict
            ):
                continue
            name = name_mapped
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            num_chunks = 1
            if is_fusion_moe_shared_experts_layer:
                num_chunks = (
                    getattr(self.config, "n_shared_experts", 1) or 1
                )
                split_dim = 1 if "down_proj.weight" in name else 0
                total = loaded_weight.shape[split_dim]
                assert total % num_chunks == 0
                chunk_size = total // num_chunks

            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                if is_fusion_moe_shared_experts_layer:
                    if split_dim == 0:
                        weight_to_load = loaded_weight[
                            j * chunk_size:(j + 1) * chunk_size, :
                        ]
                    else:
                        weight_to_load = loaded_weight[
                            :, j * chunk_size:(j + 1) * chunk_size
                        ]
                    chunk_name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts + j}",
                    )

                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue
                    is_expert_weight = True
                    name_mapped = chunk_name.replace(
                        weight_name, param_name
                    )
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        if not is_fusion_moe_shared_experts_layer:
                            name = name_mapped
                        else:
                            loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = m.maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        if not is_fusion_moe_shared_experts_layer:
            loaded_params.add(name)
    return loaded_params


def apply_deepseek_v4_mtp_hetero_patch():
    global _PATCHED, _ORIG_MTP_LOAD_WEIGHTS
    if _PATCHED:
        return
    import vllm_ascend.models.deepseek_v4_mtp as m

    _ORIG_MTP_LOAD_WEIGHTS = m.DeepSeekV4MTP.load_weights
    m.DeepSeekV4MTP.load_weights = _patched_mtp_load_weights
    _PATCHED = True
