#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""DeepSeek-V4 heterogeneous-TP patches for vllm_ascend.models.deepseek_v4.

The patch is applied at plugin startup and only activates when
``ParallelConfig.is_heterogeneous_tp`` is True (which, in the
zero-interrupt flow, is derived from the decision-center strategy
``engine_parallel_config`` / ``tp_asymmetric_shardings``).

Reference implementation: hetero_cp ``feat/hetero-cp-dsa-cp`` on
vllm/vllm-ascend v0.23.0.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    sequence_parallel_chunk,
)

from vllm_custom_plugins.plugins.zero_interrupt.vllm.model_executor.layers.patch_linear import (
    ColumnParallelLinearAsymmetric,
    MergedColumnParallelLinearAsymmetric,
    RowParallelLinearAsymmetric,
)

_ORIG_DSV4_MLP_INIT = None
_ORIG_DSV4_MOE_INIT = None
_ORIG_DSV4_MOE_FORWARD = None
_ORIG_DSV4_ATTN_INIT = None
_ORIG_DSV4_LOAD_WEIGHTS = None
_PATCHED = False


def _get_vllm_config():
    return get_current_vllm_config_or_none()


def _is_hetero_tp() -> bool:
    cfg = _get_vllm_config()
    if cfg is None:
        return False
    return bool(
        getattr(cfg.parallel_config, "is_heterogeneous_tp", False)
    )


def _get_ratios() -> list[int] | None:
    cfg = _get_vllm_config()
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


def _patched_dsv4_mlp_init(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    swiglu_limit: float | None = None,
    quant_config=None,
    reduce_results: bool = True,
    is_sequence_parallel=False,
    prefix: str = "",
) -> None:
    # Under heterogeneous TP the shared expert must be replicated on every
    # TP rank: the hetero MoE path reduces the routed output with an EP
    # reduce-scatter and skips the final TP all-reduce, so a TP-sharded
    # shared expert would never get cross-rank reduced.
    if ".shared_experts" in prefix and _is_hetero_tp():
        is_sequence_parallel = True
    _ORIG_DSV4_MLP_INIT(
        self,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        swiglu_limit=swiglu_limit,
        quant_config=quant_config,
        reduce_results=reduce_results,
        is_sequence_parallel=is_sequence_parallel,
        prefix=prefix,
    )


def _patched_dsv4_moe_init(
    self,
    config,
    parallel_config,
    quant_config=None,
    prefix: str = "",
    is_draft_layer: bool = False,
) -> None:
    self.is_heterogeneous_tp = bool(
        getattr(parallel_config, "is_heterogeneous_tp", False)
    )
    _ORIG_DSV4_MOE_INIT(
        self,
        config=config,
        parallel_config=parallel_config,
        quant_config=quant_config,
        prefix=prefix,
        is_draft_layer=is_draft_layer,
    )

    if not self.is_heterogeneous_tp:
        return

    # Mirror determine_expert_map's remainder distribution: the first
    # ``remainder`` EP ranks own one extra expert. 256 experts over 15
    # heterogeneous-TP ranks is the canonical example.
    physical_base = self.n_physical_experts // self.ep_size
    physical_remainder = self.n_physical_experts % self.ep_size
    self.n_local_physical_experts = physical_base + (
        1 if self.ep_rank < physical_remainder else 0
    )
    self.physical_expert_start = (
        self.ep_rank * physical_base + min(self.ep_rank, physical_remainder)
    )
    self.physical_expert_end = (
        self.physical_expert_start + self.n_local_physical_experts
    )


def _patched_dsv4_moe_forward(self, hidden_states: torch.Tensor,
                              input_ids=None) -> torch.Tensor:
    m = _module()
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # With FlashComm1/2 the attention output is already TP-chunked.  Without
    # SP the attention returns the full replicated stream while the hetero
    # MoE comm still uses the EP gather/reduce-scatter path, which expects
    # one chunk per TP rank: chunk here and all-gather after FusedMoE.
    forward_ctx = get_forward_context()
    stream_is_chunked = bool(
        getattr(forward_ctx, "flash_comm_v1_enabled", False)
        or getattr(forward_ctx, "flashcomm_v2_enabled", False)
    )
    chunk_for_moe = self.is_sequence_parallel or (
        self.is_heterogeneous_tp and not stream_is_chunked
    )
    if chunk_for_moe:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
    else:
        router_logits = torch.nn.functional.linear(
            hidden_states.float(), self.gate.weight
        )
        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

    fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
    if fused_moe_out_is_tuple:
        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        if hidden_states.dtype != torch.float16:
            if not self.is_rocm_aiter_moe_enabled:
                if self.shared_experts is not None:
                    assert shared_output is not None
                    final_hidden_states = m.muls_add_triton(
                        final_hidden_states,
                        shared_output,
                        self.routed_scaling_factor,
                    )
                else:
                    final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states = m.muls_add_triton(
                shared_output,
                final_hidden_states,
                1.0 / self.routed_scaling_factor,
            )
    else:
        final_hidden_states = fused_moe_out

    if chunk_for_moe:
        final_hidden_states = tensor_model_parallel_all_gather(
            final_hidden_states, 0
        )
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1 and fused_moe_out_is_tuple:
        final_hidden_states = (
            self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        )

    return final_hidden_states.view(-1, hidden_dim)


def _patched_dsv4_attention_init(
    self,
    vllm_config,
    config,
    max_position_embeddings: int = 0,
    cache_config=None,
    quant_config=None,
    prefix: str = "",
    topk_indices_buffer: torch.Tensor | None = None,
) -> None:
    _ORIG_DSV4_ATTN_INIT(
        self,
        vllm_config=vllm_config,
        config=config,
        max_position_embeddings=max_position_embeddings,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=prefix,
        topk_indices_buffer=topk_indices_buffer,
    )

    ratios = _get_ratios()
    if ratios is None:
        return

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    self.n_local_heads = _partition_size(
        config.num_attention_heads, tp_rank, tp_size, ratios
    )
    self.n_local_groups = _partition_size(
        config.o_groups, tp_rank, tp_size, ratios
    )

    dsa_attn = getattr(self, "dsa_attn", None)
    if dsa_attn is not None:
        dsa_attn.n_local_heads = self.n_local_heads
        dsa_attn.n_local_groups = self.n_local_groups
        inner = getattr(dsa_attn, "dsa_attn", None)
        if inner is not None:
            inner.n_local_heads = self.n_local_heads
            inner.n_local_groups = self.n_local_groups

    # With DSA-CP the attn_sink is intentionally replicated with all heads;
    # only the non-CP path shards it across TP ranks.
    if getattr(self, "enable_dsa_cp", False):
        return

    # The original __init__ created attn_sink with the uniform size.  Replace
    # it and refresh every DSA reference to the parameter.
    old_sink = self.attn_sink
    new_sink = nn.Parameter(
        torch.empty(self.n_local_heads, dtype=old_sink.dtype)
    )
    if old_sink is not None and old_sink.numel() > 0:
        copied = min(old_sink.numel(), new_sink.numel())
        with torch.no_grad():
            new_sink.data[:copied] = old_sink.data[:copied]
    self.attn_sink = new_sink
    if dsa_attn is not None:
        dsa_attn.attn_sink = new_sink
        if inner is not None:
            inner.attn_sink = new_sink
        modules = getattr(dsa_attn, "dsa_modules", None)
        if modules is not None:
            modules.attn_sink = new_sink


def _patched_dsv4_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    m = _module()
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
            self.model,
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
            self.model,
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

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    # Attention heads per rank, with asymmetric sharding support.
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

    for name, loaded_weight in weights:
        spec_layer = m.get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue

        if not name.startswith("model"):
            name = f"model.{name}"

        if ".w1." in name:
            name = name.replace(".w1.", ".gate_proj.")
        if ".w2." in name:
            name = name.replace(".w2.", ".down_proj.")
        if ".w3." in name:
            name = name.replace(".w3.", ".up_proj.")
        if "model.head." in name and "model.lm_head." not in name:
            name = name.replace("model.head.", "lm_head.")
        if "model.lm_head." in name:
            name = name.replace("model.lm_head.", "lm_head.")
        if "embed." in name and "embed_token." not in name:
            name = name.replace("embed.", "embed_tokens.")
        if "attn" in name and "self_attn" not in name:
            name = name.replace(".attn.", ".self_attn.")
        if ".ffn." in name:
            name = name.replace(".ffn.", ".mlp.")
        if ".ffn_norm." in name:
            name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        if ".attn_norm." in name:
            name = name.replace(".attn_norm.", ".input_layernorm.")
        if name.endswith(".scale"):
            name = name.replace(".scale", ".weight_scale")
        if "rotary_emb.inv_freq" in name:
            continue
        if ".gate.bias" in name:
            name = name.replace(
                ".gate.bias", ".gate.e_score_correction_bias"
            )

        if "sink" in name:
            if m.is_pp_missing_parameter(name, self):
                continue
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
            if m.is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False

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

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    is_expert_weight = True
                    name_mapped = chunk_name.replace(
                        weight_name, param_name
                    )
                    if m.is_pp_missing_parameter(name_mapped, self):
                        continue
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
                    if m.is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        if not is_fusion_moe_shared_experts_layer:
            loaded_params.add(name)

    return loaded_params


def _module():
    import vllm_ascend.models.deepseek_v4 as m

    return m


def apply_deepseek_v4_hetero_patch():
    """Apply DeepSeek-V4 heterogeneous TP patches."""
    global _PATCHED, _ORIG_DSV4_MLP_INIT, _ORIG_DSV4_MOE_INIT
    global _ORIG_DSV4_MOE_FORWARD, _ORIG_DSV4_ATTN_INIT
    global _ORIG_DSV4_LOAD_WEIGHTS

    if _PATCHED:
        return
    m = _module()

    _ORIG_DSV4_MLP_INIT = m.DeepseekV2MLP.__init__
    _ORIG_DSV4_MOE_INIT = m.DeepseekV4MoE.__init__
    _ORIG_DSV4_MOE_FORWARD = m.DeepseekV4MoE.forward
    _ORIG_DSV4_ATTN_INIT = m.DeepseekV4Attention.__init__
    _ORIG_DSV4_LOAD_WEIGHTS = m.AscendDeepseekV4ForCausalLM.load_weights

    # Bind asymmetric-TP linear classes in the DeepSeek-V4 module namespace.
    # DeepSeek-V4 constructs these classes through module-global names, so
    # swapping the names is the surgical way to make all its MLP/MLA linear
    # layers honor tp_asymmetric_shardings (e.g. [2,1,1]).
    m.ColumnParallelLinear = ColumnParallelLinearAsymmetric
    m.MergedColumnParallelLinear = MergedColumnParallelLinearAsymmetric
    m.RowParallelLinear = RowParallelLinearAsymmetric

    m.DeepseekV2MLP.__init__ = _patched_dsv4_mlp_init
    m.DeepseekV4MoE.__init__ = _patched_dsv4_moe_init
    m.DeepseekV4MoE.forward = _patched_dsv4_moe_forward
    m.DeepseekV4Attention.__init__ = _patched_dsv4_attention_init
    m.AscendDeepseekV4ForCausalLM.load_weights = _patched_dsv4_load_weights

    _PATCHED = True
