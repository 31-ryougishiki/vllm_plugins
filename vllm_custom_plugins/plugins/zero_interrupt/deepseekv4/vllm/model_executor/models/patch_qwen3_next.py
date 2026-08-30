# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3Next and Qwen3.5 asymmetric TP patch for zero-interrupt inference.

This module provides asymmetric tensor parallel support for Qwen3Next base model
and Qwen3.5 GatedDeltaNet (linear attention) layers.

Architecture:
- Qwen3NextModel → Qwen3NextDecoderLayer → Qwen3NextAttention (full_attention)
                                                 → Qwen3NextGatedDeltaNet (linear_attention)
- Qwen3.5 uses Qwen3_5GatedDeltaNet which has different in_proj_qkvz/in_proj_ba layouts
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention as OrigQwen3NextAttention,
    Qwen3NextGatedDeltaNet as OrigQwen3NextGatedDeltaNet,
    Qwen3NextModel as OrigQwen3NextModel,
    Qwen3NextDecoderLayer as OrigQwen3NextDecoderLayer,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.layers.patch_linear import (
    MergedColumnParallelLinearAsymmetric,
    QKVParallelLinearAsymmetric,
    RowParallelLinearAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.models.patch_qwen2 import (
    Qwen2ModelAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
    get_tp_asymmetric_shardings,
)
from vllm_ascend.patch.worker.patch_qwen3_5 import AscendQwen3NextAttention,AscendQwen3_5DecoderLayer
from vllm_ascend.patch.worker.patch_qwen3_next import AscendQwen3Next_GatedDeltaNet



class Qwen3NextModelAsymmetric(Qwen2ModelAsymmetric):
    """Qwen3Next base model with asymmetric TP support.

    Inherits from Qwen2ModelAsymmetric but uses Qwen3NextDecoderLayer
    which supports both linear_attention and full_attention layer types.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3NextDecoderLayerAsymmetric,
        )


class Qwen3NextAttentionAsymmetric(OrigQwen3NextAttention):
    """Qwen3Next Attention with asymmetric TP partitioning.

    Key differences from Qwen3NextAttention:
    - Uses QKVParallelLinearAsymmetric for qkv_proj
    - Uses RowParallelLinearAsymmetric for o_proj
    - Handles non-divisible head distribution across TP ranks
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        if not asym:
            # For symmetric TP, just call parent's full __init__
            OrigQwen3NextAttention.__init__(
                self,
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - do custom initialization
        nn.Module.__init__(self)
        text_config = config.text_config
        # [mzm] Use text_config because AscendQwen3NextAttention.forward expects
        # self.config.rms_norm_eps which exists in text_config, not in top-level config
        self.config = text_config
        self.hidden_size = text_config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = text_config.num_attention_heads
        self.total_num_kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim or (self.hidden_size // self.total_num_heads)
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            text_config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(text_config, "attn_output_gate", True)

        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
        world_split_size = sum(tp_asymmetric_shardings)
        split_size = tp_asymmetric_shardings[tp_rank]

        self.num_heads = self.total_num_heads * split_size // world_split_size
        self.num_kv_heads = max(
            1, self.total_num_kv_heads * split_size // world_split_size
        )



        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # QKV projection with asymmetric TP
        self.qkv_proj = QKVParallelLinearAsymmetric(
            text_config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(text_config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        # Output projection with asymmetric TP
        self.o_proj = RowParallelLinearAsymmetric(
            self.total_num_heads * self.head_dim,
            text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=text_config.max_position_embeddings,
            rope_parameters=text_config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": None,  # Will be set by the caller
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        # Use Qwen3NextRMSNorm from the parent module
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=text_config.rms_norm_eps)



class Qwen3NextGatedDeltaNetAsymmetric(OrigQwen3NextGatedDeltaNet):
    """GatedDeltaNet (linear attention) with asymmetric TP.

    Key differences from Qwen3NextGatedDeltaNet:
    - Uses MergedColumnParallelLinearAsymmetric for in_proj_qkvz (4 outputs: Q,K,V,Z)
    - Uses MergedColumnParallelLinearAsymmetric for in_proj_ba (2 outputs: B,A)
    - Uses RowParallelLinearAsymmetric for out_proj
    - Handles non-divisible head distribution across TP ranks
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.num_v_heads = text_config.linear_num_value_heads
        self.num_k_heads = text_config.linear_num_key_heads
        self.head_k_dim = text_config.linear_key_head_dim
        self.head_v_dim = text_config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = None  # Will be set by the caller
        self.activation = config.hidden_act
        from transformers.activations import ACT2FN
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # Check for asymmetric TP
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        if asym:
            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
            world_split_size = sum(tp_asymmetric_shardings)
            split_size = tp_asymmetric_shardings[self.tp_rank]
        else:
            tp_asymmetric_shardings = None
            world_split_size = self.tp_size
            split_size = 1

        self.asym = asym
        self.tp_asymmetric_shardings = tp_asymmetric_shardings
        self.world_split_size = world_split_size

        # QKV - compute local sizes for asymmetric
        self.local_num_v_heads = self.num_v_heads * split_size // world_split_size
        self.local_num_k_heads = max(
            1, self.num_k_heads * split_size // world_split_size
        )

        # Conv1d (same as parent, no TP sharding on conv dim)
        from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.layers.patch_linear import ColumnParallelLinearAsymmetric

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinearAsymmetric(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Input projections with asymmetric TP
        if asym:
            # in_proj_qkvz: 4 outputs [K, K, V, V] for Qwen3.5 layout
            self.in_proj_qkvz = MergedColumnParallelLinearAsymmetric(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )
            # in_proj_ba: 2 outputs [B, A]
            self.in_proj_ba = MergedColumnParallelLinearAsymmetric(
                input_size=self.hidden_size,
                output_sizes=[self.num_v_heads, self.num_v_heads],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )
        else:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = self.create_ba_proj(
                hidden_size=self.hidden_size,
                num_v_heads=self.num_v_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
            )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # Time step projection parameters (same sharding as parent)
        from vllm.distributed import divide

        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(divide(self.num_v_heads, self.tp_size)),
        )



        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # RMSNormGated for output
        from vllm.model_executor.layers.layernorm import RMSNormGated

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
        )

        # Output projection with asymmetric TP
        if asym:
            self.out_proj = RowParallelLinearAsymmetric(
                self.value_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )
        else:
            self.out_proj = RowParallelLinear(
                self.value_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
            )

        self.chunk_gated_delta_rule = OrigQwen3NextGatedDeltaNet.__dict__["chunk_gated_delta_rule"].__get__(self)
        self.enable_packed_recurrent_decode = False  # Default, inherited classes may override

        # Register with compilation static forward context
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Compute state shape with asymmetric TP support.

        This override avoids calling MambaStateShapeCalculator.gated_delta_net_state_shape
        which doesn't have access to vllm_config in all contexts.

        Key insight: In asymmetric TP scenarios (degrade/recover), the conv1d weight is
        loaded in full (conv_dim), but each rank processes only its portion of mixed_qkv.
        The KV cache conv_dim must match what each rank actually processes, which is
        determined by split_size relative to world_split_size.
        """
        if self.asym and self.tp_asymmetric_shardings is not None:
            rank_split_size = self.tp_asymmetric_shardings[self.tp_rank]
            rank_proportion = rank_split_size / self.world_split_size
            conv_state_dim = int(self.conv_dim * rank_proportion)

            conv_state_shape = (
                conv_state_dim,
                self.conv_kernel_size - 1 + self.num_spec,
            )
        else:
            from vllm.distributed.utils import divide

            conv_state_shape = (
                divide(self.conv_dim, self.tp_size),
                self.conv_kernel_size - 1 + self.num_spec,
            )

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        temporal_state_shape = (
            self.local_num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
        )

        return conv_state_shape, temporal_state_shape


# [mzm] Wrapper for GatedDeltaNet forward to add logging
def _gated_delta_forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
    result = AscendQwen3Next_GatedDeltaNet.forward(self, hidden_states, output)
    return result

Qwen3NextGatedDeltaNetAsymmetric.forward = _gated_delta_forward
Qwen3NextGatedDeltaNetAsymmetric._forward_core = AscendQwen3Next_GatedDeltaNet._forward_core


class Qwen3NextDecoderLayerAsymmetric(OrigQwen3NextDecoderLayer):
    """Decoder layer that selects asymmetric attention based on layer_type.

    For linear_attention layers: uses Qwen3NextGatedDeltaNetAsymmetric
    For full_attention layers: uses Qwen3NextAttentionAsymmetric
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = None  # Will be set by the caller

        from vllm.model_executor.models.utils import extract_layer_index

        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNetAsymmetric(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttentionAsymmetric(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")
        text_config = config.text_config
        # MLP (same as parent - uses Qwen3NextMLP or Qwen3NextSparseMoeBlock)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP

            self.mlp = Qwen3NextMLP(
                hidden_size=text_config.hidden_size,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        # Layer norms (same as parent)
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm

        self.input_layernorm = Qwen3NextRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    text_config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    text_config.hidden_size,
                ),
            )


# [mzm] Wrapper for DecoderLayer forward to trace hidden_states corruption
# This must be after Qwen3NextDecoderLayerAsymmetric class definition
def _decoder_layer_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None, positions: torch.Tensor = None, **kwargs):
    # Call parent's forward method
    result = OrigQwen3NextDecoderLayer.forward(self, hidden_states, residual, positions, **kwargs)
    return result

Qwen3NextDecoderLayerAsymmetric.forward = _decoder_layer_forward
