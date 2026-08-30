# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-VL vision module asymmetric TP patch for zero-interrupt inference.

This module provides asymmetric tensor parallel support for Qwen3-VL vision
modules when mm_encoder_tp_mode is not "data".

Architecture:
- Qwen3VLForConditionalGeneration
  └── visual: Qwen3_VisionTransformer
        ├── Qwen3_VisionPatchMerger (linear_fc1: ColumnParallelLinear)
        └── Qwen3_VisionBlock
              └── Qwen3_VisionMLP (linear_fc1, linear_fc2)
"""

from functools import partial
from typing import Callable

import torch
import einops
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.distributed import parallel_state
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention as OrigQwen2_5_VisionAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionBlock as OrigQwen3_VisionBlock,
    Qwen3_VisionMLP as OrigQwen3_VisionMLP,
    Qwen3_VisionPatchEmbed,
    Qwen3_VisionPatchMerger as OrigQwen3_VisionPatchMerger,
    Qwen3_VisionTransformer as OrigQwen3_VisionTransformer,
    is_vit_use_data_parallel,
)
from vllm.model_executor.models.vision import get_vit_attn_backend

from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.layers.patch_linear import (
    ColumnParallelLinearAsymmetric,
    QKVParallelLinearAsymmetric,
    RowParallelLinearAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
    get_tp_asymmetric_shardings,
)
import torch.nn.functional as F



class Qwen3_VisionMLPAsymmetric(OrigQwen3_VisionMLP):
    """Vision MLP with asymmetric TP support."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None and not is_vit_use_data_parallel()

        if not asym:
            # For symmetric TP, call parent's __init__
            OrigQwen3_VisionMLP.__init__(
                self,
                in_features=in_features,
                hidden_features=hidden_features,
                act_fn=act_fn,
                bias=bias,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - call parent's __init__ to create the structure,
        # then replace linear layers with asymmetric versions
        # Replace with asymmetric linear layers
        nn.Module.__init__(self)
        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
        self.linear_fc1 = ColumnParallelLinearAsymmetric(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        self.linear_fc2 = RowParallelLinearAsymmetric(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        self.act_fn = act_fn


class Qwen2_5_VisionAttentionAsymmetric(OrigQwen2_5_VisionAttention):
    """Vision Attention with asymmetric TP support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None and not is_vit_use_data_parallel()

        if not asym:
            # For symmetric TP, call parent's __init__
            OrigQwen2_5_VisionAttention.__init__(
                self,
                embed_dim=embed_dim,
                num_heads=num_heads,
                projection_size=projection_size,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - custom initialization
        nn.Module.__init__(self)

        # Set up tp size/rank like parent does
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Get shardings for asymmetric TP calculation
        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
        world_split_size = sum(tp_asymmetric_shardings)
        split_size = tp_asymmetric_shardings[self.tp_rank]

        # Use same calculation as qkv layer output heads
        # num_heads = divide(total_num_heads, world_split_size) * split_size
        from vllm.distributed.utils import divide
        self.num_attention_heads_per_partition = divide(num_heads, world_split_size) * split_size

        self.hidden_size_per_attention_head = projection_size // num_heads

        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)

        # Replace qkv with asymmetric version
        self.qkv = QKVParallelLinearAsymmetric(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        # Replace proj with asymmetric version
        # Note: return_bias=True to match original Qwen2_5_VisionAttention.forward
        # which expects self.proj(context_layer) to return a tuple (output, _)
        self.proj = RowParallelLinearAsymmetric(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            return_bias=True,
            prefix=f"{prefix}.proj",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        # Keep the attention computation components from parent
        from vllm.model_executor.layers.attention import MMEncoderAttention
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:

        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            qk_reshaped = qk_reshaped.contiguous()
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        context_layer = einops.rearrange(
            context_layer, "b s h d -> s b (h d)", b=batch_size
        ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen3_VisionBlockAsymmetric(OrigQwen3_VisionBlock):
    """Vision Block with asymmetric TP support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None and not is_vit_use_data_parallel()

        if not asym:
            # For symmetric TP, call parent's __init__
            OrigQwen3_VisionBlock.__init__(
                self,
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                act_fn=act_fn,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - custom initialization
        nn.Module.__init__(self)

        self.norm1 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)
        self.norm2 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)

        # Create asymmetric attn and mlp
        self.attn = Qwen2_5_VisionAttentionAsymmetric(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_VisionMLPAsymmetric(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchMergerAsymmetric(OrigQwen3_VisionPatchMerger):
    """Vision Patch Merger with asymmetric TP support."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None and not is_vit_use_data_parallel()

        if not asym:
            # For symmetric TP, call parent's __init__
            OrigQwen3_VisionPatchMerger.__init__(
                self,
                d_model=d_model,
                context_dim=context_dim,
                norm_layer=norm_layer,
                spatial_merge_size=spatial_merge_size,
                use_postshuffle_norm=use_postshuffle_norm,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - custom initialization
        nn.Module.__init__(self)
        self.use_postshuffle_norm = use_postshuffle_norm
        merger_hidden_size = context_dim * (spatial_merge_size**2)
        self.hidden_size = merger_hidden_size
        self.d_model = d_model
        self.spatial_merge_size = spatial_merge_size
        self.quant_config = quant_config
        self.use_data_parallel = False

        # Create norm layers - 注意：原始实现使用 context_dim，不是 merger_hidden_size
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # 原始 Qwen3_VisionPatchMerger: self.norm = norm_layer(context_dim)
        # 注意：如果 use_postshuffle_norm=True，context_dim 会被更新为 self.hidden_size
        if use_postshuffle_norm:
            context_dim = merger_hidden_size
        self.norm = norm_layer(context_dim)
        if use_postshuffle_norm:
            self.post_shuffle_norm = norm_layer(merger_hidden_size)

        # Get asymmetric shardings and create layers
        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
        self.linear_fc1 = ColumnParallelLinearAsymmetric(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        self.linear_fc2 = RowParallelLinearAsymmetric(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel = self.linear_fc1(x)
        if isinstance(x_parallel, tuple):
            x_parallel = x_parallel[0]
        x_parallel = self.act_fn(x_parallel)
        out = self.linear_fc2(x_parallel)
        if isinstance(out, tuple):
            out = out[0]
        return out



class Qwen3_VisionTransformerAsymmetric(OrigQwen3_VisionTransformer):
    """Vision Transformer with asymmetric TP support."""

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None and not is_vit_use_data_parallel()

        if not asym:
            # For symmetric TP, call parent's __init__ normally
            OrigQwen3_VisionTransformer.__init__(
                self,
                vision_config=vision_config,
                norm_eps=norm_eps,
                quant_config=quant_config,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - skip parent's __init__ and manually create components
        nn.Module.__init__(self)

        # Set up tp size/rank like parent does
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.vision_config = vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.quant_config = quant_config
        self.out_hidden_size = vision_config.out_hidden_size * (1 + len(self.deepstack_visual_indexes))

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads

        # Create additional components needed by parent's forward
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionPatchEmbed
        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

        merger_hidden_size = vision_config.hidden_size * (self.spatial_merge_size**2)

        # Create asymmetric components directly
        # 注意：d_model 应该是 vision_config.out_hidden_size，不是 merger_hidden_size
        # merger_hidden_size = hidden_size * spatial_merge_size^2 = 1024 * 4 = 4096
        # out_hidden_size = 2560 (vision module output dimension)
        self.merger = Qwen3_VisionPatchMergerAsymmetric(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_VisionPatchMergerAsymmetric(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlockAsymmetric(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    # dtype and device properties are inherited from parent