# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3.5 asymmetric TP patch for zero-interrupt inference.

This module provides asymmetric tensor parallel support for Qwen3.5 dense models.
MoE support will be added later.

Architecture:
- Qwen3_5ForCausalLMBase → Qwen3_5Model → Qwen3_5DecoderLayer
-                                             ├── linear_attention → Qwen3_5GatedDeltaNetAsymmetric
-                                             └── full_attention → Qwen3NextAttentionAsymmetric
"""

from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile
import torch
from torch import nn

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer as OrigQwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration as OrigQwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet as OrigQwen3_5GatedDeltaNet,
    Qwen3_5Model as OrigQwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    ChunkGatedDeltaRule,
    Qwen3NextAttention as OrigQwen3NextAttention,
    Qwen3NextGatedDeltaNet as OrigQwen3NextGatedDeltaNet,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_layers,
    maybe_prefix,
    make_empty_intermediate_tensors_factory
)
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

from vllm_ascend.patch.worker.patch_qwen3_5 import AscendQwen3_5GatedDeltaNet

from vllm_custom_plugins.plugins.zero_interrupt.vllm.model_executor.layers.patch_linear import (
    MergedColumnParallelLinearAsymmetric,
    RowParallelLinearAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.vllm.model_executor.layers.patch_vocab_parallel_embedding import (
    ParallelLMHeadAsymmetric,
    VocabParallelEmbeddingAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.vllm.v1.executor.utils import (
    get_tp_asymmetric_shardings,
)
from .patch_qwen3_next import (
    Qwen3NextAttentionAsymmetric,
    Qwen3NextGatedDeltaNetAsymmetric,
)
from .patch_qwen3_vl import Qwen3_VisionTransformerAsymmetric
from .patch_qwen2_moe import Qwen2MoeMLPAsymmetric as Qwen3NextMLPAsymmetric



class Qwen3_5GatedDeltaNetAsymmetric(OrigQwen3_5GatedDeltaNet):
    """Qwen3.5 GatedDeltaNet with asymmetric TP.

    Inherits from Qwen3_5GatedDeltaNet which inherits from Qwen3NextGatedDeltaNet.
    Uses asymmetric linear layers for in_proj_qkvz, in_proj_ba, and out_proj.
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
        # Check for asymmetric TP first
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        if asym:
            # For asymmetric TP, manually init and replace layers
            nn.Module.__init__(self)
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()

            # [mzm] Debug logging for GatedDeltaNet initialization
            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)

            # Qwen3_5Config wraps actual config in text_config
            text_config = config.text_config
            self.hidden_size = text_config.hidden_size
            self.num_v_heads = text_config.linear_num_value_heads
            self.num_k_heads = text_config.linear_num_key_heads
            self.head_k_dim = text_config.linear_key_head_dim
            self.head_v_dim = text_config.linear_value_head_dim
            self.key_dim = self.head_k_dim * self.num_k_heads
            self.value_dim = self.head_v_dim * self.num_v_heads

            self.conv_kernel_size = text_config.linear_conv_kernel_dim
            self.layer_idx = None
            self.activation = text_config.hidden_act
            from transformers.activations import ACT2FN
            self.act = ACT2FN[text_config.hidden_act]
            self.layer_norm_epsilon = text_config.rms_norm_eps
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

            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
            world_split_size = sum(tp_asymmetric_shardings)
            split_size = tp_asymmetric_shardings[self.tp_rank]

            self.asym = True
            self.tp_asymmetric_shardings = tp_asymmetric_shardings
            self.world_split_size = world_split_size

            self.local_num_v_heads = self.num_v_heads * split_size // world_split_size
            self.local_num_k_heads = max(
                1, self.num_k_heads * split_size // world_split_size
            )

            # Conv1d (same as parent, no TP sharding on conv dim)
            from vllm_custom_plugins.plugins.zero_interrupt.vllm.model_executor.layers.patch_linear import ColumnParallelLinearAsymmetric
            self.conv_dim = self.key_dim * 2 + self.value_dim
            self.conv1d = ColumnParallelLinearAsymmetric(
                input_size=self.conv_kernel_size,
                output_size=self.conv_dim,
                bias=False,
                prefix=f"{prefix}.conv1d",
                tp_asymmetric_shardings=tp_asymmetric_shardings
            )
            self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

            # Asymmetric input projections
            self.in_proj_qkvz = MergedColumnParallelLinearAsymmetric(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )

            self.in_proj_ba = MergedColumnParallelLinearAsymmetric(
                input_size=self.hidden_size,
                output_sizes=[self.num_v_heads, self.num_v_heads],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )

            query_key_settings = (self.key_dim, 0, False)
            value_settings = (self.value_dim, 0, False)

            delattr(self.conv1d.weight, "weight_loader")
            from vllm.model_executor.utils import set_weight_attrs
            from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader

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



            # Time step projection parameters (use local_num_v_heads for asymmetric TP)
            self.dt_bias = nn.Parameter(
                torch.ones(self.local_num_v_heads),
            )
            self.A_log = nn.Parameter(
                torch.empty(self.local_num_v_heads),
            )
            from vllm.model_executor.model_loader.weight_utils import default_weight_loader

            # Create asymmetric weight loader that handles non-uniform TP sharding
            def asymmetric_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
                """Weight loader for asymmetric TP - loads the correct slice for this rank."""
                shard_size = param.shape[0]
                # Calculate start index: sum of previous ranks' shard sizes
                shard_unit = self.num_v_heads // world_split_size  # size of each unit shard
                start_idx = sum(tp_asymmetric_shardings[:self.tp_rank]) * shard_unit
                # The loaded_weight contains full weight, narrow to our slice
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                param.data.copy_(loaded_weight)

            set_weight_attrs(self.A_log, {"weight_loader": asymmetric_weight_loader})
            set_weight_attrs(self.dt_bias, {"weight_loader": asymmetric_weight_loader})

            # RMSNormGated for output
            from vllm.model_executor.layers.layernorm import RMSNormGated
            self.norm = RMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                group_size=None,
                norm_before_gate=True,
            )

            # Asymmetric output projection
            self.out_proj = RowParallelLinearAsymmetric(
                self.value_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
                tp_asymmetric_shardings=tp_asymmetric_shardings,
            )

            self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
            self.enable_packed_recurrent_decode = False

            # Register with compilation static forward context
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
        else:
            # For symmetric TP, just call parent's full __init__
            OrigQwen3_5GatedDeltaNet.__init__(
                self,
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=prefix,
            )
            # Store symmetric config for get_state_shape (needed since parent class has this method)
            self.asym = False
            self.tp_asymmetric_shardings = None
            self.world_split_size = self.tp_size
            self.local_num_v_heads = self.num_v_heads // self.tp_size

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


# [mzm] Wrapper for Qwen3_5 GatedDeltaNet forward to add logging
def _qwen3_5_gated_delta_forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
    original_tp_size = self.tp_size
    try:
        if hasattr(self, 'asym') and self.asym and hasattr(self, 'tp_asymmetric_shardings'):
            world_split_size = sum(self.tp_asymmetric_shardings)
            split_size = self.tp_asymmetric_shardings[self.tp_rank]
            self.tp_size = world_split_size // split_size
        result = AscendQwen3_5GatedDeltaNet.forward(self, hidden_states, output)
    finally:
        self.tp_size = original_tp_size
    return result

Qwen3_5GatedDeltaNetAsymmetric.forward = _qwen3_5_gated_delta_forward
Qwen3_5GatedDeltaNetAsymmetric._forward_core = AscendQwen3_5GatedDeltaNet._forward_core


class Qwen3_5DecoderLayerAsymmetric(OrigQwen3_5DecoderLayer):
    """Qwen3.5 Decoder layer with asymmetric TP.

    Uses Qwen3_5GatedDeltaNetAsymmetric for linear_attention
    and Qwen3NextAttentionAsymmetric for full_attention.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        # Check for asymmetric TP first
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        if not asym:
            # For symmetric TP, just call parent's full __init__
            OrigQwen3_5DecoderLayer.__init__(
                self,
                vllm_config=vllm_config,
                layer_type=layer_type,
                prefix=prefix,
            )
            return

        # Asymmetric TP path - do custom initialization
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)


        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNetAsymmetric(
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

        # MLP - same as parent (Qwen3NextMLP or Qwen3NextSparseMoeBlock)
        # Qwen3_5Config wraps actual config in text_config
        text_config = config.text_config
        if text_config.model_type == "qwen3_5_moe_text":
            from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif text_config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLPAsymmetric(
                hidden_size=text_config.hidden_size,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                vllm_config=vllm_config
            )

        # Layer norms
        self.input_layernorm = Qwen3_5RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, text_config.hidden_size),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, text_config.hidden_size),
            )

@support_torch_compile
class Qwen3_5ModelAsymmetric(OrigQwen3_5Model):
    """Qwen3.5 base model with asymmetric TP support.

    Uses Qwen3_5DecoderLayerAsymmetric for decoder layers.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Check for asymmetric TP first
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        if not asym:
            # For symmetric TP, just call parent's full __init__
            OrigQwen3_5Model.__init__(self, vllm_config=vllm_config, prefix=prefix)
            return

        # Asymmetric TP path - do custom initialization
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.vocab_size = config.vocab_size
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
        self.embed_tokens = VocabParallelEmbeddingAsymmetric(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayerAsymmetric(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize aux_hidden_state_layers
        self.aux_hidden_state_layers: tuple[int, ...] = ()


        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


class Qwen3_5ForCausalLMAsymmetric(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, SupportsEagle3
):
    """Qwen3.5 ForCausalLM with asymmetric TP support.

    This class handles the lm_head sharding for asymmetric TP:
    - If asym=True: uses ParallelLMHeadAsymmetric
    - If tie_word_embeddings=True: uses model.embed_tokens
    - Otherwise: uses standard ParallelLMHead
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # Check for asymmetric TP
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None)
        asym = zero_interrupt_config is not None

        # Create the base model with asymmetric support
        self.model = Qwen3_5ModelAsymmetric(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # Handle lm_head based on asymmetric config
        if asym:
            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
            if config.tie_word_embeddings:
                # When tie_word_embeddings=True, lm_head and embed_tokens share
                # the same parameter. This matches the original behavior.
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHeadAsymmetric(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                    tp_asymmetric_shardings=tp_asymmetric_shardings,
                )
        else:
            if get_pp_group().is_last_rank:
                if config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        config.vocab_size,
                        config.hidden_size,
                        quant_config=quant_config,
                        prefix=maybe_prefix(prefix, "lm_head"),
                    )
            else:
                self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # Log top-k tokens for debugging
        topk_values, topk_indices = torch.topk(logits[0], k=5)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForConditionalGenerationAsymmetric(
    OrigQwen3_5ForConditionalGeneration, IsHybrid
):
    """Qwen3.5 ForConditionalGeneration with asymmetric TP support.

    This patches the vision module to use asymmetric TP when needed.
    """

    supports_multimodal_pruning = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            # Use asymmetric VisionTransformer
            self.visual = Qwen3_VisionTransformerAsymmetric(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLMAsymmetric(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        head_k_dim = hf_config.linear_key_head_dim
        head_v_dim = hf_config.linear_value_head_dim
        num_k_heads = hf_config.linear_num_key_heads
        num_v_heads = hf_config.linear_num_value_heads
        conv_kernel_size = hf_config.linear_conv_kernel_dim

        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

        # Check for asymmetric TP from vllm_config directly
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None) if additional_config else None

        if not zero_interrupt_config:
            # Symmetric TP: use divide() for exact divisibility check
            from vllm.distributed.utils import divide
            conv_state_dim = divide(conv_dim, tp_size)
            v_heads_dim = divide(num_v_heads, tp_size)
        else:
            # Asymmetric TP: use shardings from vllm_config
            tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)
            if not tp_asymmetric_shardings or len(tp_asymmetric_shardings) != tp_size:
                from vllm.distributed.utils import divide
                conv_state_dim = divide(conv_dim, tp_size)
                v_heads_dim = divide(num_v_heads, tp_size)
            else:
                # Get tp_rank at runtime
                try:
                    tp_rank = get_tensor_model_parallel_rank()
                except AssertionError:
                    # TP group not initialized, fallback to symmetric
                    from vllm.distributed.utils import divide
                    conv_state_dim = divide(conv_dim, tp_size)
                    v_heads_dim = divide(num_v_heads, tp_size)
                else:
                    world_split_size = sum(tp_asymmetric_shardings)
                    rank_shard_size = tp_asymmetric_shardings[tp_rank]
                    conv_state_dim = conv_dim * rank_shard_size // world_split_size
                    v_heads_dim = num_v_heads * rank_shard_size // world_split_size

        conv_state_shape = (
            conv_state_dim,
            conv_kernel_size - 1 + num_spec,
        )
        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        temporal_state_shape = (
            v_heads_dim,
            head_v_dim,
            head_k_dim,
        )
        return conv_state_shape, temporal_state_shape