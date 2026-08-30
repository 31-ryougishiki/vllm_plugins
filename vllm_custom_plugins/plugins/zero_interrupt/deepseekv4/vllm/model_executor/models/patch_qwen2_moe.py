# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen2 MoE asymmetric TP patch for zero-interrupt inference.

This module provides asymmetric tensor parallel support for Qwen2 MoE models.
Replaces MergedColumnParallelLinear and RowParallelLinear with their asymmetric
counterparts in MLP layers.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.activation import SiluAndMul

from vllm.model_executor.models.qwen2_moe import (
    Qwen2MoeMLP as OrigQwen2MoeMLP,
)

from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.model_executor.layers.patch_linear import (
    MergedColumnParallelLinearAsymmetric,
    RowParallelLinearAsymmetric,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
    get_tp_asymmetric_shardings,
)


class Qwen2MoeMLPAsymmetric(OrigQwen2MoeMLP):
    """Qwen2 MoE MLP with asymmetric TP support.

    Replaces MergedColumnParallelLinear with MergedColumnParallelLinearAsymmetric
    and RowParallelLinear with RowParallelLinearAsymmetric to support non-uniform
    tensor parallel shardings (e.g., [1, 1, 2] for TP=4 with 3 active ranks).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
        # Check for asymmetric TP
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = additional_config.get("zero_interrupt_config", None) if additional_config else None
        asym = zero_interrupt_config is not None

        if not asym:
            # For symmetric TP, just call parent's __init__
            OrigQwen2MoeMLP.__init__(
                self,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                quant_config=quant_config,
                reduce_results=reduce_results,
                expert_gate=expert_gate,
                prefix=prefix,
            )
            return

        # Asymmetric TP path
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.reduce_results = reduce_results
        self.expert_gate = expert_gate

        # Get asymmetric shardings
        tp_asymmetric_shardings = get_tp_asymmetric_shardings(zero_interrupt_config)

        # gate_up_proj: MergedColumnParallelLinearAsymmetric
        # Output dimension (intermediate_size * 2) is sharded asymmetrically
        self.gate_up_proj = MergedColumnParallelLinearAsymmetric(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        # down_proj: RowParallelLinearAsymmetric
        # Input dimension (intermediate_size) is sharded asymmetrically
        self.down_proj = RowParallelLinearAsymmetric(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
            tp_asymmetric_shardings=tp_asymmetric_shardings,
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = torch.sigmoid(self.expert_gate(x)[0]) * out

        return out