# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Heterogeneous-TP patch for the upstream ``MoERunner.forward``.

Under DP4TP(3,4,4,4) the shared-expert branch processes the local SP-padded
stream while the routed branch is unpadded by the heterogeneous EP gather
path (or vice versa).  Their token counts can differ in either direction, so
pad the shorter branch before ``shared_output + fused_output``.
"""

from __future__ import annotations

import torch

_PATCHED = False


def _patched_moe_runner_forward(
    self,
    hidden_states,
    router_logits,
    input_ids=None,
):
    # Copy of v0.23.0 MoERunner.forward with the heterogeneous shape fix.
    hidden_states, shared_experts_input = self.apply_routed_input_transform(
        hidden_states
    )

    routed_hidden_dim = hidden_states.shape[-1]
    hidden_states, og_hidden_dim = self._maybe_pad_hidden_states(
        shared_experts_input,
        hidden_states,
    )
    hidden_dim_was_padded = hidden_states.shape[-1] > routed_hidden_dim

    result = self._forward_entry(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        self._encode_layer_name(),
        self._trtllm_mxfp4_unpadded_dim(),
    )

    shared_output, fused_output = _unpack(result)
    if (
        shared_output is not None or self.routed_output_transform is not None
    ) and hidden_dim_was_padded:
        fused_output = fused_output[..., :routed_hidden_dim]

    shared_output = self._maybe_reduce_shared_expert_output(shared_output)

    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
        shared_output, fused_output
    )

    fused_output = self.apply_routed_output_transform(fused_output)

    if shared_output is not None:
        # Heterogeneous TP: the shared expert processes the local SP-padded
        # stream while the routed output is unpadded by the EP all_gather +
        # ragged-unpad path (shared > routed), or the EP gather keeps a
        # uniform padded slot while the shared stream is padded only to the
        # local tp_size (routed > shared).  Pad the shorter branch so the
        # add is well-defined; residual handling later un-pads/truncates.
        if shared_output.shape[0] != fused_output.shape[0]:
            if fused_output.shape[0] < shared_output.shape[0]:
                fused_output = torch.nn.functional.pad(
                    fused_output,
                    (0, 0, 0, shared_output.shape[0] - fused_output.shape[0]),
                )
            else:
                shared_output = torch.nn.functional.pad(
                    shared_output,
                    (0, 0, 0, fused_output.shape[0] - shared_output.shape[0]),
                )
        result = shared_output + fused_output
    else:
        result = fused_output

    result = self._maybe_reduce_final_output(result, og_hidden_dim)

    return self._maybe_add_zero_expert_output(result)


def apply_hetero_moe_runner_patch():
    """Install the heterogeneous-TP MoERunner.forward patch."""
    global _PATCHED
    if _PATCHED:
        return
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as mod

    # The copied body references the private ``_unpack`` helper from the
    # target module globals.
    mod.MoERunner.forward.__code__ = _patched_moe_runner_forward.__code__
    _PATCHED = True
