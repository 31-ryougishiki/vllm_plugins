# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm_plugins zero_interrupt plugin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime monkey-patch for heterogeneous TP (DP4TP4 -> DP4TP(3,4,4,4)).

The functions in this module are copied from the ``hetero_cp`` final
implementations of vllm_ascend's fused-moe communication classes.  They are
applied by replacing only the target function ``__code__`` so that already
bound methods and other references to the original function objects keep
working.
"""

import torch
import torch.nn.functional as F

_HETERO_MOE_PATCH_APPLIED = False


def _patch_code(target, new_func):
    """Replace only the code object of *target* with that of *new_func*."""
    target.__code__ = new_func.__code__


def _ensure_global(module, name, value):
    """Add *name* to *module* globals when it is missing."""
    if name not in module.__dict__:
        module.__dict__[name] = value


# ---------------------------------------------------------------------------
# prepare_finalize: PrepareAndFinalizeWithAll2All / MC2 / AllGather
# ---------------------------------------------------------------------------


def _patched_restore_tp_across_dp_all2all(self):
    """Restore original TP configuration (same as MC2).

    Under heterogeneous TP the comm method may be constructed while the
    draft model has temporarily patched the TP group to size-1, so read
    the true per-DP-rank TP size/rank from the parallel config.
    """
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import get_world_group

    cfg = get_current_vllm_config_or_none()
    if cfg is not None and cfg.parallel_config.is_heterogeneous_tp:
        pc = cfg.parallel_config
        self.tp_size = pc.tensor_parallel_size
        self.tp_rank = (
            get_world_group().rank
            - pc.get_rank_offset_for_dp(pc.data_parallel_rank)
        )
    else:
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()


def _patched_restore_tp_across_dp_mc2(self):
    """Restore original TP configuration.

    vLLM flattens TP and DP into a single dimension; this method recovers
    the true TP world size and rank for correct tensor slicing.

    Under heterogeneous TP the comm method may be constructed while the
    draft model has temporarily patched the TP group to size-1, so
    ``get_tensor_model_parallel_world_size()`` can return 1 instead of the
    real per-DP-rank TP size. Read the true values from the config instead.
    """
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import get_world_group

    cfg = get_current_vllm_config_or_none()
    if cfg is not None and cfg.parallel_config.is_heterogeneous_tp:
        pc = cfg.parallel_config
        self.tp_size = pc.tensor_parallel_size
        self.tp_rank = (
            get_world_group().rank - pc.get_rank_offset_for_dp(pc.data_parallel_rank)
        )
    else:
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()


def _patched_prepare_allgather(
    self,
    hidden_states,
    router_logits,
    enable_shared_expert_dp,
    replace_allreduce,
    quant_type,
):
    # Heterogeneous TP always uses the EP group path: the DP groups are
    # position-based ({0,3,7,11}, {1,4,8,12}, ...) so the DP-group gather
    # below would mix tokens across DP replicas and drop expert
    # contributions from orphaned ranks.  The per-forward FlashComm1 flag
    # cannot be used here either — it is False for decode batches
    # (num_tokens <= 1000), which would silently fall into the broken
    # DP path.  The hetero-aware gates live in the custom ops.
    if enable_sp() or enable_sp_by_pass() or getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None:
        return self._prepare_with_ep_group(hidden_states, router_logits, quant_type)

    return self._prepare_with_dp_group(hidden_states, router_logits, enable_shared_expert_dp, replace_allreduce)


def _patched_finalize_allgather(self, hidden_states, reduce_results, padded_hidden_states_shape=None):
    if enable_sp() or enable_sp_by_pass() or getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None:
        return self._finalize_with_ep_group(hidden_states)

    return self._finalize_with_dp_group(hidden_states, reduce_results)


def _patched_all_gather_input_id_with_dp_group(self, input_ids):
    # Detect heterogeneous TP from _EXTRA_CTX (set by
    # ascend_forward_context), NOT from
    # get_current_vllm_config_or_none() — the module-level global may
    # be None in the MoE execution path.
    tp_sizes = getattr(_EXTRA_CTX, 'per_dp_tp_sizes', None)
    is_hetero = tp_sizes is not None
    true_dp_size = len(tp_sizes) if is_hetero else self.moe_config.dp_size
    if true_dp_size > 1:
        if is_hetero and enable_sp_by_pass():
            # sp_by_pass leaves the custom gather op un-unpadded: every
            # rank pads its local stream to the same uniform per-rank slot
            # and the EP all_gather output stays at
            # ep_size * uniform_rank.  Match that layout here so input_ids
            # stays aligned with hidden_states/router_logits.
            per_dp = getattr(_EXTRA_CTX, "per_dp_padded_lengths", None)
            if per_dp is not None:
                padded_num = int(
                    getattr(_EXTRA_CTX, "padded_num_tokens", 0) or 0
                )
                if padded_num <= 0:
                    padded_num = max(per_dp)
                stream_padded = bool(
                    getattr(_EXTRA_CTX, "flash_comm_v1_enabled", False)
                    or getattr(_EXTRA_CTX, "flashcomm_v2_enabled", False)
                )
                uniform_rank = max(
                    (padded_num if stream_padded else per_dp[i])
                    // tp_sizes[i]
                    for i in range(len(tp_sizes))
                )
                if input_ids.shape[0] < uniform_rank:
                    input_ids = nn.functional.pad(
                        input_ids, (0, uniform_rank - input_ids.shape[0])
                    )
            return get_ep_group().all_gather(input_ids, 0)

        max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
        # NOTE: pad against the LOCAL input_ids length, not self.num_tokens.
        # In the EP-group prepare path self.num_tokens is set to the total
        # (gathered) token count, so the old reference never padded and the
        # EP all_gather below would receive unequal-sized contributions
        # whenever the DP token counts differ.
        pad_size = max_tokens_across_dp - input_ids.shape[0]
        if pad_size > 0:
            input_ids = nn.functional.pad(input_ids, (0, pad_size))

        if is_hetero:
            # All TP ranks within a DP rank hold identical input_ids.
            # Every rank contributes exactly max_tokens_across_dp rows
            # after the pad above, so the per-DP blocks sit at
            # tp_i * max_tokens_across_dp strides.  Slice each DP's block
            # by its OWN stream width (num_tokens_across_dp_cpu[i]) to
            # match the hidden_states/topk streams produced by the
            # unpad walk in _maybe_all_gather_and_maybe_unpad_impl.
            num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
            all_gathered = get_ep_group().all_gather(input_ids, 0)
            parts = []
            offset = 0
            for i, tp_i in enumerate(tp_sizes):
                width = int(num_tokens_across_dp_cpu[i].item())
                parts.append(all_gathered[offset : offset + width])
                offset += tp_i * max_tokens_across_dp
            input_ids = torch.cat(parts, dim=0)
        else:
            input_ids = self.moe_config.dp_group.all_gather(input_ids, 0)
    return input_ids


# ---------------------------------------------------------------------------
# token_dispatcher: TokenDispatcherWithAllGather / TokenDispatcherWithAll2AllV
# ---------------------------------------------------------------------------


def _patched_token_dispatch_allgather(self, token_dispatch_input):
    quant_type = token_dispatch_input.quant.quant_type
    dynamic_scale = token_dispatch_input.routing.pertoken_scale
    unquantized_mxfp4_dispatch = quant_type == QuantType.MXFP4 and dynamic_scale is None
    # Without prepare-stage scales, MXFP4 stays unquantized in dispatch and
    # is quantized again inside the MLP path.
    with_quant = token_dispatch_input.quant.dispatch_with_quant and quant_type != QuantType.W8A8FP8
    with_quant = with_quant and not unquantized_mxfp4_dispatch
    is_mxfp = token_dispatch_input.quant.is_mxfp
    hidden_states = token_dispatch_input.hidden_states
    topk_weights = token_dispatch_input.topk_weights
    topk_ids = token_dispatch_input.topk_ids
    expert_map = token_dispatch_input.routing.expert_map
    act_quant_type = (
        token_dispatch_input.quant.mxfp.act_quant_type
        if token_dispatch_input.quant.mxfp is not None and not unquantized_mxfp4_dispatch
        else None
    )
    global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
    restore_shape = hidden_states.shape
    # Fuse the first dynamic quant of moe_mlp into initrouting when
    # dispatch_with_quant is on but got a None dynamic_scale.
    if with_quant and dynamic_scale is None:
        if quant_type == QuantType.MXFP4:
            quant_mode = 9
        else:
            quant_mode = 3 if is_mxfp else 1
    else:
        quant_mode = -1

    num_tokens = hidden_states.shape[:-1].numel()
    apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
    if apply_router_weight_on_input:
        assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
        _, topk = topk_weights.shape
        assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
    if expert_map is not None:
        global_num_experts = len(expert_map) + global_redundant_expert_num
        mask = expert_map[topk_ids] != -1
        topk_weights = topk_weights * mask
        # First global expert index of this rank under linear placement.
        # Do NOT use `rank * num_experts_local` -- that assumes uniform
        # per-rank expert counts, which fails when num_experts is not
        # divisible by ep_size (e.g. 256 experts over 15 ranks).
        _ep_rank = get_ep_group().rank_in_group
        _base = self.num_experts // self.ep_size
        _rem = self.num_experts % self.ep_size
        first_expert_idx = _ep_rank * _base + min(_ep_rank, _rem)
        last_expert_idx = first_expert_idx + self.num_experts_local
    else:
        first_expert_idx = 0
        last_expert_idx = self.num_experts_local
        global_num_experts = self.num_experts_local
    sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
        hidden_states,
        topk_ids,
        scale=dynamic_scale,
        active_num=num_tokens * self.top_k,
        expert_num=global_num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[first_expert_idx, last_expert_idx],
        quant_mode=quant_mode,
        act_quant_type=act_quant_type,
    )
    expert_tokens = expert_tokens.to(torch.int64)
    group_list_type = 1  # `count` mode

    return MoETokenDispatchOutput(
        hidden_states=sorted_hidden_states,
        dynamic_scale=dynamic_scale if with_quant else None,
        group_list=expert_tokens,
        group_list_type=group_list_type,
        combine_metadata=MoEAllGatherCombineMetadata(
            topk_weights=topk_weights,
            expanded_row_idx=expanded_row_idx,
            restore_shape=restore_shape,
        ),
    )


def _patched_init_all2allv(self, **kwargs):
    # Equivalent to the class-body ``super().__init__(**kwargs)`` from the
    # hetero_cp source.  ``TokenDispatcherWithAll2AllV`` is resolved from the
    # patched target module globals at call time.
    super(TokenDispatcherWithAll2AllV, self).__init__(**kwargs)
    self.num_local_experts = kwargs.get("num_local_experts", 0)

    assert self.num_local_experts > 0, "Expected at least one expert"
    # Expert ranges per EP rank under linear placement. May be uneven
    # (e.g. 256 experts over 15 ranks -> rank 0 holds 18, others 17),
    # so do NOT assume num_experts is evenly divisible by ep_size.
    ep_size = self.ep_size
    num_experts = self.num_experts
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    self._per_rank_expert_counts = [
        base + (1 if r < remainder else 0) for r in range(ep_size)
    ]
    self._per_rank_expert_starts = []
    _s = 0
    for _n in self._per_rank_expert_counts:
        self._per_rank_expert_starts.append(_s)
        _s += _n

    self.num_local_experts = self._per_rank_expert_counts[self.ep_rank]
    start_idx = self._per_rank_expert_starts[self.ep_rank]
    self.local_expert_indices = [
        start_idx + i for i in range(self.num_local_experts)
    ]
    assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
    for i in range(len(self.local_expert_indices) - 1):
        assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
            "local_expert_indices must be continuous"
        )

    if self.num_local_experts > 1:
        # Global expert -> local expert id within its owning rank.
        expert_ids = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=torch.npu.current_device()
        )
        for r in range(ep_size):
            cnt = self._per_rank_expert_counts[r]
            s = self._per_rank_expert_starts[r]
            expert_ids[s : s + cnt] = torch.arange(
                cnt, dtype=torch.int32, device=expert_ids.device
            )
        self.expert_ids_per_ep_rank = expert_ids

    # TODO: Try local_rank = ep_group.rank_in_group
    local_rank = torch.distributed.get_rank(group=self.ep_group)
    backend = self.ep_group._get_backend(torch.device("npu"))
    self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)


def _patched_token_combine_all2allv(self, hidden_states, combine_metadata, bias=None):
    assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

    # 1. Preprocess using metadata
    hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

    # 2. AllToAll
    _, permutated_local_input_tokens, handle = async_all_to_all(
        hidden_states,
        combine_metadata.input_splits,
        combine_metadata.output_splits,
        self.ep_group,
    )
    handle.wait()
    hidden_states.untyped_storage().resize_(0)

    # 3. Postprocess using metadata
    output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

    return output


def _patched_preprocess_all2allv(self, topk_ids):
    import numpy as np

    num_local_tokens_per_expert = torch.histc(
        topk_ids, bins=self.num_experts, min=0, max=self.num_experts
    )

    ep_size = self.ep_size
    num_out_tokens = topk_ids.numel()

    # Sum tokens per EP rank's actual (possibly uneven) expert range.
    input_splits = np.zeros(ep_size, dtype=np.int64)
    for r in range(ep_size):
        s = self._per_rank_expert_starts[r]
        cnt = self._per_rank_expert_counts[r]
        input_splits[r] = num_local_tokens_per_expert[s : s + cnt].sum().item()

    num_global_tokens_per_expert = gather_from_sequence_parallel_region(
        num_local_tokens_per_expert, group=self.ep_group
    ).reshape(ep_size, self.num_experts)

    # rank_local_tokens[r][j] = tokens that rank r routes to this rank's
    # j-th local expert. The all_to_all output arrives grouped by sender
    # rank, so per-row (sender) local-expert counts are the receive splits.
    rank_local_tokens = num_global_tokens_per_expert[
        :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
    ]
    output_splits = np.zeros(ep_size, dtype=np.int64)
    for r in range(ep_size):
        output_splits[r] = rank_local_tokens[r].sum().item()
    num_tokens_per_local_expert = rank_local_tokens.sum(axis=0)

    global_input_tokens_local_experts_indices = None
    if self.num_local_experts > 1:
        local_ids = torch.arange(
            self.num_local_experts,
            dtype=torch.int32,
            device=num_global_tokens_per_expert.device,
        )
        indices_parts = [
            torch.repeat_interleave(local_ids, rank_local_tokens[r])
            for r in range(ep_size)
        ]
        global_input_tokens_local_experts_indices = torch.cat(indices_parts)
    else:
        torch.npu.synchronize()

    return (
        num_tokens_per_local_expert,
        input_splits,
        output_splits,
        global_input_tokens_local_experts_indices,
        num_out_tokens,
    )


# ---------------------------------------------------------------------------
# experts_selector: _select_experts_with_fusion_ops
# ---------------------------------------------------------------------------


def _patched_select_experts_with_fusion_ops(
    hidden_states,
    router_logits,
    top_k,
    use_grouped_topk,
    renormalize,
    e_score_correction_bias,
    topk_group,
    num_expert_group,
    scoring_func="softmax",
    routed_scaling_factor=1.0,
    tid2eid=None,
    input_ids=None,
):
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
            else:
                input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(input_ids)

            if forward_context.flash_comm_v1_enabled and forward_context.moe_comm_type != MoECommType.ALLGATHER:
                # Process for Flash Comm V1
                tp_size = get_tp_group().world_size
                tp_rank = get_tp_group().rank_in_group
                # The per-rank router_logits are aligned to padded_num_tokens
                # (LCM of all DP-rank TP sizes), which can exceed the raw
                # input_ids length under heterogeneous TP (e.g. 1425 -> 1428,
                # giving 476 rows/rank at tp=3). Pad input_ids to that same
                # target so the split stays aligned, then split evenly.
                target = getattr(_EXTRA_CTX, "padded_num_tokens", None)
                if target is not None and input_ids.shape[0] < target:
                    input_ids = F.pad(input_ids, (0, target - input_ids.shape[0]))
                elif input_ids.shape[0] % tp_size != 0:
                    pad_len = tp_size - (input_ids.shape[0] % tp_size)
                    input_ids = F.pad(input_ids, (0, pad_len))
                chunk = input_ids.shape[0] // tp_size
                input_ids = input_ids[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=e_score_correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid_ones,
            k_group=topk_group,
            group_count=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            # The hash custom op currently rejects renorm != 0. Apply
            # norm_topk_prob in Python below before returning to MoE compute.
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        return topk_weights, topk_ids
    norm_type = 0 if scoring_func == "softmax" else 1
    if e_score_correction_bias is not None and e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=e_score_correction_bias,
    )

    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def apply_hetero_moe_patch():
    """Patch installed vllm_ascend fused-moe classes for heterogeneous TP.

    Only the ``__code__`` of each target function is replaced.  The target
    function object keeps its own ``__globals__``, ``__defaults__`` and
    ``__kwdefaults__``; missing global helpers used by the copied code are
    injected into the target module globals.
    """
    global _HETERO_MOE_PATCH_APPLIED

    if _HETERO_MOE_PATCH_APPLIED:
        return

    from vllm.distributed.parallel_state import get_ep_group
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe import (
        experts_selector,
        prepare_finalize,
        token_dispatcher,
    )

    # Global helper names referenced by the copied code but not imported by
    # the installed (origin) target modules.
    _ensure_global(prepare_finalize, "get_ep_group", get_ep_group)
    _ensure_global(experts_selector, "_EXTRA_CTX", _EXTRA_CTX)

    _patch_code(
        prepare_finalize.PrepareAndFinalizeWithAll2All._restore_tp_across_dp,
        _patched_restore_tp_across_dp_all2all,
    )
    _patch_code(
        prepare_finalize.PrepareAndFinalizeWithMC2._restore_tp_across_dp,
        _patched_restore_tp_across_dp_mc2,
    )
    _patch_code(
        prepare_finalize.PrepareAndFinalizeWithAllGather.prepare,
        _patched_prepare_allgather,
    )
    _patch_code(
        prepare_finalize.PrepareAndFinalizeWithAllGather.finalize,
        _patched_finalize_allgather,
    )
    _patch_code(
        prepare_finalize.PrepareAndFinalizeWithAllGather.all_gather_input_id_with_dp_group,
        _patched_all_gather_input_id_with_dp_group,
    )

    _patch_code(
        token_dispatcher.TokenDispatcherWithAllGather.token_dispatch,
        _patched_token_dispatch_allgather,
    )
    _patch_code(
        token_dispatcher.TokenDispatcherWithAll2AllV.__init__,
        _patched_init_all2allv,
    )
    _patch_code(
        token_dispatcher.TokenDispatcherWithAll2AllV.token_combine,
        _patched_token_combine_all2allv,
    )
    _patch_code(
        token_dispatcher.TokenDispatcherWithAll2AllV._preprocess,
        _patched_preprocess_all2allv,
    )

    _patch_code(
        experts_selector._select_experts_with_fusion_ops,
        _patched_select_experts_with_fusion_ops,
    )

    _HETERO_MOE_PATCH_APPLIED = True
