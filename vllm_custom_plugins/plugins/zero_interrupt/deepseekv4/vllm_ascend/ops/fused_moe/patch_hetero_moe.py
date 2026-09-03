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
_HASH_SELECT_DIAG_LOGGED = False
_HASH_TOPK_DUMP_LOGGED = False
_HASH_TOPK_DUMP_COUNT = 0
_HASH_TOPK_DUMP_CAP = 512

def _hash_select_diag_once(
    _logger,
    *,
    router_logits,
    input_ids,
    moe_comm_type,
    flash_comm_v1_enabled,
    per_dp_tp_sizes,
):
    """One-shot warning for the hash-select input alignment triage.

    The caller is a __code__-swapped function and imports this helper inside
    its own body; the flag lives here in the plugin module so no extra global
    has to be injected into the patched vllm_ascend module.
    """
    global _HASH_SELECT_DIAG_LOGGED
    if _HASH_SELECT_DIAG_LOGGED:
        return
    _HASH_SELECT_DIAG_LOGGED = True
    _logger.warning_once(
        "[hetero-moe diag] hash select input alignment: "
        "router_logits=%s input_ids=%s moe_comm_type=%s "
        "flash_comm_v1_enabled=%s per_dp_tp_sizes=%s",
        tuple(router_logits.shape),
        tuple(input_ids.shape),
        moe_comm_type,
        flash_comm_v1_enabled,
        str(per_dp_tp_sizes),
    )


def _hash_topk_dump_once(
    _logger,
    *,
    input_ids,
    topk_ids,
    topk_weights,
    moe_comm_type,
):
    """One-shot dump of the first real request's hash-routing topk.

    Gated by VLLM_ITS_DUMP_HASH_TOPK=1 and a module-level flag so each
    worker prints only the first non-profile hash call. This makes the
    DP16 (default MC2) and DP15 (or forced-ALLGATHER) runs directly
    comparable value by value.
    """
    import os as _os

    global _HASH_TOPK_DUMP_LOGGED
    if _HASH_TOPK_DUMP_LOGGED or _os.getenv("VLLM_ITS_DUMP_HASH_TOPK", "0") != "1":
        return
    _HASH_TOPK_DUMP_LOGGED = True
    _logger.warning_once(
        "[hetero-moe diag] hash topk dump: moe_comm_type=%s "
        "input_ids16=%s topk_ids4=%s topk_weights4=%s",
        moe_comm_type,
        str(input_ids.detach().cpu().flatten()[:16].tolist()),
        str(topk_ids.detach().cpu()[:4].float().tolist()),
        str(topk_weights.detach().cpu()[:4].float().tolist()),
    )


def _hash_topk_dump_every(
    _logger,
    *,
    input_ids,
    topk_ids,
    topk_weights,
    moe_comm_type,
    router_rows,
):
    """Per-forward hash-topk dump for cross-request determinism triage.

    Gated by VLLM_ITS_DUMP_HASH_TOPK_EVERY=1 and capped so a long
    speculative decode does not flood the dp logs.  The dump is emitted for
    every non-profile hash-select call, which lets us compare the same
    prompt across repeated requests step by step: if the first-layer
    prefill topk already differs between two identical requests, the
    divergence starts in the input/hidden states before MoE; if the topk
    rows agree but the generated text still differs, the divergence is
    downstream (KV/attention, deeper layers, or speculative acceptance).
    """
    import os as _os

    global _HASH_TOPK_DUMP_COUNT
    if (
        _os.getenv("VLLM_ITS_DUMP_HASH_TOPK_EVERY", "0") != "1"
        or _HASH_TOPK_DUMP_COUNT >= _HASH_TOPK_DUMP_CAP
    ):
        return
    _HASH_TOPK_DUMP_COUNT += 1
    _ids = input_ids.detach().cpu().flatten()[:16].tolist()
    _tk = topk_ids.detach().cpu()[:4].to(torch.int64).tolist()
    _tw = topk_weights.detach().cpu()[:4].tolist()
    _logger.warning(
        "[hetero-moe diag] hash topk every#%s: moe_comm_type=%s rows=%s "
        "input_ids16=%s topk_ids4=%s topk_weights4=%s",
        _HASH_TOPK_DUMP_COUNT,
        moe_comm_type,
        router_rows,
        str(_ids),
        str(_tk),
        str(_tw),
    )


def _patch_code(target, new_func):
    """Replace only the code object of *target* with that of *new_func*."""
    target.__code__ = new_func.__code__


def _bind_method(cls, method_name, new_func):
    """Bind *new_func* as a class attribute on *cls*.

    Used instead of ``__code__`` replacement for ``__init__`` methods whose
    compiled code closes over ``__class__`` (zero-arg ``super()`` in the
    installed v0.23 source).  A code object with a different free-var list
    cannot be assigned to such a function object, so the whole function is
    rebound and its required globals are injected into this module first.
    """
    setattr(cls, method_name, new_func)


def _ensure_global(module, name, value):
    """Add *name* to *module* globals when it is missing."""
    if name not in module.__dict__:
        module.__dict__[name] = value


def _ensure_plugin_global(name, value):
    """Expose *value* to whole-method patches bound from this module."""
    if name not in globals():
        globals()[name] = value


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

    prepare_output = self._prepare_with_dp_group(
        hidden_states, router_logits, enable_shared_expert_dp, replace_allreduce
    )
    _expected_rows = self.moe_config.dp_size * _EXTRA_CTX.max_tokens_across_dp
    assert prepare_output.hidden_states.shape[0] == _expected_rows, (
        f"[hetero-moe diag] prepare_allgather_dp gathered rows "
        f"{prepare_output.hidden_states.shape[0]} != dp_size "
        f"{self.moe_config.dp_size} * max_tokens_across_dp "
        f"{_EXTRA_CTX.max_tokens_across_dp}"
    )
    if not getattr(self, "_prepare_allgather_diag_checked", False):
        self._prepare_allgather_diag_checked = True
        from vllm.logger import init_logger as _init_logger

        _init_logger(__name__).warning_once(
            "[hetero-moe diag] prepare_allgather_dp: local_rows=%s "
            "max_tokens_across_dp=%s gathered_rows=%s",
            hidden_states.shape[0],
            _EXTRA_CTX.max_tokens_across_dp,
            prepare_output.hidden_states.shape[0],
        )
    return prepare_output


def _patched_finalize_allgather(self, hidden_states, reduce_results, padded_hidden_states_shape=None):
    if enable_sp() or enable_sp_by_pass() or getattr(_EXTRA_CTX, "per_dp_tp_sizes", None) is not None:
        return self._finalize_with_ep_group(hidden_states)

    _expected_rows = self.moe_config.dp_size * _EXTRA_CTX.max_tokens_across_dp
    assert hidden_states.shape[0] == _expected_rows, (
        f"[hetero-moe diag] finalize_allgather_dp input rows "
        f"{hidden_states.shape[0]} != dp_size {self.moe_config.dp_size} "
        f"* max_tokens_across_dp {_EXTRA_CTX.max_tokens_across_dp}"
    )
    out = self._finalize_with_dp_group(hidden_states, reduce_results)
    if not self.enable_shared_expert_dp:
        assert out.shape[0] == self.num_tokens, (
            f"[hetero-moe diag] finalize_allgather_dp output rows "
            f"{out.shape[0]} != self.num_tokens {self.num_tokens}"
        )
    if not getattr(self, "_finalize_allgather_diag_checked", False):
        self._finalize_allgather_diag_checked = True
        from vllm.logger import init_logger as _init_logger

        _init_logger(__name__).warning_once(
            "[hetero-moe diag] finalize_allgather_dp: input_rows=%s "
            "expected_rows=%s output_rows=%s self.num_tokens=%s",
            hidden_states.shape[0],
            _expected_rows,
            out.shape[0],
            self.num_tokens,
        )
    return out


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
        assert input_ids.shape[0] <= max_tokens_across_dp, (
            f"[hetero-moe diag] input_ids rows {input_ids.shape[0]} exceed "
            f"max_tokens_across_dp {max_tokens_across_dp}"
        )
        _local_input_rows = input_ids.shape[0]
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
            _log_input_gather = not getattr(
                self, "_input_id_gather_diag_checked", False
            )
            input_ids = self.moe_config.dp_group.all_gather(input_ids, 0)
            assert input_ids.shape[0] == true_dp_size * max_tokens_across_dp, (
                f"[hetero-moe diag] gathered input_ids rows "
                f"{input_ids.shape[0]} != true_dp_size "
                f"{true_dp_size} * max_tokens_across_dp "
                f"{max_tokens_across_dp}"
            )
            if _log_input_gather:
                self._input_id_gather_diag_checked = True
                from vllm.logger import init_logger as _init_logger

                _init_logger(__name__).warning_once(
                    "[hetero-moe diag] input_id_dp_gather: local_rows=%s "
                    "max_tokens_across_dp=%s self.num_tokens=%s "
                    "gathered_rows=%s",
                    _local_input_rows,
                    max_tokens_across_dp,
                    getattr(self, "num_tokens", None),
                    input_ids.shape[0],
                )
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
        if not getattr(self, "_hetero_map_checked", False):
            # One-time NPU -> CPU sync: verify the remainder-aware expert
            # range against the mapped expert ids.  This function is applied
            # via __code__ replacement, so every new dependency must be a
            # local import or a local variable.
            from vllm.logger import init_logger as _init_logger

            _diag_logger = _init_logger(__name__)
            # expert_map is a global->local mapping: the non-(-1) INDICES are
            # global expert ids, while the VALUES are this rank's local ids.
            _mapped_global_ids = torch.where(expert_map != -1)[0].tolist()
            _mapped_local_ids = expert_map[expert_map != -1].tolist()
            _diag_logger.warning_once(
                "[hetero-moe diag] token_dispatch_allgather expert map: "
                "num_experts=%s ep_size=%s num_experts_local=%s first=%s "
                "last=%s len(expert_map)=%s mapped_global_ids=%s "
                "mapped_local_ids=%s",
                self.num_experts,
                self.ep_size,
                self.num_experts_local,
                first_expert_idx,
                last_expert_idx,
                len(expert_map),
                str(_mapped_global_ids),
                str(_mapped_local_ids),
            )
            assert len(_mapped_global_ids) == self.num_experts_local, (
                f"[hetero-moe diag] mapped global expert count "
                f"{len(_mapped_global_ids)} != self.num_experts_local "
                f"{self.num_experts_local}"
            )
            assert _mapped_global_ids == list(
                range(first_expert_idx, last_expert_idx)
            ), (
                f"[hetero-moe diag] mapped global expert ids "
                f"{_mapped_global_ids} != expected range "
                f"{list(range(first_expert_idx, last_expert_idx))}"
            )
            assert _mapped_local_ids == list(range(self.num_experts_local)), (
                f"[hetero-moe diag] mapped local expert ids "
                f"{_mapped_local_ids} != expected range "
                f"{list(range(self.num_experts_local))}"
            )
            self._hetero_map_checked = True
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
    # Cheap shape assert runs every call; the .item() diagnostics below sync
    # only once per dispatcher instance.
    assert expert_tokens.shape[0] == self.num_experts_local, (
        f"[hetero-moe diag] npu_moe_init_routing returned "
        f"expert_tokens.shape[0]={expert_tokens.shape[0]}, expected "
        f"self.num_experts_local={self.num_experts_local}"
    )
    if not getattr(self, "_hetero_dispatch_shapes_checked", False):
        from vllm.logger import init_logger as _init_logger

        _init_logger(__name__).warning_once(
            "[hetero-moe diag] token_dispatch_allgather shapes: "
            "hidden_states=%s topk_ids=%s topk_min=%s topk_max=%s "
            "sorted_hidden_states=%s expanded_row_idx=%s "
            "expert_tokens_sum=%s",
            tuple(hidden_states.shape),
            tuple(topk_ids.shape),
            int(topk_ids.min().item()),
            int(topk_ids.max().item()),
            tuple(sorted_hidden_states.shape),
            tuple(expanded_row_idx.shape),
            int(expert_tokens.sum().item()),
        )
        self._hetero_dispatch_shapes_checked = True
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
            assert input_ids.shape[0] == router_logits.shape[0], (
                f"[hetero-moe diag] hash select: input_ids rows "
                f"{input_ids.shape[0]} != router_logits rows "
                f"{router_logits.shape[0]}"
            )
            from vllm.logger import init_logger as _init_logger
            from vllm_ascend.ascend_forward_context import (
                _EXTRA_CTX as _diag_extra,
            )
            from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.ops.fused_moe import (  # noqa: E501
                patch_hetero_moe as _diag_mod,
            )

            _diag_mod._hash_select_diag_once(
                _init_logger(__name__),
                router_logits=router_logits,
                input_ids=input_ids,
                moe_comm_type=forward_context.moe_comm_type,
                flash_comm_v1_enabled=forward_context.flash_comm_v1_enabled,
                per_dp_tp_sizes=getattr(_diag_extra, "per_dp_tp_sizes", None),
            )
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
        if (
            tid2eid is not None
            and not getattr(forward_context, "in_profile_run", False)
        ):
            import os as _os

            if _os.getenv("VLLM_ITS_DUMP_HASH_TOPK", "0") == "1":
                from vllm.logger import init_logger as _init_logger
                from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.ops.fused_moe import (  # noqa: E501
                    patch_hetero_moe as _diag_mod,
                )

                _diag_mod._hash_topk_dump_once(
                    _init_logger(__name__),
                    input_ids=input_ids,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    moe_comm_type=forward_context.moe_comm_type,
                )
            if (
                _os.getenv("VLLM_ITS_DUMP_HASH_TOPK_EVERY", "0") == "1"
                and int(getattr(forward_context, "layer_idx", -1) or -1)
                == 0
            ):
                from vllm.logger import init_logger as _init_logger
                from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.ops.fused_moe import (  # noqa: E501
                    patch_hetero_moe as _diag_mod,
                )

                _diag_mod._hash_topk_dump_every(
                    _init_logger(__name__),
                    input_ids=input_ids,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    moe_comm_type=forward_context.moe_comm_type,
                    router_rows=router_logits.shape[0],
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
# fused_moe_0_23_0: AscendFusedMoE.__init__
# ---------------------------------------------------------------------------


def _patched_ascend_fused_moe_init(self, *args, **kwargs):
    # Copy of hetero_cp v0.23 AscendFusedMoE.__init__ with the remainder-based
    # local expert distribution (256 experts over 15 ranks -> rank 0 owns 18).
    _ = kwargs.pop("hash") if "hash" in kwargs else None
    tid2eid = kwargs.pop("tid2eid") if "tid2eid" in kwargs else None

    self._original_routed_scaling_factor = kwargs.get(
        "routed_scaling_factor", 1.0
    )
    super(AscendFusedMoE, self).__init__(*args, **kwargs)
    self.use_overlapped = True
    self._routed_input_transform = kwargs.get("routed_input_transform")
    self._shared_experts = kwargs.get("shared_experts")
    self.shared_expert_stream = None
    has_shared_experts = self._shared_experts is not None
    num_experts = kwargs["num_experts"]
    intermediate_size = kwargs["intermediate_size"]
    num_shared_experts = kwargs.get("n_shared_experts", 0)

    AscendFusedMoE.moe_counter += 1
    self.moe_instance_id = AscendFusedMoE.moe_counter

    self._expert_map = None
    self.log2phy = None

    self.tid2eid = tid2eid

    if self.quant_config is None:
        self.quant_method = AscendUnquantizedFusedMoEMethod(
            self.moe_config, tid2eid=self.tid2eid
        )
    else:
        self.quant_method = self.quant_config.get_quant_method(
            self, self.layer_name, tid2eid=self.tid2eid
        )

    assert self.quant_method is not None
    self.base_quant_method = self.quant_method

    self.moe_config.tp_group = get_tp_group()
    self.moe_config.dp_group = get_dp_group()
    if self.moe_config.ep_size > 1:
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
    self.moe_config.supports_eplb = self.quant_method.supports_eplb
    ascend_config = get_ascend_config()
    self.multistream_overlap_shared_expert = (
        ascend_config.multistream_overlap_shared_expert
        and has_shared_experts
    )
    self.shared_multistream_overlap_gate = (
        ascend_config.multistream_overlap_gate and has_shared_experts
    )
    if self.multistream_overlap_shared_expert:
        logger.info_once(
            "[fused_moe/layer] Multistream overlap shared expert is enabled."
        )
    if enable_sp() and has_shared_experts:
        logger.info_once(
            "[fused_moe/layer] Sequence parallelism is enabled, shared "
            "experts are replicated for best performance."
        )

    self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
    if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
        AscendFusedMoE.gate_stream = torch.npu.Stream()
    if self.multistream_overlap_gate:
        logger.info_once("[fused_moe/layer] Multistream overlap gate is enabled.")
    vllm_config = get_current_vllm_config()
    if (
        self.custom_routing_function is None
        and self.e_score_correction_bias is not None
        and not vllm_config.model_config.is_deepseek_mla
    ):
        self.e_score_correction_bias.data = (
            self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )
        )
    self._gate = kwargs.get("gate")

    # init moe
    eplb_config = ascend_config.eplb_config
    self.mix_placement = getattr(ascend_config, "mix_placement", False)
    self.n_shared_experts = num_shared_experts
    num_experts += num_shared_experts if self.mix_placement else 0
    self.moe_config.num_experts = num_experts
    (
        self.global_expert_map,
        self._expert_map,
        self.log2phy,
        self.global_redundant_expert_num,
    ) = init_eplb_config(
        eplb_config,
        self.moe_instance_id,
        self.moe_config,
        self.mix_placement,
        num_shared_experts,
        tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
    )
    self.global_num_experts = num_experts + self.global_redundant_expert_num
    self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
    # Match determine_expert_map's remainder-based distribution so
    # local_num_experts agrees with expert_map when
    # global_num_experts % ep_size != 0 (e.g. 256 experts / 15 ranks).
    self.local_num_experts = self.global_num_experts // self.ep_size
    if (
        self.global_num_experts % self.ep_size != 0
        and self.ep_rank < self.global_num_experts % self.ep_size
    ):
        self.local_num_experts += 1
    self.expert_map_manager._local_num_experts = self.local_num_experts
    self.expert_map_manager._expert_map = self._expert_map
    if self._expert_map is not None:
        # Whole-function __init__ binding keeps this module's globals, but
        # import the logger locally anyway for consistency with the
        # __code__-swapped diagnostics above.
        from vllm.logger import init_logger as _init_logger

        _diag_logger = _init_logger(__name__)
        _mapped_count = int((self._expert_map != -1).sum().item())
        assert _mapped_count == self.local_num_experts, (
            f"[hetero-moe diag] ascend_fused_moe_init mapped expert count "
            f"{_mapped_count} != local_num_experts {self.local_num_experts}"
        )
        # expert_map maps global expert id (index) -> local expert id (value).
        _mapped_global_ids = torch.where(self._expert_map != -1)[0].tolist()
        _mapped_local_ids = self._expert_map[
            self._expert_map != -1
        ].tolist()
        _diag_logger.warning_once(
            "[hetero-moe diag] ascend_fused_moe_init local experts: "
            "ep_size=%s ep_rank=%s local_num_experts=%s "
            "mapped_global_ids=%s mapped_local_ids=%s",
            self.ep_size,
            self.ep_rank,
            self.local_num_experts,
            str(_mapped_global_ids),
            str(_mapped_local_ids),
        )
    if self._expert_map is not None:
        logger.info_once(
            "[fused_moe/layer] Expert parallelism is enabled."
            " ep_rank=%s/%s, local_num_experts=%s, global_num_experts=%s,"
            " expert_map=%s",
            self.ep_rank,
            self.ep_size,
            self.local_num_experts,
            self.global_num_experts,
            get_compressed_expert_map(self._expert_map),
        )
    if self.dynamic_eplb:
        self.multi_stage = False
        self.moe_load = torch.zeros(
            self.local_num_experts, dtype=torch.int64
        ).npu()
        if eplb_config.eplb_policy_type == 3:
            self.multi_stage = True
            self.load_counter = torch.tensor(
                0, dtype=torch.int32, device="npu"
            )
            self.num_iter = eplb_config.expert_heat_collection_interval
            self.moe_load = torch.zeros(
                (self.num_iter, self.local_num_experts),
                dtype=torch.int32,
                device="npu",
            )

    self.moe_config.num_experts = self.global_num_experts
    self.moe_config.num_local_experts = self.local_num_experts
    self.moe_config.global_redundant_expert_num = (
        self.global_redundant_expert_num
    )
    self.swiglu_limit = getattr(
        self.vllm_config.model_config.hf_config, "swiglu_limit", 0
    )

    moe_quant_params = {
        "num_experts": self.local_num_experts,
        "hidden_size": self.hidden_size,
        "intermediate_size_per_partition": (
            self.intermediate_size_per_partition
        ),
        "params_dtype": self.params_dtype,
        "weight_loader": self.weight_loader,
    }
    if self.quant_method.__class__.__name__ in (
        "GPTQMarlinMoEMethod",
        "CompressedTensorsWNA16MoEMethod",
    ):
        moe_quant_params["intermediate_size_full"] = intermediate_size
    self.quant_method.create_weights(layer=self, **moe_quant_params)

    self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
    self.enable_npugraph_ex_static_kernel = (
        ascend_config.ascend_compilation_config.enable_static_kernel
    )

    setup_moe_comm_method(self.moe_config)
    self.quant_type = self._get_quant_type()

    self.runner = AscendMoERunner(
        self.layer_name,
        self.moe_config,
        self.router,
        self._routed_input_transform,
        kwargs.pop("gate", None),
        kwargs.pop("shared_experts", None),
        self.quant_method,
        self.vllm_config.parallel_config.enable_dbo,
    )

    if self.multistream_overlap_shared_expert:
        original_process_weights = (
            self.quant_method.process_weights_after_loading
        )

        @wraps(original_process_weights)
        def wrapped_process_weights(*wargs, **wkwargs):
            result = original_process_weights(*wargs, **wkwargs)
            self._validate_shared_expert_consistency()
            return result

        self.quant_method.process_weights_after_loading = (
            wrapped_process_weights
        )

    VllmEplbAdaptor.register_layer(self)


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

    # v0.23 AscendFusedMoE lives in fused_moe_0_23_0 and is re-exported
    # through fused_moe.py.  Patch the private class directly.
    from vllm_ascend.ops.fused_moe import fused_moe_0_23_0 as fused_moe_023

    # ``AscendFusedMoE.__init__`` is compiled with a ``__class__`` free var
    # (zero-arg super()), so its code object cannot be replaced by a function
    # compiled without that closure.  Bind the whole function instead and
    # expose the module globals the copied body references.
    for _name in (
        "AscendFusedMoE",
        "AscendUnquantizedFusedMoEMethod",
        "get_tp_group",
        "get_dp_group",
        "get_ep_group",
        "get_mc2_group",
        "get_ascend_config",
        "enable_sp",
        "get_current_vllm_config",
        "init_eplb_config",
        "get_compressed_expert_map",
        "setup_moe_comm_method",
        "AscendMoERunner",
        "VllmEplbAdaptor",
        "logger",
        "wraps",
    ):
        _ensure_plugin_global(_name, getattr(fused_moe_023, _name))
    _bind_method(
        fused_moe_023.AscendFusedMoE,
        "__init__",
        _patched_ascend_fused_moe_init,
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
    # Same ``__class__`` free-var issue as AscendFusedMoE.__init__: the
    # installed TokenDispatcherWithAll2AllV.__init__ uses zero-arg super().
    _ensure_plugin_global(
        "TokenDispatcherWithAll2AllV",
        token_dispatcher.TokenDispatcherWithAll2AllV,
    )
    _bind_method(
        token_dispatcher.TokenDispatcherWithAll2AllV,
        "__init__",
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
