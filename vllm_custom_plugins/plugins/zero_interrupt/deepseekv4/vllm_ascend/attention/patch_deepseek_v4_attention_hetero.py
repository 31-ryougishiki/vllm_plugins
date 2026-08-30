# -*- coding: utf-8 -*-
"""Heterogeneous-TP monkey-patches for vLLM Ascend DeepSeek-V4 attention.

This module belongs to the vllm_plugins zero_interrupt plugin.  It patches the
installed ``vllm_ascend.attention.dsa_v1`` and
``vllm_ascend.attention.context_parallel.dsa_cp`` modules so that DeepSeek-V4
attention metadata and the o_proj TP-head restore path work correctly under
heterogeneous TP (e.g. DP4TP(3,4,4,4) with DP0 tp=3 using
tp_sharding_ratios [2,1,1]).

Call ``apply_deepseek_v4_attention_hetero_patch()`` once after vLLM Ascend has
been imported.  The patch is idempotent.  Standalone copied method bodies are
installed via ``target_method.__code__ = new_function.__code__``; wrappers
that must call the original are installed as replacement function objects so
the saved original cannot be mutated into recursion.
"""

from __future__ import annotations

import math

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank

try:
    from vllm.distributed.utils import (
        get_tp_partition_offset,
        get_tp_partition_size,
    )
except ImportError:  # stock v0.23 tree without the hetero utils patch
    from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.distributed.patch_hetero_utils import (
        get_tp_partition_offset,
        get_tp_partition_size,
    )

_ATTENTION_HETERO_PATCH_APPLIED = False

# Originals saved in THIS module.  The wrapper functions below resolve these
# names through their own ``__globals__``.  Saving them on the target module
# instead and then replacing the target method's ``__code__`` mutates the very
# function object that was just saved, which makes the wrapper call itself.
_ORIG_BUILD_LOCAL_TOKEN_METADATA = None
_ORIG_DSA_CP_INIT = None
_ORIG_DSA_CP_PROCESS_WEIGHTS = None


def _get_dsa_local_heads(vllm_config: VllmConfig | None, total_num_heads: int, tp_size: int) -> int:
    """Return the number of local attention heads for DSA metadata.

    Under heterogeneous TP with asymmetric sharding ratios, the uniform
    ``total_num_heads // tp_size`` yields the wrong value (e.g. 21 instead
    of 32/16/16 for tp=3 with ratios [2,1,1]).  Use get_tp_partition_size
    when ratios are set.

    NOTE: the per-rank vllm config global is NOT set inside the model
    forward at runtime, so the ratios must be read from the builder's own
    vllm_config (stored in __init__) -- get_current_tp_sharding_ratios()
    returns None there and silently falls back to the wrong uniform value.
    """
    ratios = None
    if vllm_config is not None and vllm_config.parallel_config.is_heterogeneous_tp:
        ratios = vllm_config.parallel_config.get_sharding_ratios_for_dp(
            vllm_config.parallel_config.data_parallel_rank
        )
    if ratios is not None:
        tp_rank = get_tensor_model_parallel_rank()
        return get_tp_partition_size(total_num_heads, tp_rank, tp_size, ratios)
    return total_num_heads // tp_size



# =====================================================================
# Copied from hetero_cp/vllm-ascend/vllm_ascend/attention/dsa_v1.py
# =====================================================================


# --- AscendDSAMetadataBuilder.build_prefill_metadata ---

def build_prefill_metadata(
    self,
    common_prefix_len: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    num_reqs_actual: int | None,
) -> AscendDSAPrefillMetadata:
    assert self.prefill_ratio_to_sas_metadata is not None
    assert self.decode_ratio_to_sas_metadata is not None
    query_start_loc = common_attn_metadata.query_start_loc

    # reqs_start: the start request position of prefill request
    reqs_start = self.num_decodes
    # reqs_start: the start token position of prefill request
    tokens_start = self.num_decode_tokens

    if self.prefill_ratio_to_sas_metadata.get("prefill_input_positions", None) is None:
        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()
        max_query_len = self.query_lens[reqs_start:].max().item()
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seq_lens = _seq_lens_cpu[reqs_start:].max().item()
        self.prefill_ratio_to_sas_metadata["input_positions"] = input_positions
        self.prefill_ratio_to_sas_metadata["max_query_len"] = max_query_len
        self.prefill_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens

        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        prefill_input_positions = input_positions[tokens_start:]
        self.prefill_ratio_to_sas_metadata["prefill_input_positions"] = prefill_input_positions
        self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"] = prefill_query_start_loc

        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)
        self.prefill_ratio_to_sas_metadata["cos"] = cos
        self.prefill_ratio_to_sas_metadata["sin"] = sin

        prefill_seq_lens = self.seq_lens[reqs_start:]
        num_prefill = prefill_seq_lens.shape[0]
        self.prefill_ratio_to_sas_metadata["prefill_seq_lens"] = prefill_seq_lens
        self.prefill_ratio_to_sas_metadata["num_prefill"] = num_prefill
    else:
        input_positions = self.prefill_ratio_to_sas_metadata["input_positions"]
        max_query_len = self.prefill_ratio_to_sas_metadata["max_query_len"]
        max_seq_lens = self.prefill_ratio_to_sas_metadata["max_seq_lens"]
        prefill_input_positions = self.prefill_ratio_to_sas_metadata["prefill_input_positions"]
        prefill_query_start_loc = self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"]
        cos = self.prefill_ratio_to_sas_metadata["cos"]
        sin = self.prefill_ratio_to_sas_metadata["sin"]
        prefill_seq_lens = self.prefill_ratio_to_sas_metadata["prefill_seq_lens"]
        num_prefill = self.prefill_ratio_to_sas_metadata["num_prefill"]

    assert self.start_pos_prefill is not None
    self.start_pos_prefill.fill_(0)
    seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
    self.start_pos_prefill[:num_prefill] = self.seq_lens[reqs_start:] - seq_lens_q
    num_prefills_actual = num_prefill
    if num_reqs_actual is not None:
        num_prefills_actual = max(min(num_reqs_actual - reqs_start, num_prefill), 0)
        if num_prefills_actual < num_prefill:
            self.start_pos_prefill[num_prefills_actual:num_prefill].fill_(0)
            self.block_table[
                reqs_start + num_prefills_actual : reqs_start + num_prefill,
                ...,
            ].fill_(0)

    layer_name = f"c{self.compressor_ratio}"
    full_compress_cos, full_compress_sin = None, None
    if self.compressor_ratio > 1:
        # Keep only graph inputs here. The compressor metadata op itself is
        # launched in forward at the real compressor consumer.
        num_compressed_tokens = self._num_compressor_metadata_rows(
            BUILD_METADATA_STEP_PREFILL,
            common_attn_metadata,
        )
        full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
        prefill_slot_mapping = None
    else:
        num_compressed_tokens = self.num_prefill_tokens
        prefill_slot_mapping = self.slot_mapping[tokens_start : tokens_start + self.num_prefill_tokens]

    tp_size = get_tensor_model_parallel_world_size()
    n_local_heads = _get_dsa_local_heads(self.vllm_config, self.model_config.hf_config.num_attention_heads, tp_size)
    index_topk = self.model_config.hf_config.index_topk

    cu_c4_cmp_seqlen_list = None
    cu_c128_cmp_seqlen_list = None

    metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
    metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
    if self.compressor_ratio <= 1:
        if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
            self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=prefill_query_start_loc,
                cu_seqlens_ori_kv=prefill_query_start_loc,
                cu_seqlens_cmp_kv=None,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[reqs_start:],
                max_seqlen_q=seq_lens_q.max(),
                max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                batch_size=len(self.seq_lens[reqs_start:]),
                cmp_ratio=1,
                ori_mask_mode=4,  # 4:sliding window
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=False,
            )
        sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
    elif self.compressor_ratio == 4:
        if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
            self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=prefill_query_start_loc,
                cu_seqlens_ori_kv=prefill_query_start_loc,
                cu_seqlens_cmp_kv=cu_c4_cmp_seqlen_list,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[reqs_start:],
                max_seqlen_q=seq_lens_q.max(),
                max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                batch_size=len(self.seq_lens[reqs_start:]),
                cmp_topk=index_topk,
                # topk=index_topk,
                cmp_ratio=4,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=True,
            )
        sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
    else:
        if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
            self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=prefill_query_start_loc,
                cu_seqlens_ori_kv=prefill_query_start_loc,
                cu_seqlens_cmp_kv=cu_c128_cmp_seqlen_list,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[reqs_start:],
                max_seqlen_q=seq_lens_q.max(),
                max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                batch_size=len(self.seq_lens[reqs_start:]),
                cmp_ratio=128,  #
                ori_mask_mode=4,  # 4:sliding window
                cmp_mask_mode=3,  # 3:causal
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=True,
            )
        sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
    if self.prefill_ratio_to_sas_metadata.get("qli") is None:
        self.prefill_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=prefill_query_start_loc[1:].clone(),
            actual_seq_lengths_key=self.seq_lens[reqs_start:].clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,  # 128
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=len(self.seq_lens[reqs_start:]),
            max_seqlen_q=seq_lens_q.max().item(),
            max_seqlen_k=self.seq_lens[reqs_start:].max().item(),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,  # 512
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )
    qli_metadata = self.prefill_ratio_to_sas_metadata.get("qli")

    return AscendDSAPrefillMetadata(
        attn_mask=None,
        query_lens=self.query_lens[reqs_start:].to(torch.int32),
        seq_lens=self.seq_lens[reqs_start:],
        context_lens=self.seq_lens[reqs_start:],
        input_positions=prefill_input_positions,
        block_table=self.block_table[reqs_start:, ...],
        slot_mapping=prefill_slot_mapping,
        block_size=self.block_size,
        num_compressed_tokens=num_compressed_tokens,
        max_query_len=max_query_len,
        max_seq_lens=max_seq_lens,
        query_start_loc=prefill_query_start_loc,
        sin=sin,
        cos=cos,
        full_compress_sin=full_compress_sin,
        full_compress_cos=full_compress_cos,
        start_pos=self.start_pos_prefill[:num_prefill],
        num_reqs_actual=num_prefills_actual,
        sas_metadata=sas_metadata,
        qli_metadata=qli_metadata,
        cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
        cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list,
    )

# --- AscendDSAMetadataBuilder.build_decode_metadata ---

def build_decode_metadata(
    self,
    common_prefix_len: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    num_reqs_actual: int | None,
) -> AscendDSADecodeMetadata:
    assert self.decode_ratio_to_sas_metadata is not None
    if self.decode_ratio_to_sas_metadata.get("query_start_loc", None) is None:
        query_start_loc = common_attn_metadata.query_start_loc[: self.num_decodes + 1]
        self.decode_ratio_to_sas_metadata["query_start_loc"] = query_start_loc
        input_positions = common_attn_metadata.positions[: self.num_decode_tokens].long()
        self.decode_ratio_to_sas_metadata["input_positions"] = input_positions
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
        self.decode_ratio_to_sas_metadata["cos"] = cos
        self.decode_ratio_to_sas_metadata["sin"] = sin

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: self.num_decodes + 1]

        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seq_lens = _seq_lens_cpu[: self.num_decodes].max().item()
        seq_lens_list = _seq_lens_cpu[: self.num_decodes].tolist()
        self.decode_ratio_to_sas_metadata["query_start_loc_cpu"] = query_start_loc_cpu
        self.decode_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens
        self.decode_ratio_to_sas_metadata["seq_lens_list"] = seq_lens_list

        max_seqlen_kv = torch.max(_seq_lens_cpu[: self.num_decodes]).item()
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        self.decode_ratio_to_sas_metadata["max_seqlen_kv"] = max_seqlen_kv
        self.decode_ratio_to_sas_metadata["max_seqlen_q"] = max_seqlen_q

        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        start_pos_decode = self.seq_lens[: self.num_decodes] - seq_lens_q
        self.decode_ratio_to_sas_metadata["start_pos_decode"] = start_pos_decode
    else:
        query_start_loc = self.decode_ratio_to_sas_metadata["query_start_loc"]
        input_positions = self.decode_ratio_to_sas_metadata["input_positions"]
        cos = self.decode_ratio_to_sas_metadata["cos"]
        sin = self.decode_ratio_to_sas_metadata["sin"]
        query_start_loc_cpu = self.decode_ratio_to_sas_metadata["query_start_loc_cpu"]
        max_seq_lens = self.decode_ratio_to_sas_metadata["max_seq_lens"]
        seq_lens_list = self.decode_ratio_to_sas_metadata["seq_lens_list"]
        max_seqlen_kv = self.decode_ratio_to_sas_metadata["max_seqlen_kv"]
        max_seqlen_q = self.decode_ratio_to_sas_metadata["max_seqlen_q"]
        start_pos_decode = self.decode_ratio_to_sas_metadata["start_pos_decode"]

    block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)

    cp_seq_len, batch_seq_mask = None, None

    assert self.start_pos_decode is not None
    self.start_pos_decode.fill_(0)
    self.start_pos_decode[: self.num_decodes] = start_pos_decode

    if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
        self.start_pos_decode[num_reqs_actual:].fill_(0)
        self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)
    num_decodes_actual = min(num_reqs_actual, self.num_decodes) if num_reqs_actual is not None else self.num_decodes

    layer_name = f"c{self.compressor_ratio}"
    full_compress_cos, full_compress_sin = None, None
    if self.compressor_ratio > 1:
        # Keep only graph inputs here. The compressor metadata op itself is
        # launched in forward at the real compressor consumer.
        num_compressed_tokens = self._num_compressor_metadata_rows(
            BUILD_METADATA_STEP_DECODE,
            common_attn_metadata,
        )
        full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
        slot_mapping = None
    else:
        num_compressed_tokens = self.num_decode_tokens
        slot_mapping = DeviceOperator.pad_dsa_decode_slot_mapping(
            self.slot_mapping[: self.num_decode_tokens],
            self.num_decode_tokens,
            self.compressor_ratio,
            self.num_decodes,
        )

    tp_size = get_tensor_model_parallel_world_size()
    n_local_heads = _get_dsa_local_heads(self.vllm_config, self.model_config.hf_config.num_attention_heads, tp_size)
    index_topk = self.model_config.hf_config.index_topk

    assert self.decode_sas_metadata is not None

    cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
        self.decode_ratio_to_sas_metadata,
        "cu_seqlens_ori_kv",
        self.seq_lens,
        self.num_decodes,
        self._zero_i32,
        self.cu_seqlens_ori_kv,
    )
    metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
    metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
    cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
    if self.compressor_ratio <= 1:
        if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
            self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,  # cached
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                cmp_ratio=1,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=False,
            )
        self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
    elif self.compressor_ratio == 4:
        if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
            self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,  # cached
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                cmp_topk=index_topk,
                # topk=index_topk,
                cmp_ratio=4,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=True,
            )
        self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
    else:
        if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
            self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=self.seq_lens[: self.num_decodes],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(self.seq_lens[: self.num_decodes]),
                cmp_ratio=128,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=True,
            )
        self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
    assert self.decode_qli_metadata is not None
    if self.decode_ratio_to_sas_metadata.get("qli") is None:
        self.decode_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=query_start_loc[1:].clone(),
            actual_seq_lengths_key=self.seq_lens[: self.num_decodes].clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,  # 128
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=len(self.seq_lens[: self.num_decodes]),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,  # 512
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )
    self.decode_qli_metadata[:1024] = self.decode_ratio_to_sas_metadata.get("qli")
    decode_metadata = AscendDSADecodeMetadata(
        input_positions=input_positions,
        block_table=self.block_table[:block_table_size, ...],
        slot_mapping=slot_mapping,
        block_size=self.block_size,
        num_compressed_tokens=num_compressed_tokens,
        seq_lens=self.seq_lens[: self.num_decodes],  # cached
        seq_lens_list=seq_lens_list,
        max_seq_lens=max_seq_lens,
        max_seqlen_kv=max_seqlen_kv,
        max_seqlen_q=max_seqlen_q,
        attn_mask=None,
        query_start_loc=query_start_loc,  # cached
        query_start_loc_cpu=query_start_loc_cpu,
        sin=sin[: self.num_decode_tokens, ...],
        cos=cos[: self.num_decode_tokens, ...],
        full_compress_sin=full_compress_sin,
        full_compress_cos=full_compress_cos,
        cp_seq_len=cp_seq_len,
        batch_seq_mask=batch_seq_mask,
        start_pos=self.start_pos_decode[: self.num_decodes],  # cached
        num_reqs_actual=num_decodes_actual,
        sas_metadata=self.decode_sas_metadata,
        qli_metadata=self.decode_qli_metadata,
    )
    return decode_metadata

# --- AscendDSAMetadataBuilder.build_prefill_metadata_for_drafting ---

def build_prefill_metadata_for_drafting(
    self,
    draft_index: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    **kwargs,
) -> AscendDSAPrefillMetadata:
    tp_size = get_tensor_model_parallel_world_size()
    n_local_heads = _get_dsa_local_heads(self.vllm_config, self.model_config.hf_config.num_attention_heads, tp_size)
    reqs_start = kwargs.get("reqs_start")
    tokens_start = kwargs.get("tokens_start")
    num_prefill_tokens = kwargs.get("num_prefill_tokens")
    query_start_loc = common_attn_metadata.query_start_loc
    prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
    seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
    seq_lens = common_attn_metadata.seq_lens[reqs_start:]

    num_actual_tokens = common_attn_metadata.num_actual_tokens
    input_positions = common_attn_metadata.positions[:num_actual_tokens].long()
    prefill_input_positions = input_positions[tokens_start:]
    cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

    prefill_slot_mapping = self.spec_slot_mapping[draft_index - 1][tokens_start:num_prefill_tokens]  # type: ignore[index]
    block_table = common_attn_metadata.block_table_tensor[: common_attn_metadata.num_reqs]

    metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
    metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
    sas_metadata = metadata_op(
        **metadata_kwargs,
        num_heads_q=n_local_heads,
        num_heads_kv=1,
        head_dim=self.model_config.get_head_size(),
        cu_seqlens_q=prefill_query_start_loc,
        cu_seqlens_ori_kv=prefill_query_start_loc,
        cu_seqlens_cmp_kv=None,
        seqused_q=self.seqused_q,
        seqused_kv=seq_lens,
        max_seqlen_q=seq_lens_q.max(),
        max_seqlen_kv=seq_lens.max(),
        batch_size=len(seq_lens),
        cmp_ratio=1,
        ori_mask_mode=4,
        ori_win_left=self.model_config.hf_config.sliding_window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )

    return AscendDSAPrefillMetadata(
        attn_mask=None,
        query_lens=None,
        seq_lens=seq_lens,
        context_lens=None,
        input_positions=None,  # type: ignore[arg-type]
        block_table=block_table[reqs_start:, ...],
        slot_mapping=prefill_slot_mapping,
        block_size=self.block_size,
        max_query_len=None,  # type: ignore[arg-type]
        max_seq_lens=None,  # type: ignore[arg-type]
        query_start_loc=prefill_query_start_loc,
        sin=sin,
        cos=cos,
        start_pos=None,
        sas_metadata=sas_metadata,
        qli_metadata=None,
        cu_c4_cmp_seqlen_list=None,
        cu_c128_cmp_seqlen_list=None,
    )

# --- AscendDSAMetadataBuilder.build_decode_metadata_for_drafting ---

def build_decode_metadata_for_drafting(
    self,
    draft_index: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    **kwargs,
) -> AscendDSADecodeMetadata:
    tp_size = get_tensor_model_parallel_world_size()
    n_local_heads = _get_dsa_local_heads(self.vllm_config, self.model_config.hf_config.num_attention_heads, tp_size)
    num_decodes = kwargs.get("num_decodes")
    num_decode_tokens = kwargs.get("num_decode_tokens")
    num_decodes_typed = num_decodes or 0
    num_decode_tokens_typed = num_decode_tokens or 0
    query_start_loc = common_attn_metadata.query_start_loc[: num_decodes_typed + 1]
    seq_lens = common_attn_metadata.seq_lens
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_decodes_typed + 1]
    max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()

    if common_attn_metadata._seq_lens_cpu is not None:
        _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
    elif common_attn_metadata.seq_lens_cpu is not None:
        _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
    else:
        _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
    max_seqlen_kv = torch.max(_seq_lens_cpu[:num_decodes]).item()

    input_positions = common_attn_metadata.positions[:num_decode_tokens_typed].long()
    # disable use_cache, otherwise, draft_index>0 will override draft_index=0
    # take care of this, if full graph is needed then rope cache is inevitable
    cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True, draft_index=draft_index)

    slot_mapping = self.spec_slot_mapping[draft_index - 1][:num_decode_tokens_typed]  # type: ignore[index]
    block_table = common_attn_metadata.block_table_tensor

    metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
    metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)

    decode_sas_metadata = metadata_op(
        **metadata_kwargs,
        num_heads_q=n_local_heads,
        num_heads_kv=1,
        head_dim=self.model_config.get_head_size(),
        cu_seqlens_q=query_start_loc,
        cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
        seqused_q=self.seqused_q,
        seqused_kv=seq_lens[:num_decodes],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=len(seq_lens[:num_decodes]),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=self.model_config.hf_config.sliding_window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    self.spec_sas_metadata[draft_index - 1][:1024].copy_(decode_sas_metadata[:1024])
    decode_sas_metadata = self.spec_sas_metadata[draft_index - 1]

    decode_metadata = AscendDSADecodeMetadata(
        input_positions=None,
        block_table=block_table[:num_decodes, ...],
        slot_mapping=slot_mapping,
        block_size=self.block_size,
        seq_lens=seq_lens[:num_decodes],
        seq_lens_list=None,  # type: ignore[arg-type]
        max_seq_lens=None,  # type: ignore[arg-type]
        max_seqlen_kv=None,  # type: ignore[arg-type]
        max_seqlen_q=None,  # type: ignore[arg-type]
        attn_mask=None,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=None,
        sin=sin[:num_decode_tokens, ...],
        cos=cos[:num_decode_tokens, ...],
        cp_seq_len=None,
        batch_seq_mask=None,
        start_pos=None,
        sas_metadata=decode_sas_metadata,
        qli_metadata=None,
    )
    return decode_metadata


# --- AscendDSAImpl.forward ---

def dsa_impl_forward(  # type: ignore[override]
    self,
    layer_name,
    hidden_states: torch.Tensor,  # query in unified attn
    kv_cache: tuple[torch.Tensor, ...] | None,
    attn_metadata: DSAMetadataList,
    need_gather_q_kv: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided."
    output_padded = output
    forward_context = get_forward_context()
    o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads, self.head_dim)
    if attn_metadata is None:
        # Profiling run: run o_proj on zero input so HCCL collectives are
        # captured by the ACL graph.  Non-OTP just zeros the output.
        if oproj_tp_enable():
            o_proj_input = torch.zeros(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            self._forward_o_proj(o_proj_input, output)
        else:
            output.fill_(0)
        return output
    if not isinstance(attn_metadata, list):
        attn_metadata = [attn_metadata]
    # Process for Flash Comm V1
    has_prefill = attn_metadata[0].num_prefills > 0
    has_decode = attn_metadata[0].num_decodes > 0
    decode_tokens = attn_metadata[0].num_decode_tokens
    actual_tokens = attn_metadata[0].num_actual_tokens

    # Process for Flash Comm V1
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
    prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
    decode_hidden_states = hidden_states[:decode_tokens]

    o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    assert kv_cache is not None, "kv_cache tensor tuple must be provided."
    wait_for_kv_layer_from_connector(layer_name)
    if has_prefill:
        assert attn_metadata[0].prefill is not None
        output_prefill = self._forward_prefill(
            layer_name,
            prefill_hidden_states,
            kv_cache,
            attn_metadata,
        )  # type: ignore[arg-type]
        o_proj_input[decode_tokens:actual_tokens] = output_prefill
        cos = attn_metadata[0].prefill.cos[layer_name]
        sin = attn_metadata[0].prefill.sin[layer_name]

    if has_decode:
        assert attn_metadata[0].decode is not None
        output_decode = self._forward_decode(layer_name, decode_hidden_states, kv_cache, attn_metadata)
        o_proj_input[:decode_tokens] = output_decode
        cos = attn_metadata[0].decode.cos[layer_name]
        sin = attn_metadata[0].decode.sin[layer_name]

    cos = attn_metadata[0].cos[layer_name]
    sin = attn_metadata[0].sin[layer_name]

    # FIX (hetero): the padding rows [actual_tokens:num_tokens] come from
    # torch.empty() and hold uninitialized (often NaN) memory.  Feeding NaN
    # into the rope kernel makes its vectorized output diverge per-rank
    # (bit-identical input + cos -> different output across DP groups with
    # different n_local_heads).  Zero them BEFORE the rope, and rope ONLY
    # the real rows [0:actual_tokens] — exactly the shape the (clean) q-rope
    # uses — so the o_proj rope never sees a padding row or a NaN.
    if actual_tokens < o_proj_input.shape[0]:
        o_proj_input[actual_tokens:] = 0

    torch.ops._C_ascend.inplace_partial_rotary_mul(
        o_proj_input[:actual_tokens].unsqueeze(1),
        cos[:actual_tokens],
        -sin[:actual_tokens],
        rotary_mode="interleave",
        partial_slice=[self.nope_head_dim, self.head_dim],
    )

    # o
    self._forward_o_proj(o_proj_input, output)

    maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

    return output_padded

def _patched_build_local_token_metadata(
    self,
    num_reqs,
    num_input_tokens,
    input_positions,
    query_start_loc,
    seq_lens,
    use_cache,
    local_query_start_loc=None,
    local_seq_lens=None,
):
    """Wrapper for AscendDSACPMetadataBuilder._build_local_token_metadata.

    Under heterogeneous TP the FlashComm1 SP stream is padded to the LCM of all
    per-DP tp sizes (e.g. lcm(3,4)=12), not just the local tp_size.  Align
    num_input_tokens to that LCM before the original implementation computes
    num_tokens_pad = ceil(num_input_tokens / tp_size) * tp_size.

    The LCM alignment must NOT run for draft-model metadata: the MTP draft
    forward runs with flash_comm_v1_enabled=False (the hidden stream has only
    the real tokens), so padding local_cos / attention output to the cross-DP
    LCM makes them wider than the hidden-state buffer and crashes either in
    the RoPE kernel or later in ``output[...] = ...``.
    ``AscendSpecDecodeBaseProposer`` marks its DSA-CP draft builders with
    ``_is_dsa_cp_draft_builder``.
    """
    if self.vllm_config.parallel_config.is_heterogeneous_tp:
        draft_builder = getattr(self, "_is_dsa_cp_draft_builder", False)
        # Dense drafters run with flash_comm_v1_enabled=False and must NOT be
        # LCM-padded.  MoE drafters (e.g. DeepSeek-V4 MTP) are different:
        # set_ascend_forward_context marks them as a context MoE model and
        # keeps flash_comm_v1_enabled=True, so their hidden stream IS padded
        # to lcm(tp_sizes) and reduce_scattered before the first draft layer.
        # In that case the draft metadata must use the exact same LCM padding;
        # the local-TP fallback below makes local_cos wider/narrower than q and
        # the inplace_partial_rotary_mul kernel reports ``dim0 must be equal``.
        use_lcm_alignment = not draft_builder
        if draft_builder:
            try:
                from vllm_ascend.utils import is_drafter_moe_model

                use_lcm_alignment = is_drafter_moe_model(self.vllm_config)
            except Exception:  # noqa: BLE001
                use_lcm_alignment = False
        if use_lcm_alignment:
            align = math.lcm(
                *[
                    self.vllm_config.parallel_config.get_tp_size_for_dp(i)
                    for i in range(
                        self.vllm_config.parallel_config.data_parallel_size
                    )
                ]
            )
            num_input_tokens = math.ceil(num_input_tokens / align) * align
    return _ORIG_BUILD_LOCAL_TOKEN_METADATA(
        self,
        num_reqs,
        num_input_tokens,
        input_positions,
        query_start_loc,
        seq_lens,
        use_cache,
        local_query_start_loc,
        local_seq_lens,
    )


def _patched_dsa_cp_init(self, *args, **kwargs):
    """Wrapper for AscendDSACPImpl.__init__.

    Runs the original initializer, then records heterogeneous TP head sharding
    ratios and disables the A5 o_proj full-gather path for ratio-sharded
    groups.
    """
    _ORIG_DSA_CP_INIT(self, *args, **kwargs)
    self._hetero_head_ratios = None
    vllm_config = get_current_vllm_config()
    if vllm_config.parallel_config.is_heterogeneous_tp:
        ratios = vllm_config.parallel_config.get_sharding_ratios_for_dp(
            vllm_config.parallel_config.data_parallel_rank
        )
        if ratios is not None:
            self._hetero_head_ratios = list(ratios)
            self.enable_dsa_cp_with_o_proj_tp = False
            # The original __init__ was called with the UNIFORM head count
            # (e.g. 16 at tp=4) because DeepseekV4Attention is patched from
            # the outside after its DSA impl was already constructed.  The
            # impl object holds its own copies of n_local_heads /
            # n_local_groups, so overwrite them here too -- otherwise
            # _restore_tp_head_layout and the o_proj path use the wrong
            # shard width on DP0 (must be 32/16/16 heads for [2,1,1]).
            self.n_local_heads = get_tp_partition_size(
                self.num_heads, self.tp_rank, self.tp_size, ratios
            )
            self.n_local_groups = get_tp_partition_size(
                self.n_group, self.tp_rank, self.tp_size, ratios
            )


def _patched_dsa_cp_process_weights_after_loading(self, act_dtype):
    """Make ``wo_a`` 3-D for ``npu_transpose_batchmatmul``.

    With ModelSlim W8A8 DeepSeek-V4 the ``wo_a`` linear layer is
    intentionally unquantized, so the FP8 ``process_weights_after_loading``
    reshape to ``[n_local_groups, input, o_lora_rank]`` never runs.  The
    DSA-CP forward passes ``self.wo_a.weight`` directly to
    ``npu_transpose_batchmatmul(perm_x2=(0, 1, 2))``, which requires a 3-D
    weight and otherwise fails with ``IndexError: ... got 2`` on a 2-D
    parameter.
    """
    _ORIG_DSA_CP_PROCESS_WEIGHTS(self, act_dtype)
    weight = self.wo_a.weight
    if weight.dim() == 2 and not self.enable_dsa_cp_with_o_proj_tp:
        weight.data = (
            weight.data.view(self.n_local_groups, self.o_lora_rank, -1)
            .transpose(1, 2)
            .contiguous()
        )



# =====================================================================
# Copied from hetero_cp/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py
# =====================================================================


# --- AscendDSACPImpl._restore_tp_head_layout ---

def restore_tp_head_layout(
    self,
    local_attn_output: torch.Tensor,
    layer_name: str,
    attn_metadata: M,
    skip_all_to_all: bool = False,
) -> torch.Tensor:
    assert attn_metadata.req_metadata is not None
    req_metadata = attn_metadata.req_metadata
    cp_metadata = req_metadata.cp_metadata
    num_tokens = local_attn_output.shape[0]
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        local_attn_output.unsqueeze(1),
        cp_metadata.local_cos[layer_name],
        -cp_metadata.local_sin[layer_name],
        rotary_mode="interleave",
        partial_slice=[self.nope_head_dim, self.head_dim],
    )

    if self.tp_size == 1 or skip_all_to_all:
        return local_attn_output

    if self._hetero_head_ratios is None:
        send = (
            local_attn_output.view(
                num_tokens,
                self.tp_size,
                self.n_local_heads,
                self.head_dim,
            )
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, self.n_local_heads, self.head_dim)
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.tp_group.device_group)
        return recv

    # Heterogeneous TP with asymmetric head sharding (e.g. tp=3 with
    # ratios [2,1,1] -> 32/16/16 local heads).  Every rank computed all
    # ``num_heads`` locally, but the output shard width differs per rank,
    # so a uniform all_to_all_single cannot split/assemble the tensor.
    # Pack each destination's head partition into a contiguous chunk and
    # exchange with explicit per-source/per-destination split sizes.
    head_sizes = [
        get_tp_partition_size(
            self.num_heads, rank, self.tp_size, self._hetero_head_ratios
        )
        for rank in range(self.tp_size)
    ]
    head_offsets = [
        get_tp_partition_offset(
            self.num_heads, rank, self.tp_size, self._hetero_head_ratios
        )
        for rank in range(self.tp_size)
    ]
    send_chunks = []
    for rank in range(self.tp_size):
        head_start = head_offsets[rank]
        head_end = head_start + head_sizes[rank]
        send_chunks.append(
            local_attn_output[:, head_start:head_end, :]
            .contiguous()
            .view(-1)
        )
    send = torch.cat(send_chunks, dim=0)
    recv = torch.empty(
        num_tokens * self.tp_size * self.n_local_heads * self.head_dim,
        dtype=local_attn_output.dtype,
        device=local_attn_output.device,
    )
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=[
            num_tokens * self.n_local_heads * self.head_dim
        ] * self.tp_size,
        input_split_sizes=[
            num_tokens * head_sizes[rank] * self.head_dim
            for rank in range(self.tp_size)
        ],
        group=self.tp_group.device_group,
    )
    return recv.view(-1, self.n_local_heads, self.head_dim)

_DS_V1_PATCHED_SYMBOLS = (
    "AscendDSAMetadataBuilder.build_prefill_metadata",
    "AscendDSAMetadataBuilder.build_decode_metadata",
    "AscendDSAMetadataBuilder.build_prefill_metadata_for_drafting",
    "AscendDSAMetadataBuilder.build_decode_metadata_for_drafting",
    "AscendDSAImpl.forward",
)

_DS_CP_PATCHED_SYMBOLS = (
    "AscendDSACPMetadataBuilder._build_local_token_metadata",
    "AscendDSACPImpl.__init__",
    "AscendDSACPImpl._restore_tp_head_layout",
)


def apply_deepseek_v4_attention_hetero_patch() -> dict[str, list[str]]:
    """Apply heterogeneous-TP DSA attention patches to installed vllm_ascend.

    Idempotent.  The dsa_cp wrappers that must call their saved original are
    installed as replacement function attributes (not by swapping ``__code__``):
    swapping code mutates the saved original function object in-place, so the
    wrapper's ``original`` reference ends up pointing at itself and recurses.
    """
    global _ATTENTION_HETERO_PATCH_APPLIED
    global _ORIG_BUILD_LOCAL_TOKEN_METADATA, _ORIG_DSA_CP_INIT
    global _ORIG_DSA_CP_PROCESS_WEIGHTS
    if _ATTENTION_HETERO_PATCH_APPLIED:
        return {
            "vllm_ascend.attention.dsa_v1": list(_DS_V1_PATCHED_SYMBOLS),
            "vllm_ascend.attention.context_parallel.dsa_cp": list(_DS_CP_PATCHED_SYMBOLS),
        }

    from vllm_ascend.attention import dsa_v1 as dsa_v1_module
    from vllm_ascend.attention.context_parallel import dsa_cp as dsa_cp_module

    # Helper globals referenced by the copied method bodies.
    dsa_v1_module.__dict__.setdefault("_get_dsa_local_heads", _get_dsa_local_heads)
    dsa_cp_module.__dict__.setdefault("get_tp_partition_size", get_tp_partition_size)
    dsa_cp_module.__dict__.setdefault("get_tp_partition_offset", get_tp_partition_offset)

    # Save original dsa_cp method function objects in this module, then bind
    # the wrappers as new class attributes.  The originals keep their own
    # ``__code__`` and ``__globals__``, so calling them cannot recurse.
    _ORIG_BUILD_LOCAL_TOKEN_METADATA = (
        dsa_cp_module.AscendDSACPMetadataBuilder._build_local_token_metadata
    )
    _ORIG_DSA_CP_INIT = dsa_cp_module.AscendDSACPImpl.__init__
    _ORIG_DSA_CP_PROCESS_WEIGHTS = (
        dsa_cp_module.AscendDSACPImpl.process_weights_after_loading
    )

    # dsa_v1 metadata builders and forward.
    dsa_v1_module.AscendDSAMetadataBuilder.build_prefill_metadata.__code__ = (
        build_prefill_metadata.__code__
    )
    dsa_v1_module.AscendDSAMetadataBuilder.build_decode_metadata.__code__ = (
        build_decode_metadata.__code__
    )
    dsa_v1_module.AscendDSAMetadataBuilder.build_prefill_metadata_for_drafting.__code__ = (
        build_prefill_metadata_for_drafting.__code__
    )
    dsa_v1_module.AscendDSAMetadataBuilder.build_decode_metadata_for_drafting.__code__ = (
        build_decode_metadata_for_drafting.__code__
    )
    dsa_v1_module.AscendDSAImpl.forward.__code__ = dsa_impl_forward.__code__

    # dsa_cp builder/impl helpers.  ``__init__`` and
    # ``_build_local_token_metadata`` are wrapper function objects that call
    # the saved originals through this module's globals, so bind them directly.
    dsa_cp_module.AscendDSACPMetadataBuilder._build_local_token_metadata = (
        _patched_build_local_token_metadata
    )
    dsa_cp_module.AscendDSACPImpl.__init__ = _patched_dsa_cp_init
    # W8A8 modelslim keeps wo_a unquantized (2-D); reshape it to the 3-D
    # layout npu_transpose_batchmatmul expects after the original processing.
    dsa_cp_module.AscendDSACPImpl.process_weights_after_loading = (
        _patched_dsa_cp_process_weights_after_loading
    )
    # ``_restore_tp_head_layout`` is a standalone copied body whose globals
    # must stay the target module namespace (torch, dist, helpers above), so
    # code-object replacement is the right installation method for it.
    dsa_cp_module.AscendDSACPImpl._restore_tp_head_layout.__code__ = (
        restore_tp_head_layout.__code__
    )

    _ATTENTION_HETERO_PATCH_APPLIED = True
    return {
        "vllm_ascend.attention.dsa_v1": list(_DS_V1_PATCHED_SYMBOLS),
        "vllm_ascend.attention.context_parallel.dsa_cp": list(_DS_CP_PATCHED_SYMBOLS),
    }

