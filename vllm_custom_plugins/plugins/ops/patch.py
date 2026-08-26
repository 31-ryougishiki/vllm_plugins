"""
Ascend NPU custom operators patch for vLLM.

This module provides two mechanisms for Ascend NPU acceleration:
1. CustomOp.register_oot: For graph mode compatibility (ACLGraph)
2. Surgery patch: For eager mode

The forward_oot method is provided for graph mode, while the patch
applies forward replacement for eager mode.
"""
import os
import logging
from typing import Optional

import torch
import torch.nn as nn
# from vllm.model_executor.models.gemma4 import Gemma4Attention
from vllm.model_executor.models.qwen3 import Qwen3Attention as BaseQwen3Attention
from vllm.model_executor.models.qwen2 import Qwen2Attention as BaseQwen2Attention
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Attention
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

import vllm_ascend.ops.rotary_embedding as rotary_emb_module
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice
from vllm_ascend.patch.worker import patch_minimax_m2
from vllm_ascend.patch.worker import patch_qwen3_5
from vllm_custom_plugins.core import VLLMPatch, min_vllm_version

import ascend_custom_ops

logger = logging.getLogger("vllm_custom_plugins")


# For now(20260514) vllm-ascend v0.18 don't support deepseek v4
# deepseek v4 is supported through another docker image. 
# Code below is not compatible with v0.18, therefore add a env guard.
if os.getenv("PATCH_DEEPSEEKV4", None):
    from vllm_ascend.models.deepseek_v4 import DeepseekV2DecoderLayer as BaseDeepseekV2DecoderLayer

    @min_vllm_version("0.11.0")
    class HcPreForDeepseekV2DecoderLayer(BaseDeepseekV2DecoderLayer):
        def _patch_hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor,
                hc_scale: torch.Tensor, hc_base: torch.Tensor):

            y = torch.ops._C_its_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base,
                                            self.hc_mult,
                                            self.hc_sinkhorn_iters,
                                            self.norm_eps, self.hc_eps)
            return y



    @min_vllm_version("0.11.0")
    class HcPreForDeepseekV2DecoderLayerPatch(VLLMPatch[BaseDeepseekV2DecoderLayer]):
        def hc_pre(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor
        ) -> torch.Tensor:
            return self.hc_pre_oot(positions, hidden_states, hc_scale, hc_base)

        # 添佊|  forward_oot 潔¨乾N佛¾模廾O佅¼容
        hc_pre_oot = HcPreForDeepseekV2DecoderLayer._patch_hc_pre

@min_vllm_version("0.11.0")
class MRopeSplitForQwen3VL(BaseQwen3Attention):
    def forward_oot(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        positions = positions.contiguous() if not positions.is_contiguous() else positions
        is_vl = positions.shape[0]
        if is_vl == 3:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = torch.ops._C_its_ascend.npu_split_rmsnorm_mrope_qwen3vl(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output
        else :
            qkv, _ = self.qkv_proj(hidden_states)
            # 惰性初始化 _cos_sin_cache（只在第一次调用时）
            # 因为 DynamicNTKScalingRotaryEmbedding 不会被 CustomOp.register_oot 替换，
            # 所以 _record_cos_sin_cache() 不会被调用，需要手动初始化
            if rotary_emb_module._cos_sin_cache is None:
                rotary_emb_module._cos_sin_cache = self.rotary_emb.cos_sin_cache
            # 更新全局 _cos_slice 和 _sin_slice
            rotary_emb_module.update_cos_sin(positions)
            cos, sin = get_cos_and_sin_slice()
            q, k, v = torch.ops._C_its_ascend.npu_split_rmsnorm_rope_qwen3llm(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                sin,
                cos,
                self.num_heads,
                self.num_kv_heads,
            )
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output


@min_vllm_version("0.11.0")
class MRopeSplitForQwen3VLPatch(VLLMPatch[BaseQwen3Attention]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_oot(positions, hidden_states)

    # 添加 forward_oot 用于图模式兼容
    forward_oot = MRopeSplitForQwen3VL.forward_oot


@min_vllm_version("0.11.0")
class MRopeSplitForQwen2VL(BaseQwen2Attention):
    def forward_oot(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.qk_norm:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Apply QK normalization if enabled (before RoPE)

            # Reshape to apply per-head normalization
            # q shape: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            total_tokens = q.shape[0]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

            # Apply normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Reshape back
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)

            q, k = self.rotary_emb(positions, q, k)
        else:
            positions = positions.contiguous() if not positions.is_contiguous() else positions
            q, k, v = torch.ops._C_its_ascend.npu_split_mrope_qwen25vl(
                qkv,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

@min_vllm_version("0.11.0")
class MRopeSplitForQwen2VLPatch(VLLMPatch[BaseQwen2Attention]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_oot(positions, hidden_states)

    # 添加 forward_oot 用于图模式兼容
    forward_oot = MRopeSplitForQwen2VL.forward_oot


@min_vllm_version("0.11.0")
class RotaryEmbeddingPatch(VLLMPatch[rotary_emb_module]):
    """
        下面例子展示如何patch vllm_ascend.ops.rotary_embedding.py 中的update_cos_sin
    """
    def update_cos_sin(position):
        logger.info("RotaryEmbeddingPatch new")

        # replace origin global var with following
        # global _cos
        # global _sin
        # global _cos_slice
        # global _sin_slice
        # global _cos_sin_cache
        _cos = rotary_emb_module._cos
        _sin = rotary_emb_module._sin
        _sin = rotary_emb_module._cos_slice
        _sin = rotary_emb_module._sin_slice
        _cos_sin_cache = rotary_emb_module._cos_sin_cache

        if _cos_sin_cache is None or _cos is None or _sin is None:
            return

        num_tokens = positions.size(0)
        _cos[:, :num_tokens] = _cos_sin_cache.index_select(0, positions).view(
            num_tokens, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)[0]
        _sin[:, :num_tokens] = _cos_sin_cache.index_select(0, positions).view(
            num_tokens, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)[1]

        # write back to global
        rotary_emb_module._cos = _cos
        rotary_emb_module._sin = _sin
        rotary_emb_module._cos_slice = _cos[:, :num_tokens]
        rotary_emb_module._sin_slice = _sin[:, :num_tokens]


@min_vllm_version("0.18.0")
class SplitRmsRopeForQwen3NextPatch(VLLMPatch[Qwen3NextAttention]):
    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)

        if "qwen3_5" in self.config.model_type:
            positions = positions.contiguous() if not positions.is_contiguous() else positions
            if self.attn_output_gate:
                q, k, v, gate = torch.ops._C_its_ascend.npu_split_rmsnorm_mrope_gate_qwen35(
                    qkv, self.q_norm.weight, self.k_norm.weight,
                    self.rotary_emb.cos_sin_cache, positions,
                    self.num_heads, self.num_kv_heads, self.head_dim
                )
            else:
                q, k, v, gate = torch.ops._C_its_ascend.npu_split_rmsnorm_mrope_qwen35(
                    qkv, self.q_norm.weight, self.k_norm.weight,
                    self.rotary_emb.cos_sin_cache, positions,
                    self.num_heads, self.num_kv_heads, self.head_dim
                )
        else:
            cos, sin = get_cos_and_sin_slice()
            if self.attn_output_gate:
                q, k, v, gate = torch.ops._C_its_ascend.npu_split_rmsnorm_rope_gate_qwennext(
                    qkv, self.q_norm.weight, self.k_norm.weight,
                    sin, cos, self.num_heads, self.num_kv_heads, self.head_dim
                )
            else:
                q, k, v, gate = torch.ops._C_its_ascend.npu_split_rmsnorm_rope_qwennext(
                    qkv, self.q_norm.weight, self.k_norm.weight,
                    sin, cos, self.num_heads, self.num_kv_heads, self.head_dim
                )

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


# NOTE: when use this function, make sure the CustomOp you want to regsiter
# is not registered by vllm-ascend, otherwise it conflicts with vllm-ascend and will throw err
def register_ascend_ops():
    """
    Register Ascend custom ops via vLLM's CustomOp.register_oot mechanism.

    This enables the Ascend NPU implementation to be used automatically
    when vLLM detects an out-of-tree device (Ascend NPU), and is compatible
    with graph mode (ACLGraph).
    """
    if not ASCEND_OPS_AVAILABLE:
        logger.warning(
            "Cannot register Ascend ops: ascend_custom_ops not installed"
        )
        return False

    try:
        from vllm.model_executor.custom_op import CustomOp
        CustomOp.register_oot(_decorated_op_cls=AscendSiluAndMul, name="SiluAndMul")
        logger.info("Registered AscendSiluAndMul via CustomOp.register_oot")
        return True
    except Exception as e:
        logger.error(f"Failed to register Ascend ops: {e}")
        return False

@min_vllm_version("0.11.0")
class MinMaxPatchForward(VLLMPatch[patch_minimax_m2]):

    def _patch_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, qk_var = torch.ops._C_its_ascend.npu_qkv_slice_var(qkv, self.q_size, self.kv_size)

        if self.q_norm.tp_world > 1:
            from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
            qk_var = tensor_model_parallel_all_reduce(qk_var)
        q, k = torch.ops._C_its_ascend.npu_qk_norm_rope(
            qk_var,
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.q_norm.variance_epsilon,
            1.0/self.q_norm.tp_world
        )
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    MiniMaxM2Attention.forward = _patch_forward


# @min_vllm_version("0.18.0")
# class Rmsnorm2RopeNormForGemma4Patch(VLLMPatch[Gemma4Attention]):
#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         **kwargs,
#     ) -> torch.Tensor:
#         # Unified QKV path (works for both k_eq_v and standard layers).
#         qkv, _ = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#
#         if not self.is_kv_shared_layer:
#             q = q.unflatten(-1, (self.num_heads, self.head_dim)).contiguous()
#             k = k.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
#             v = v.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
#
#             q, k, v = torch.ops._C_its_ascend.npu_rmsnorm2_rope_norm(
#                 q, k, v, self.rotary_emb.cos_sin_cache,
#                 self.q_norm.weight, self.k_norm.weight, positions
#             )
#
#             q = q.flatten(-2, -1)
#             k = k.flatten(-2, -1)
#             v = v.flatten(-2, -1)
#
#         else:
#             # Q norm (always applied)
#             q = q.unflatten(-1, (self.num_heads, self.head_dim))
#             q = self.q_norm(q)
#             q = q.flatten(-2, -1)
#             positions = _maybe_align_positions(q, positions)
#
#            # Shared: only apply RoPE to Q
#             q = self.rotary_emb(positions, q, k)[0]
#
#         attn_output = self.attn(q, k, v)
#         output, _ = self.o_proj(attn_output)
#
#         return output
