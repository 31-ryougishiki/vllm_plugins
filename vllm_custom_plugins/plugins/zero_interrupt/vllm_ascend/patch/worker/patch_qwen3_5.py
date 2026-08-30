#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# mypy: ignore-errors

import torch
from einops import rearrange
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention as _GDNBaseCls,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

try:
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors
except ImportError:
    Qwen3_5MultiTokenPredictor = None
    IntermediateTensors = None
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.utils import is_310p

_GDN_PATCH_TARGET = _GDNBaseCls


class AscendQwen3_5GatedDeltaNet(_GDNBaseCls):
    """v0.23.0-compatible Qwen3.5 GDN forward with asymmetric-TP support.

    0829 分支的自定义 forward：按 ``tp_asymmetric_shardings`` 计算本 rank
    的 qkv/z/v-head 局部尺寸；对称 TP 时退化为普通 ``// tp_size`` 切分。
    真正的 ``_forward_core`` 统一委托给 v0.23 的
    ``vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention``。
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        asym = getattr(self, "asym", False)
        shardings = getattr(self, "tp_asymmetric_shardings", None)
        if not (asym and shardings is not None):
            # Symmetric TP：完整保留 v0.23 原实现，包括
            # in_proj_qkv / gqa_interleaved_layout 等分支。
            return AscendGatedDeltaNetAttention.forward(
                self, hidden_states, output
            )

        # Asymmetric TP local sizes (TP=3 with [1,1,2] etc.).
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        num_tokens = mixed_qkvz.size(0)
        world_split_size = sum(shardings)
        split_size = shardings[self.tp_rank]
        local_qkv_size = (
            (self.key_dim * 2 + self.value_dim)
            * split_size // world_split_size
        )
        local_z_size = self.value_dim * split_size // world_split_size
        local_num_v_heads = (
            self.num_v_heads * split_size // world_split_size
        )

        mixed_qkv, z = mixed_qkvz.split(
            [local_qkv_size, local_z_size], dim=-1
        )
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        # Part 2: core attention (custom op, dispatches to the patched
        # real v0.23.0 _forward_core).
        core_attn_out = torch.zeros(
            (num_tokens, local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
        )

        # Part 3: output projection.
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        o_out, _ = self.out_proj(core_attn_out)
        actual_num_tokens = o_out.shape[0]
        output[:actual_num_tokens] = o_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Asymmetric-TP-aware QKV split.

        非对称 TP 下 mixed_qkv 已经是本 rank 的局部长度，不能用
        ``self.tp_size`` 均分，需要按 ``tp_asymmetric_shardings`` 拆分。
        对称 TP 直接使用 vllm 基类实现。
        """
        if mixed_qkv is None:
            return None, None, None

        asym = getattr(self, "asym", False)
        shardings = getattr(self, "tp_asymmetric_shardings", None)
        if not (asym and shardings is not None):
            return _GDNBaseCls.rearrange_mixed_qkv(self, mixed_qkv)

        world_split_size = sum(shardings)
        split_size = shardings[self.tp_rank]
        local_key_size = self.key_dim * split_size // world_split_size
        local_value_size = self.value_dim * split_size // world_split_size

        query, key, value = torch.split(
            mixed_qkv,
            [local_key_size, local_key_size, local_value_size],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)

        return query.contiguous(), key.contiguous(), value.contiguous()


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            n_out = (hidden_states.shape[0] + tp_size - 1) // tp_size
            hidden_dim = hidden_states.shape[-1]
            self_attention_output = torch.empty(
                (n_out, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            self_attention_output = torch.empty_like(hidden_states)

        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


if Qwen3_5MultiTokenPredictor is not None:

    def qwen3_5_mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Backport upstream Qwen3.5 MTP behavior: the local drafter runs on the
        # last PP stage and should always combine token embeddings with the
        # target hidden states instead of consuming PP intermediate tensors.
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    Qwen3_5MultiTokenPredictor.forward = qwen3_5_mtp_forward


Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward

# v0.23.0 GDN 基础 patch，A/B 两个场景共用。
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend

if is_310p():
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310

    # 310P 保持 A/v0.23 原行为：只替换 _forward_core/state dtype，
    # forward 继续使用 QwenGatedDeltaNetAttention 基类实现。
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention310._forward_core
    _GDN_PATCH_TARGET.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
else:
    # 0829 非对称感知 forward + v0.23 真实 _forward_core。
    _GDN_PATCH_TARGET.forward = AscendQwen3_5GatedDeltaNet.forward
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    _GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels
    _GDN_PATCH_TARGET.rearrange_mixed_qkv = (
        AscendQwen3_5GatedDeltaNet.rearrange_mixed_qkv
    )
