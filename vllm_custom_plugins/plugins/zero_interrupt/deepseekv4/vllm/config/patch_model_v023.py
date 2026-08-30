# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v0.23-compatible ModelConfig.verify_with_parallel_config patch.

Kept in a standalone module so vllm_plugins does not need to import the
legacy v0.18 patch_model.py in v0.23 environments.
"""


def verify_with_parallel_config(self, parallel_config) -> None:
    total_num_attention_heads = (
        self.model_arch_config.total_num_attention_heads
    )
    tensor_parallel_size = parallel_config.tensor_parallel_size

    has_asymmetric = bool(
        getattr(parallel_config, "is_heterogeneous_tp", False)
        and parallel_config.get_sharding_ratios_for_dp(
            parallel_config.data_parallel_rank
        )
        is not None
    )
    if (
        not has_asymmetric
        and total_num_attention_heads % tensor_parallel_size != 0
    ):
        raise ValueError(
            f"Total number of attention heads ({total_num_attention_heads})"
            " must be divisible by tensor parallel size "
            f"({tensor_parallel_size})."
        )

    if parallel_config.enable_expert_parallel:
        self._verify_with_expert_parallelism()

    pipeline_parallel_size = parallel_config.pipeline_parallel_size
    if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
        self.architectures, self
    ):
        raise NotImplementedError(
            "Pipeline parallelism is not supported for this model. "
            "Supported models implement the `SupportsPP` interface."
        )

    decode_context_parallel_size = parallel_config.decode_context_parallel_size
    if decode_context_parallel_size > 1 and not self.use_mla:
        total_num_kv_heads = self.get_total_num_kv_heads()
        assert tensor_parallel_size > total_num_kv_heads, (
            f"tensor parallel size {tensor_parallel_size} must be greater "
            f"than total num kv heads {total_num_kv_heads} when enable "
            f"decode context parallel for GQA/MQA"
        )

        max_dcp_size = tensor_parallel_size // total_num_kv_heads
        assert decode_context_parallel_size <= max_dcp_size, (
            f"decode context parallel size must less than or equal to "
            f"(tensor parallel size {tensor_parallel_size} // total "
            f"num kv heads {total_num_kv_heads}) = {max_dcp_size}, "
            f"but got {decode_context_parallel_size}"
        )

        num_q_per_kv = total_num_attention_heads // total_num_kv_heads
        assert num_q_per_kv % decode_context_parallel_size == 0, (
            f"Total number of q per kv attn heads ({num_q_per_kv})"
            " must be divisible by dcp world size when enable "
            "decode context parallel for GQA "
            f"({parallel_config.decode_context_parallel_size})."
        )

    if (
        self.multimodal_config is not None
        and self.multimodal_config.mm_tensor_ipc == "torch_shm"
        and parallel_config.world_size_across_dp > 1
    ):
        raise ValueError(
            "mm_tensor_ipc='torch_shm' is not supported with "
            "data_parallel_size > 1 or tensor_parallel_size > 1 "
            "or pipeline_parallel_size > 1."
        )
