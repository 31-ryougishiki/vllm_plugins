#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""ITS Plugin patch for vLLM Custom Plugins.

本模块提供将 ITS（零中断推理）插件集成到 vLLM 的主要 patch 类。

Patch 替换默认的 MultiprocExecutor 为 ITSMultiprocExecutor，
并应用 EngineCore patches 以支持部署策略执行。
"""

import logging

logger = logging.getLogger("vllm_custom_plugins")


def apply():
    """应用 ITS 插件 patches。

    此函数：
    1. Patch EngineCore.run_busy_loop 以支持部署策略执行
    2. 替换 MultiprocExecutor 为 ITSMultiprocExecutor
    3. 替换 WorkerProc 为 ITSNPUWorker
    """
    logger.info("Applying ITS plugin patch")

    # API server 在加载模型前就会校验 --tool-call-parser 是否注册。
    # 先补注册 deepseek_v4（不存在 v4 parser 时回退 v32/v31/v3），
    # 避免 api_server.validate_api_server_args 报 invalid tool call parser。
    try:
        from .vllm.entrypoints.openai.patch_tool_parser import (
            apply_deepseek_v4_tool_parser_patch,
        )
        apply_deepseek_v4_tool_parser_patch()
    except Exception as e:
        logger.warning(f"Failed to register deepseek_v4 tool parser: {e}")

    # 应用 EngineCore patch (使用相对导入)
    try:
        from .vllm.v1.engine import \
            engine_core_patch
        engine_core_patch.patch_engine_core()
    except ImportError as e:
        logger.warning(f"Failed to import engine_core_patch: {e}")

    # 应用 DPLBAsyncMPClient patch (客户端过滤)
    try:
        from .vllm.v1.engine import \
            core_client_patch
        core_client_patch.patch_dplb_client()
    except ImportError as e:
        logger.warning(f"Failed to import core_client_patch: {e}")

    # Patch MultiprocExecutor
    try:
        import vllm.v1.executor.multiproc_executor as mp_module
        from .vllm.v1.executor import ITSMultiprocExecutor, ITSNPUWorker

        mp_module.MultiprocExecutor = ITSMultiprocExecutor
        mp_module.WorkerProc = ITSNPUWorker
        logger.info("Replaced MultiprocExecutor with ITSMultiprocExecutor，Replaced WorkerProc with ITSNPUWorker")
    except ImportError as e:
        logger.warning(f"Failed to patch MultiprocExecutor: {e}")

    # 同时 patch AscendMultiprocExecutor（如果存在）
    try:
        import vllm_ascend.patch.platform.patch_multiproc_executor as ascend_module
        from .vllm.v1.executor import ITSMultiprocExecutor, ITSNPUWorker

        ascend_module.MultiprocExecutor = ITSMultiprocExecutor
        ascend_module.WorkerProc = ITSNPUWorker
        logger.info("Replaced AscendMultiprocExecutor with ITSMultiprocExecutor，Replaced WorkerProc with ITSNPUWorker")
    except ImportError as e:
        # AscendMultiprocExecutor 不可用，跳过
        logger.warning(f"Failed to patch ascend MultiprocExecutor: {e}")

    # 模型计算相关patch。
    # v0.23.0 的 AscendFusedMoE/expert_map 已经由 hetero patch 处理，
    # 不再用 0.18 时代的 PatchAscendFusedMoE 覆盖，避免旧 API 导入错误。
    try:
        import vllm
        from packaging.version import Version

        _vllm_ge_023 = vllm.__version__ == "dev" or Version(vllm.__version__) >= Version("0.23.0")
    except Exception:
        _vllm_ge_023 = False

    if _vllm_ge_023:
        logger.info("vllm>=0.23.0: skip legacy AscendFusedMoE replacement (hetero patches cover it)")
    else:
        try:
            import vllm_ascend.ops.fused_moe.fused_moe as ascend_fused_moe_module
            import vllm_ascend.eplb.core.eplb_utils as ascend_eplb_utils_module
            from .vllm_ascend.eplb.core.patch_eplb_utils import patched_init_eplb_config
            from .vllm_ascend.ops.fused_moe.patch_fused_moe import PatchAscendFusedMoE

            ascend_fused_moe_module.AscendFusedMoE = PatchAscendFusedMoE
            ascend_eplb_utils_module.init_eplb_config = patched_init_eplb_config
            logger.info("Replaced AscendFusedMoE with PatchAscendFusedMoE")
            logger.info("Replaced init_eplb_config with patched_init_eplb_config")
        except ImportError as e:
            # AscendFusedMoE 不可用，跳过
            logger.warning(f"Failed to patch AscendFusedMoE: {e}")

    logger.info("ITS plugin patch applied successfully")

    if _vllm_ge_023:
        # v0.23 路径：不加载 0.18 时代的 Qwen/varlen 模型 patch。
        # DeepSeek-V4 由后面的 heterogeneous-TP patches 处理。
        logger.info("vllm>=0.23.0: skip legacy Qwen/varlen asymmetric patches")

        import vllm.config.model
        from .vllm.config.patch_model_v023 import verify_with_parallel_config

        vllm.config.model.ModelConfig.verify_with_parallel_config = (
            verify_with_parallel_config
        )
        logger.info(
            "Replaced verify_with_parallel_config with v0.23 patched version"
        )

        # 该方法由 EngineCore 使用，提前 patch（使用 v0.23 同源实现）。
        from vllm.v1.core import kv_cache_utils
        from .vllm.v1.core.patch_kv_cache_utils import get_kv_cache_configs

        kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
        logger.info(
            "Replaced vllm.v1.core.kv_cache_utils.get_kv_cache_configs "
            "with v0.23 patched version"
        )
    else:
        # ============ vLLM 0.18 legacy patch path ============
        # qwen2
        import vllm.model_executor.models.qwen2 as qwen2_model
        from .vllm.model_executor.models.patch_qwen2 import qwen2_mlp_asymmetric_init
        qwen2_model.Qwen2MLP.__init__ = qwen2_mlp_asymmetric_init

        # qwen3
        import vllm.model_executor.models.qwen3 as qwen3_model
        from .vllm.model_executor.models.patch_qwen3 import (
            Qwen3ForCausalLMAsymmetric,
            Qwen3AttentionAsymmetric,
        )
        qwen3_model.Qwen3Attention = Qwen3AttentionAsymmetric
        qwen3_model.Qwen3ForCausalLM = Qwen3ForCausalLMAsymmetric

        # qwen3_moe
        import vllm.model_executor.models.qwen3_moe as qwen3_moe_model
        from .vllm.model_executor.models.patch_qwen3_moe import (
            Qwen3MoeForCausalLMAsymmtric,
            Qwen3MoeAttentionAsymmetric,
        )
        qwen3_moe_model.Qwen3MoeAttention = Qwen3MoeAttentionAsymmetric
        qwen3_moe_model.Qwen3MoeForCausalLM = Qwen3MoeForCausalLMAsymmtric

        # qwen3.5
        import vllm.model_executor.models.qwen3_5 as qwen3_5_model
        from .vllm.model_executor.models.patch_qwen3_5 import (
            Qwen3_5DecoderLayerAsymmetric,
            Qwen3_5ForCausalLMAsymmetric,
            Qwen3_5ForConditionalGenerationAsymmetric,
            Qwen3_5GatedDeltaNetAsymmetric,
            Qwen3_5ModelAsymmetric,
        )
        qwen3_5_model.Qwen3_5ForCausalLMBase = Qwen3_5ForCausalLMAsymmetric
        qwen3_5_model.Qwen3_5ForCausalLM = Qwen3_5ForCausalLMAsymmetric
        qwen3_5_model.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGenerationAsymmetric
        qwen3_5_model.Qwen3_5Model = Qwen3_5ModelAsymmetric
        qwen3_5_model.Qwen3_5DecoderLayer = Qwen3_5DecoderLayerAsymmetric
        qwen3_5_model.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNetAsymmetric

        # qwen3_vl
        import vllm.model_executor.models.qwen3_vl as qwen3_vl_model
        from .vllm.model_executor.models.patch_qwen3_vl import (
            Qwen3_VisionBlockAsymmetric,
            Qwen3_VisionMLPAsymmetric,
            Qwen3_VisionPatchMergerAsymmetric,
            Qwen3_VisionTransformerAsymmetric,
        )
        qwen3_vl_model.Qwen3_VisionMLP = Qwen3_VisionMLPAsymmetric
        qwen3_vl_model.Qwen3_VisionBlock = Qwen3_VisionBlockAsymmetric
        qwen3_vl_model.Qwen3_VisionPatchMerger = Qwen3_VisionPatchMergerAsymmetric
        qwen3_vl_model.Qwen3_VisionTransformer = Qwen3_VisionTransformerAsymmetric

        # qwen2_moe
        import vllm.model_executor.models.qwen2_moe as qwen2_moe_model
        from .vllm.model_executor.models.patch_qwen2_moe import Qwen2MoeMLPAsymmetric
        qwen2_moe_model.Qwen2MoeMLP = Qwen2MoeMLPAsymmetric

        # 变长 allgather 系列
        import vllm_ascend.ops.vocab_parallel_embedding
        from .vllm.model_executor.layers.patch_logits_processor import LogitsProcessorVarlen
        vllm_ascend.ops.vocab_parallel_embedding.AscendLogitsProcessor = LogitsProcessorVarlen

        from .vllm.distributed.communication_op import tensor_model_parallel_all_gather_varlen
        import vllm.distributed
        vllm.distributed.tensor_model_parallel_all_gather_varlen = tensor_model_parallel_all_gather_varlen

        import vllm_ascend.distributed.device_communicators.npu_communicator
        from .vllm.distributed.device_communicators.patch_base_device_communicator import all_gather_varlen
        vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator.all_gather_varlen = all_gather_varlen

        import vllm.config.model
        from .vllm.config.patch_model import verify_with_parallel_config

        vllm.config.model.ModelConfig.verify_with_parallel_config = verify_with_parallel_config
        logger.info("Replaced verify_with_parallel_config with patched version")

        from vllm.v1.core import kv_cache_utils
        from .vllm.v1.core.patch_kv_cache_utils import get_kv_cache_configs

        kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
        logger.info("Replaced vllm.v1.core.kv_cache_utils.get_kv_cache_configs with patched version")

    # Mamba layer asymmetric TP support (legacy vLLM only)
    if not _vllm_ge_023:
        try:
            import vllm.model_executor.layers.mamba.mamba_mixer2 as mamba_module
            from .vllm.model_executor.layers.mamba.patch_mamba_mixer2 import mamba_v2_sharded_weight_loader_asymmetric
            mamba_module.mamba_v2_sharded_weight_loader = mamba_v2_sharded_weight_loader_asymmetric
            logger.info("Patched mamba_v2_sharded_weight_loader for asymmetric TP support")
        except ImportError as e:
            logger.warning(f"Failed to patch Mamba layer: {e}")

    # ==================================================================
    # DeepSeek-V4 DP4TP4 -> DP4TP(3,4,4,4) heterogeneous-TP restart
    # All model-specific logic is implemented as runtime patches.
    # ==================================================================
    try:
        from .vllm.distributed.patch_hetero_utils import (
            apply_hetero_distributed_utils_patch,
        )
        apply_hetero_distributed_utils_patch()
        logger.info("Applied heterogeneous-TP distributed utils patch")
    except Exception as e:
        logger.warning(f"Failed to patch distributed utils for heterogeneous TP: {e}")

    try:
        from .vllm.model_executor.layers.patch_hetero_parameter import (
            apply_hetero_parameter_patch,
        )
        apply_hetero_parameter_patch()
        logger.info("Applied heterogeneous-TP parameter weight-loader patch")
    except Exception as e:
        logger.warning(f"Failed to patch parameter weight loaders for heterogeneous TP: {e}")

    try:
        from .vllm.model_executor.layers.patch_hetero_vocab import (
            apply_hetero_vocab_patch,
        )
        apply_hetero_vocab_patch()
        logger.info("Applied heterogeneous-TP vocab embedding padding patch")
    except Exception as e:
        logger.warning(f"Failed to patch vocab embedding for heterogeneous TP: {e}")

    try:
        from .vllm.model_executor.layers.fused_moe.runner.patch_hetero_moe_runner import (
            apply_hetero_moe_runner_patch,
        )
        apply_hetero_moe_runner_patch()
        logger.info("Applied heterogeneous-TP MoERunner shared-output padding patch")
    except Exception as e:
        logger.warning(f"Failed to patch MoERunner for heterogeneous TP: {e}")

    try:
        from .vllm.model_executor.model_loader.patch_hetero_default_loader import (
            apply_hetero_default_loader_patch,
        )
        apply_hetero_default_loader_patch()
        logger.info("Applied heterogeneous-TP default model loader patch")
    except Exception as e:
        logger.warning(f"Failed to patch default model loader for heterogeneous TP: {e}")

    try:
        from .vllm.config.patch_speculative_hetero import (
            apply_speculative_hetero_patch,
        )
        apply_speculative_hetero_patch()
        logger.info("Applied heterogeneous-TP speculative config patch")
    except Exception as e:
        logger.warning(f"Failed to patch speculative config for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.models.patch_deepseek_v4 import (
            apply_deepseek_v4_hetero_patch,
        )
        apply_deepseek_v4_hetero_patch()
        logger.info("Applied DeepSeek-V4 heterogeneous-TP model patch")
    except Exception as e:
        logger.warning(f"Failed to patch DeepSeek-V4 model for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.models.patch_deepseek_v4_mtp import (
            apply_deepseek_v4_mtp_hetero_patch,
        )
        apply_deepseek_v4_mtp_hetero_patch()
        logger.info("Applied DeepSeek-V4 MTP heterogeneous-TP patch")
    except Exception as e:
        logger.warning(f"Failed to patch DeepSeek-V4 MTP for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.patch.patch_hetero_tp import (
            apply_hetero_forward_context_patch,
        )
        apply_hetero_forward_context_patch()
        logger.info("Applied heterogeneous-TP ascend forward context patch")
    except Exception as e:
        logger.warning(f"Failed to patch ascend forward context for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.patch.patch_hetero_ascend_config import (
            apply_hetero_ascend_config_patch,
        )
        apply_hetero_ascend_config_patch()
        logger.info("Applied heterogeneous-TP ascend config/enable_sp patch")
    except Exception as e:
        logger.warning(f"Failed to patch ascend config for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.ops.patch_hetero_custom_ops import (
            apply_hetero_custom_ops_patch,
        )
        apply_hetero_custom_ops_patch()
        logger.info("Applied heterogeneous-TP MoE custom-op patch")
    except Exception as e:
        logger.warning(f"Failed to patch MoE custom ops for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.ops.fused_moe.patch_hetero_moe import (
            apply_hetero_moe_patch,
        )
        apply_hetero_moe_patch()
        logger.info("Applied heterogeneous-TP MoE prepare/finalize/dispatcher patch")
    except Exception as e:
        logger.warning(f"Failed to patch MoE ops for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.attention.patch_deepseek_v4_attention_hetero import (
            apply_deepseek_v4_attention_hetero_patch,
        )
        apply_deepseek_v4_attention_hetero_patch()
        logger.info("Applied DeepSeek-V4 DSA attention heterogeneous-TP patch")
    except Exception as e:
        logger.warning(f"Failed to patch DeepSeek-V4 attention for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.worker.patch_hetero_model_runner import (
            apply_hetero_model_runner_patch,
        )
        apply_hetero_model_runner_patch()
        logger.info("Applied heterogeneous-TP NPUModelRunner patch")
    except Exception as e:
        logger.warning(f"Failed to patch NPUModelRunner for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.spec_decode.patch_hetero_spec_decode import (
            apply_hetero_spec_decode_patch,
        )
        apply_hetero_spec_decode_patch()
        logger.info("Applied heterogeneous-TP MTP proposer patch")
    except Exception as e:
        logger.warning(f"Failed to patch spec-decode proposer for heterogeneous TP: {e}")

    try:
        from .vllm_ascend.distributed.kv_transfer.patch_hetero_mooncake import (
            apply_hetero_mooncake_patch,
        )
        apply_hetero_mooncake_patch()
        logger.info("Applied heterogeneous-TP Mooncake hybrid connector patch")
    except Exception as e:
        logger.warning(f"Failed to patch Mooncake connector for heterogeneous TP: {e}")

# 导出以兼容插件框架
class ZeroInterruptPluginPatch:
    """ZeroInterrupt 插件主 patch 类，用于兼容性。"""

    @classmethod
    def apply(cls):
        """应用 patch。"""
        apply()