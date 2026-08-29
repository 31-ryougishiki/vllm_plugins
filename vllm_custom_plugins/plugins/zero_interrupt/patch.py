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

    # 模型计算相关patch
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
    

    #patch 适配qwen3-30B-A3B场景下非对称，并行相关的特性如sp等暂时不支持
    
    # qwen2
    import vllm.model_executor.models.qwen2 as qwen2_model
    from .vllm.model_executor.models.patch_qwen2 import qwen2_mlp_asymmetric_init
    qwen2_model.Qwen2MLP.__init__  = qwen2_mlp_asymmetric_init

    # qwen3
    import vllm.model_executor.models.qwen3 as qwen3_model
    from .vllm.model_executor.models.patch_qwen3 import (
        Qwen3ForCausalLMAsymmetric,
        Qwen3AttentionAsymmetric
    )
    qwen3_model.Qwen3Attention = Qwen3AttentionAsymmetric
    qwen3_model.Qwen3ForCausalLM = Qwen3ForCausalLMAsymmetric

    # TP非对称权重加载和头数分配
    # qwen3_moe
    import vllm.model_executor.models.qwen3_moe as qwen3_moe_model
    from .vllm.model_executor.models.patch_qwen3_moe import (
        Qwen3MoeForCausalLMAsymmtric,
        Qwen3MoeAttentionAsymmetric
    )
    qwen3_moe_model.Qwen3MoeAttention = Qwen3MoeAttentionAsymmetric
    logger.info("Replaced qwen3_moe.Qwen3MoeAttention with Qwen3MoeAttentionAsymmetric")
    qwen3_moe_model.Qwen3MoeForCausalLM = Qwen3MoeForCausalLMAsymmtric
    logger.info("Replaced qwen3_moe.Qwen3MoeForCausalLM with Qwen3MoeForCausalLMAsymmtric")

    # qwen3.5 dense model asymmetric TP support
    import vllm.model_executor.models.qwen3_5 as qwen3_5_model
    from .vllm.model_executor.models.patch_qwen3_5 import (
        Qwen3_5DecoderLayerAsymmetric,
        Qwen3_5ForCausalLMAsymmetric,
        Qwen3_5ForConditionalGenerationAsymmetric,
        Qwen3_5ModelAsymmetric,
    )
    # Patch all Qwen3.5 classes for asymmetric TP
    qwen3_5_model.Qwen3_5ForCausalLMBase = Qwen3_5ForCausalLMAsymmetric
    qwen3_5_model.Qwen3_5ForCausalLM = Qwen3_5ForCausalLMAsymmetric
    qwen3_5_model.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGenerationAsymmetric
    qwen3_5_model.Qwen3_5Model = Qwen3_5ModelAsymmetric
    qwen3_5_model.Qwen3_5DecoderLayer = Qwen3_5DecoderLayerAsymmetric
    # Qwen3_5GatedDeltaNet is not patched directly - v0.23.0 uses QwenGatedDeltaNetAttention
    # which is instantiated by Qwen3_5DecoderLayerAsymmetric internally
    logger.info("Replaced qwen3_5 classes with asymmetric variants: ForCausalLM, ForConditionalGeneration, Model, DecoderLayer")

    # Qwen3-VL multimodal vision module asymmetric TP support
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
    logger.info("Replaced qwen3_vl vision classes with asymmetric variants: VisionMLP, VisionBlock, VisionPatchMerger, VisionTransformer")

    # Qwen2_moe multimodal vision module asymmetric TP support qwen3.5
    import vllm.model_executor.models.qwen2_moe as qwen2_moe_model
    from .vllm.model_executor.models.patch_qwen2_moe import Qwen2MoeMLPAsymmetric
    qwen2_moe_model.Qwen2MoeMLP=Qwen2MoeMLPAsymmetric
    logger.info("Replaced Qwen2MoeMLP with Qwen2MoeMLPAsymmetric")
    #
    #
    # 变长allgather 1
    import vllm_ascend.ops.vocab_parallel_embedding
    from .vllm.model_executor.layers.patch_logits_processor import LogitsProcessorVarlen
    vllm_ascend.ops.vocab_parallel_embedding.AscendLogitsProcessor = LogitsProcessorVarlen
    logger.info("Replaced vllm_ascend.ops.vocab_parallel_embedding.AscendLogitsProcessor with LogitsProcessorVarlen")

    # 变长allgather 2
    from .vllm.distributed.communication_op import tensor_model_parallel_all_gather_varlen
    import vllm.distributed
    vllm.distributed.tensor_model_parallel_all_gather_varlen = tensor_model_parallel_all_gather_varlen
    logger.info("Replaced vllm.distributed.tensor_model_parallel_all_gather_varlen with tensor_model_parallel_all_gather_varlen")

    # 变长allgather 3
    import vllm_ascend.distributed.device_communicators.npu_communicator
    from .vllm.distributed.device_communicators.patch_base_device_communicator import all_gather_varlen
    vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator.all_gather_varlen = all_gather_varlen
    logger.info("Add vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator.all_gather_varlen")

    import vllm.config.model
    from .vllm.config.patch_model import verify_with_parallel_config
    vllm.config.model.ModelConfig.verify_with_parallel_config = verify_with_parallel_config
    logger.info("Replaced verify_with_parallel_config with patched version")
    #
    # TP非对称权重加载和头数分配
    # 此方法被EngineCore使用, 需要提前patch
    from vllm.v1.core import kv_cache_utils
    from .vllm.v1.core.patch_kv_cache_utils import get_kv_cache_configs

    kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
    logger.info("Replaced vllm.v1.core.kv_cache_utils.get_kv_cache_configs with patched version")
    # # end
    #
    # # Mamba layer asymmetric TP support
    try:
        import vllm.model_executor.layers.mamba.mamba_mixer2 as mamba_module
        from .vllm.model_executor.layers.mamba.patch_mamba_mixer2 import mamba_v2_sharded_weight_loader_asymmetric
        mamba_module.mamba_v2_sharded_weight_loader = mamba_v2_sharded_weight_loader_asymmetric
        logger.info("Patched mamba_v2_sharded_weight_loader for asymmetric TP support")
    except ImportError as e:
        logger.warning(f"Failed to patch Mamba layer: {e}")

# 导出以兼容插件框架
class ZeroInterruptPluginPatch:
    """ZeroInterrupt 插件主 patch 类，用于兼容性。"""

    @classmethod
    def apply(cls):
        """应用 patch。"""
        apply()