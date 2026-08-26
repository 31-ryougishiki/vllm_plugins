# Ascend NPU custom operators plugin
# Replaces SiLU activation with Ascend ACLNN custom operator

def register(manager):
    from .patch import MRopeSplitForQwen3VLPatch
    manager.register('split_rmsnorm_mrope_qwen3vl', MRopeSplitForQwen3VLPatch)
    from .patch import MRopeSplitForQwen2VLPatch
    manager.register('split_mrope_qwen25vl', MRopeSplitForQwen2VLPatch)
    
    import os
    if os.getenv("PATCH_DEEPSEEKV4", None): 
        from .patch import HcPreForDeepseekV2DecoderLayerPatch
        manager.register('hc_pre_dpskv4', HcPreForDeepseekV2DecoderLayerPatch)

	# NOTE: [lqf] we don't need RotaryEmbeddingPatch
    # because we replace rotary_embedding.py file directly.
    # from .patch import RotaryEmbeddingPatch
    from .patch import MinMaxPatchForward
    manager.register('minmax_qkv', MinMaxPatchForward)
    from .patch import SplitRmsRopeForQwen3NextPatch
    manager.register('split_rms_rope_qwen3next', SplitRmsRopeForQwen3NextPatch)
    # from .patch import Rmsnorm2RopeNormForGemma4Patch
    # manager.register('rmsnorm2_rope_norm', Rmsnorm2RopeNormForGemma4Patch)

