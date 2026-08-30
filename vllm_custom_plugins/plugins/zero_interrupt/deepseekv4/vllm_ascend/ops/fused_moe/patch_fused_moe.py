import torch
import torch.nn.functional as F
import torch_npu

from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method

try:
    # vLLM Ascend >= 0.23.0
    from vllm_ascend.ops.fused_moe.fused_moe import get_compressed_expert_map
except ImportError:
    # vLLM Ascend 0.18.x
    from vllm.model_executor.layers.fused_moe.layer import (
        get_compressed_expert_map,
    )
from vllm_ascend.ops.fused_moe.fused_moe import (AscendFusedMoE, AscendUnquantizedFusedMoEMethod)

from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.eplb.core.patch_eplb_utils import patched_init_eplb_config

class PatchAscendFusedMoE(AscendFusedMoE):
    def __init__(self, *args, **kwargs):
        """
            主要修改 local_num_experts 计算逻辑，适配非对称场景
        """
        FusedMoE.__init__(self, *args, **kwargs) # 绕开 AscendFusedMoE 的 __init__()
        # super().__init__(*args, **kwargs) # TODO: do not use AscendFusedMoE's __init__
        logger.info_once(
                "[lqf] call PatchAscendFusedMoE.__init__"
        )

        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self._expert_map = None
        self.log2phy = None

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()
        # flashcommon3 gate stream
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
            AscendFusedMoE.gate_stream = torch.npu.Stream()
        if self.custom_routing_function is None and self.e_score_correction_bias is not None:
            vllm_config = get_current_vllm_config()
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )

        # init moe
        eplb_config = ascend_config.eplb_config
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num, local_num_experts = patched_init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config
        )
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        # patch_eplb_utils.py patched_init_eplb_config returns local_num_experts=None when:
        # - ep_size == 1 (no expert parallelism), OR
        # - EPLB is disabled and expert_map_path is not configured
        # In these cases, fall back to using global_num_experts as the local count.
        if local_num_experts is None:
            local_num_experts = self.global_num_experts
        self.local_num_experts = local_num_experts
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        if self.dynamic_eplb:
            self.multi_stage = False
            self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()
            if eplb_config.eplb_policy_type == 3:
                self.multi_stage = True
                self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
                self.num_iter = eplb_config.expert_heat_collection_interval
                self.moe_load = torch.zeros((self.num_iter, self.local_num_experts), dtype=torch.int32, device="npu")

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

        # Bypass AscendFusedMoE.__init__ and call FusedMoE.__init__ directly for custom expert distribution.
        # FusedMoE.__init__ normally sets intermediate_size_per_partition at line 487 (intermediate_size // self.tp_size),
        # but this class attribute may not be accessible yet when create_weights is called in some code paths
        # (e.g., when SharedFusedMoE or other subclasses trigger weight creation before the attribute propagates).
        # Manually compute and inject intermediate_size_per_partition to ensure it's available.
        # DeepSeek-V4 异构重启：按 tp_asymmetric_shardings（例如 [2,1,1]）切分
        # 而非 intermediate_size // tp_size，否则 2048 在 tp=3 上会切成
        # 682/682/682 而不是 1024/512/512。
        _asym_ratios = None
        try:
            from vllm.config import get_current_vllm_config_or_none
            from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm.v1.executor.utils import (
                get_tp_asymmetric_shardings,
            )

            _cfg = get_current_vllm_config_or_none()
            _additional = getattr(_cfg, "additional_config", None) or {}
            _ratios = get_tp_asymmetric_shardings(
                _additional.get("zero_interrupt_config", {})
            )
            if _ratios and len(_ratios) == self.tp_size:
                _asym_ratios = [int(r) for r in _ratios]
        except Exception:
            _asym_ratios = None

        if _asym_ratios is not None:
            _world_split = sum(_asym_ratios)
            _split = _asym_ratios[self.tp_rank]
            intermediate_size_per_partition = (
                intermediate_size // _world_split
            ) * _split
            _remainder = intermediate_size % _world_split
            # remainder 给最后一个 rank，保持各 rank 分片之和等于
            # intermediate_size。
            if self.tp_rank == self.tp_size - 1:
                intermediate_size_per_partition += _remainder
        else:
            intermediate_size_per_partition = intermediate_size // self.tp_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.moe_config.intermediate_size_per_partition = intermediate_size_per_partition
        logger.info_once(f"TP={self.tp_size}, local_num_experts={self.local_num_experts}, "
                         f"hidden_size={self.hidden_size}, intermediate_size={intermediate_size}, "
                         f"intermediate_size_per_partition={intermediate_size_per_partition}")
        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod"):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel

        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()
        self.runner = self._init_runner()
