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
from vllm_ascend.ops.fused_moe.fused_moe import get_compressed_expert_map
from vllm_ascend.ops.fused_moe.fused_moe import (AscendFusedMoE, AscendUnquantizedFusedMoEMethod)

from vllm_custom_plugins.plugins.zero_interrupt.vllm_ascend.eplb.core.patch_eplb_utils import patched_init_eplb_config

class PatchAscendFusedMoE(AscendFusedMoE):

    @property
    def local_num_experts(self):
        # [0.23.0] AscendFusedMoE.__init__ sets local_num_experts = global //
        # ep_size (floor), which drops the remainder experts when the count is
        # not divisible by ep_size (e.g. DP=4->3: 128 experts / 3 = 42.67 -> 42,
        # dropping experts 42 and 85 -> IndexError at weight load). The
        # expert_map (from determine_expert_map) already distributes the
        # remainder to the leading ranks (43/43/42, covering all 128), so
        # derive local_num_experts from the expert_map count instead of the
        # floor. For divisible cases (128/4=32) the count equals the floor,
        # so behavior is unchanged.
        em = getattr(self, "_expert_map", None)
        if em is not None:
            try:
                return int((em >= 0).sum().item())
            except Exception:
                pass
        return self.__dict__.get("_local_num_experts_fallback", 0)

    @local_num_experts.setter
    def local_num_experts(self, v):
        self.__dict__["_local_num_experts_fallback"] = v

    def __init__(self, *args, **kwargs):
        """
            主要修改 local_num_experts 计算逻辑，适配非对称场景
        """
        # [0.23.0] For the standard (symmetric) path — no zero_interrupt config,
        # i.e. normal DP=4 startup and any non-degrade path — delegate to the
        # REAL 0.23.0 AscendFusedMoE.__init__. The 0.18.0-derived manual __init__
        # below is incompatible with 0.23.0's AscendFusedMoE API
        # (intermediate_size_per_partition is now a read-only @property;
        # _init_runner no longer exists; etc.). Only the asymmetric
        # degrade/recover path needs the patched EPLB expert distribution, so
        # gate the custom logic behind zero_interrupt_config.
        vllm_config = get_current_vllm_config()
        additional_config = getattr(vllm_config, "additional_config", None)
        zero_interrupt_config = (additional_config.get(
            "zero_interrupt_config", None) if additional_config else None)
        if not zero_interrupt_config:
            AscendFusedMoE.__init__(self, *args, **kwargs)
            return

        # [0.23.0] Also delegate to the real AscendFusedMoE.__init__ on the
        # asymmetric (degrade/recover) path. The 0.18.0-derived custom __init__
        # below is incompatible with 0.23.0's AscendFusedMoE API (_init_runner
        # removed, intermediate_size_per_partition is a read-only @property,
        # init_eplb_config arity/signature changed, expert_map_manager /
        # base_quant_method / swiglu_limit / mix_placement added). The real
        # 0.23.0 init_eplb_config + redundant-expert (EPLB) mechanism handles
        # expert distribution, including non-divisible counts via redundant
        # experts, so the custom patched_init_eplb_config path is not needed.
        AscendFusedMoE.__init__(self, *args, **kwargs)
        return

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
        intermediate_size_per_partition = intermediate_size // self.tp_size
        # [0.23.0] intermediate_size_per_partition is now a read-only @property
        # on FusedMoE (vllm/model_executor/layers/fused_moe/layer.py) that returns
        # self.moe_config.intermediate_size_per_partition. Assigning it on the
        # instance raises "property ... has no setter". Set it on moe_config
        # instead (which the @property reads); FusedMoE.__init__ above already
        # does this, so the line below is redundant but kept explicit.
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
