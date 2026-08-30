import json
import os.path
from collections import defaultdict

import numpy as np
import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
from vllm_ascend.eplb.core.eplb_utils import (
    expert_file_to_tensor,
    generate_global_placement,
    generate_log2phy_map
)

def patched_init_eplb_config(eplb_config, layer_id, moe_config, mix_placement=False, num_shared_experts=1, tp_size=None):
    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    global_placement = None
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    num_shared_experts = num_shared_experts if mix_placement else 0

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, None, n_redundant, None

    if expert_map_path:
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        n_redundant = physical_count - n_experts
    elif not eplb_enable:
        local_num_experts, expert_map, _ = determine_expert_map(ep_size, moe_config.ep_rank, n_experts)
        return None, expert_map, None, 0, local_num_experts

    if global_placement is None:
        global_placement = generate_global_placement(n_experts, ep_size, n_redundant, num_shared_experts)
        if mix_placement:
            n_redundant += ep_size - 1
    global_expert_map = []
    for rankid in range(ep_size):
        expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
        global_expert_map.append(expert_map)
        if rankid == moe_config.ep_rank:
            local_expert_map = expert_map
    log2phy = (
        generate_log2phy_map(
            global_expert_map,
            moe_config.ep_rank,
            tp_size=int(tp_size) if tp_size is not None else None,
        ).npu()
        if eplb_enable
        else None
    )

    return torch.stack(global_expert_map), local_expert_map, log2phy, n_redundant, local_num_experts
