# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import init_logger

errorlogger = init_logger(__name__)


def tensor_model_parallel_all_gather_varlen(
    input_: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """All-gather(varlen) the input tensor across model parallel group.
    
    新增支持非对称的allgather算子
    """
    tp_rank = get_tensor_model_parallel_rank()
    #errorlogger.debug(f"[lqf] communication_op:tp_rank:{tp_rank},input.shape={input_.shape}, dim={dim}")
    return get_tp_group().all_gather_varlen(input_, dim)
