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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import logging

import torch
from vllm.distributed.parallel_state import GroupCoordinator, get_tensor_model_parallel_rank

errorlogger = logging.getLogger(__name__)


class PatchGroupCoordinatorPatch(GroupCoordinator):

    def all_gather_varlen(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """支持dynamic长度的all_gather"""
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        dim = dim % input_.dim()
        self.use_custom_op_call = False
        if self.use_custom_op_call:
            return torch.ops.vllm.all_gather(
                input_,
                dim,
                world_size,
                group_name=self.unique_name
            )
        else:
            return self._all_gather_varlen(input_, dim)

    def _all_gather_varlen(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        return self.device_communicator.all_gather_varlen(input_, dim)
