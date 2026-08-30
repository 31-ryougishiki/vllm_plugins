from typing import TYPE_CHECKING

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.vllm_ascend.utils import asymmetric_divide

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _ratio_partition_size(total, rank, size, ratios):
    """Partition size for ``rank`` under explicit TP sharding ratios."""
    if size == 1:
        return total
    total_ratio = sum(ratios)
    sizes = [total * ratio // total_ratio for ratio in ratios]
    sizes[-1] += total - sum(sizes)
    return sizes[rank]


def _ratio_partition_offset(total, rank, size, ratios):
    """Cumulative partition offset for ``rank`` under explicit ratios."""
    if size == 1:
        return 0
    total_ratio = sum(ratios)
    return sum(total * ratios[i] // total_ratio for i in range(rank))


class PatchAscendMultiConnector(AscendMultiConnector):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # [h30014172] 非对称 TP 信息
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        total_num_kv_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        uneven_split = (total_num_kv_heads % tp_size != 0)

        # DeepSeek-V4 heterogeneous TP uses explicit per-rank ratios such as
        # [2,1,1] on DP0.  ``asymmetric_divide`` (uniform + remainder to the
        # last rank) is NOT equivalent to the ratio split and would build the
        # wrong KV head/rank map, so prefer the parallel-config ratios when
        # they are available.
        ratios = None
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if parallel_config is not None and getattr(
            parallel_config, "is_heterogeneous_tp", False
        ):
            get_ratios = getattr(
                parallel_config, "get_sharding_ratios_for_dp", None
            )
            if get_ratios is not None:
                ratios = get_ratios(parallel_config.data_parallel_rank)
        if ratios is not None and len(ratios) == tp_size:
            self._tp_sharding_ratios = [int(r) for r in ratios]
            self.local_kv_heads = _ratio_partition_size(
                total_num_kv_heads, tp_rank, tp_size, self._tp_sharding_ratios
            )
            self.kv_head_offset = _ratio_partition_offset(
                total_num_kv_heads, tp_rank, tp_size, self._tp_sharding_ratios
            )
            self.uneven_split = any(
                ratio != self._tp_sharding_ratios[0]
                for ratio in self._tp_sharding_ratios
            )
        else:
            self._tp_sharding_ratios = None
            # register_kv_caches() reads this attribute; without it every
            # register_kv_caches call raises AttributeError even in the
            # evenly-divisible case.
            self.uneven_split = uneven_split
            if uneven_split:
                self.local_kv_heads, self.kv_head_offset = asymmetric_divide(
                    total_num_kv_heads, tp_size, tp_rank
                )
            else:
                self.local_kv_heads = total_num_kv_heads // tp_size
                self.kv_head_offset = tp_rank * self.local_kv_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                c.update_state_after_alloc(request, empty_blocks, 0)

    # ========== [h30014172] 新增：非对称 TP 信息注入 ==========
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        # 将非对称信息注入到子 connector
        if self.uneven_split:
            for connector in self._connectors:
                if hasattr(connector, 'connector_worker') and connector.connector_worker is not None:
                    worker = connector.connector_worker
                    worker.uneven_split = self.uneven_split
                    worker.local_kv_heads = self.local_kv_heads
                    worker.kv_head_offset = self.kv_head_offset
                    worker.total_num_kv_heads = self.total_num_kv_heads
                    worker.tp_size = self.tp_size
                    worker.tp_rank = self.tp_rank
                    worker.tp_sharding_ratios = self._tp_sharding_ratios

    # ========== [h30014172] 新增：KV 传输映射计算 ==========
    @staticmethod
    def compute_kv_transfer_mapping(
        total_num_kv_heads: int,
        tp_size: int,
        dst_rank: int,
        uneven_split: bool = False,
        tp_sharding_ratios: list[int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """计算非对称 TP 下 KV 传输的地址映射。

        Args:
            total_num_kv_heads: 全局 kv heads 总数
            tp_size: TP 并行度
            dst_rank: 目标 rank（D 侧）
            uneven_split: 是否非对称切分
            tp_sharding_ratios: 显式 per-rank 切分配比（如 [2,1,1]）

        Returns:
            List of (src_rank, src_local_offset, dst_local_offset, num_heads)
        """
        if not uneven_split and tp_sharding_ratios is None:
            dst_kv_heads = total_num_kv_heads // tp_size
            return [(dst_rank, 0, 0, dst_kv_heads)]

        if tp_sharding_ratios is not None and len(tp_sharding_ratios) == tp_size:
            def partition_size(rank):
                return _ratio_partition_size(
                    total_num_kv_heads, rank, tp_size, tp_sharding_ratios
                )

            def partition_offset(rank):
                return _ratio_partition_offset(
                    total_num_kv_heads, rank, tp_size, tp_sharding_ratios
                )
        else:
            def partition_size(rank):
                return asymmetric_divide(total_num_kv_heads, tp_size, rank)[0]

            def partition_offset(rank):
                return asymmetric_divide(total_num_kv_heads, tp_size, rank)[1]

        dst_kv_heads = partition_size(dst_rank)
        dst_global_offset = partition_offset(dst_rank)

        mapping = []
        remaining = dst_kv_heads
        dst_local_offset = 0

        for src_rank in range(tp_size):
            if remaining <= 0:
                break
            src_kv_heads = partition_size(src_rank)
            src_global_offset = partition_offset(src_rank)
            overlap_start = max(dst_global_offset, src_global_offset)
            overlap_end = min(
                dst_global_offset + dst_kv_heads,
                src_global_offset + src_kv_heads,
            )
            if overlap_start >= overlap_end:
                continue

            transfer_heads = overlap_end - overlap_start
            src_local_offset = overlap_start - src_global_offset
            mapping.append(
                (src_rank, src_local_offset, dst_local_offset, transfer_heads)
            )
            dst_local_offset += transfer_heads
            remaining -= transfer_heads

        assert remaining == 0, (
            f"Mapping incomplete: {remaining} heads unmapped for dst_rank={dst_rank}"
        )
        return mapping
