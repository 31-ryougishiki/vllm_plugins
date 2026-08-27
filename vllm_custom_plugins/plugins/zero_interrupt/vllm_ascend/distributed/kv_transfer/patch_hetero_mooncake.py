# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous-TP PD-separation monkey-patch for Mooncake hybrid KV transfer.

This module belongs to the ``zero_interrupt`` plugin and is intentionally a
standalone patch file: it does **not** modify anything under ``hetero_cp`` or
``origin``.  It monkey-patches the *installed*
``vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector`` module
(and, optionally, the plain ``mooncake_connector`` sibling) so that a
heterogeneous TP PD-separation deployment works:

* prefill pool: DP4 TP(3,4,4,4), global ranks 0..14
* decode pool:   DP16 TP1

All function bodies below are copied/adapted from the final reference
implementations under ``hetero_cp``.  The functions are deliberately defined
with their target-method-compatible signatures because
``apply_hetero_mooncake_patch()`` copies only ``__code__`` objects, never the
whole method attribute.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        MooncakeConnectorMetadata,
    )

logger = logging.getLogger("vllm_custom_plugins")


def get_dp_side_channel_port_offset(parallel_config: ParallelConfig) -> int:
    """Return the DP-rank offset added to ``kv_port`` for side channels.

    Under heterogeneous TP the DP ranks use the cumulative device offset
    (0/3/7/11 for tp=[3,4,4,4]), not the uniform ``dp_rank * tp_size``
    stride.  Both the scheduler and the workers must agree on this value,
    otherwise the producer handshake ports advertised in
    ``request_finished_all_groups`` point at a different DP rank than the
    ports the workers actually bind.
    """
    if parallel_config.is_heterogeneous_tp:
        return parallel_config.get_rank_offset_for_dp(
            parallel_config.data_parallel_rank
        )
    return (
        parallel_config.data_parallel_rank
        * parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
    )


def _patched_scheduler_init(
    self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
):
    self.vllm_config = vllm_config
    self.kv_cache_config = kv_cache_config
    init_ascend_config(vllm_config)
    self.ascend_config = get_ascend_config()
    self.block_size = vllm_config.cache_config.block_size
    self.engine_id = engine_id
    self.local_ip = get_ip()
    logger.info("Initializing Mooncake Scheduler %s", engine_id)

    self.side_channel_host = get_ip()
    self.tp_size = vllm_config.parallel_config.tensor_parallel_size
    self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    assert self.pcp_size * self.dcp_size == 1, "Mooncake Hybrid Connector only support cp_world_size == 1. "
    parallel_config = vllm_config.parallel_config
    if parallel_config.is_heterogeneous_tp:
        # ``world_size_across_dp`` sums the per-DP tp sizes (15 for
        # tp=[3,4,4,4]) and is kept for diagnostics/limit checks.
        self.max_device_id = parallel_config.world_size_across_dp
    else:
        self.max_device_id = (
            parallel_config.tensor_parallel_size
            * parallel_config.data_parallel_size
            * parallel_config.pipeline_parallel_size
        )

    # Handshake base port.  Must match the workers' calculation exactly.
    dp_port_offset = get_dp_side_channel_port_offset(parallel_config)
    self.side_channel_port = (
        vllm_config.kv_transfer_config.kv_port + dp_port_offset
    )
    logger.debug(
        "MooncakeHybridConnector scheduler dp_rank=%d kv_port=%d "
        "dp_port_offset=%d side_channel_port=%d.",
        parallel_config.data_parallel_rank,
        vllm_config.kv_transfer_config.kv_port,
        dp_port_offset,
        self.side_channel_port,
    )
    # Requests that need to start recv.
    # New requests are added by update_state_after_alloc in
    # the scheduler. Used to make metadata passed to Worker.
    self._reqs_need_recv: dict[str, tuple[Request, BlockIds, int]] = {}
    self._reqs_need_send: dict[str, float] = {}
    self._reqs_in_batch: set[str] = set()

    # master-slave meta information for cross-nodes
    self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}

    # hybrid model config
    self.use_hybrid = (
        not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups)
        and len(kv_cache_config.kv_cache_groups) > 1
    )
    self.use_compress = hasattr(self.vllm_config.model_config.hf_config, "compress_ratios")

    self.kv_cache_specs = []
    self.need_truncate = self.use_compress
    sw_sizes_tokens: list[tuple[int, int]] = []
    self.group_block_size = []
    self.group_compress_ratio = [1 for _ in range(len(kv_cache_config.kv_cache_groups))]
    for i, g in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
            group_spec_set = []
            for layer_name in g.layer_names:
                layer_spec = g.kv_cache_spec.kv_cache_specs[layer_name]
                if layer_spec not in group_spec_set:
                    group_spec_set.append(layer_spec)
            self.kv_cache_specs.append(group_spec_set)
            self.group_block_size.append(g.kv_cache_spec.block_size)
            if isinstance(group_spec_set[0], SlidingWindowSpec):
                sw_sizes_tokens.append((group_spec_set[0].sliding_window, group_spec_set[0].block_size))
            else:
                sw_sizes_tokens.append((0, layer_spec.block_size))
                if self.use_compress and hasattr(group_spec_set[0], "compress_ratio"):
                    self.group_compress_ratio[i] = group_spec_set[0].compress_ratio
            if isinstance(layer_spec, MambaSpec):
                self.need_truncate = True
        else:
            self.group_block_size.append(g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec):
                sw_sizes_tokens.append((g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size))
            else:
                sw_sizes_tokens.append((0, g.kv_cache_spec.block_size))
                if self.use_compress and hasattr(g.kv_cache_spec, "compress_ratio"):
                    self.group_compress_ratio[i] = g.kv_cache_spec.compress_ratio
            if isinstance(g.kv_cache_spec, MambaSpec):
                self.need_truncate = True
            self.kv_cache_specs.append([g.kv_cache_spec])

    self.num_swa_blocks = [
        cdiv(n_tokens, block_size) + 1 if n_tokens else 0 for n_tokens, block_size in sw_sizes_tokens
    ]


def _patched_set_xfer_handshake_metadata(
    self, metadata: dict[int, KVConnectorHandshakeMetadata]
) -> None:
    """
    Set the KV connector handshake metadata for this connector.

    The mapping is keyed by the worker's global port offset
    (``handshake_port - kv_port``).  Under heterogeneous TP that offset
    is cumulative (0/3/7/11 for tp=[3,4,4,4]) and can no longer be
    reconstructed from the local TP rank alone.

    Args:
        metadata (dict): the handshake metadata to set.
    """
    if not metadata:
        return

    kv_port = self.vllm_config.kv_transfer_config.kv_port
    updated_mapping: dict[str, dict[str, Any]] = {}
    for local_rank, rank_metadata in metadata.items():
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            port_offset = handshake_port - kv_port
        else:
            # Backward compatibility with older producers that did not
            # publish handshake_port in MooncakeAgentMetadata.
            port_offset = int(local_rank)
        updated_mapping[str(port_offset)] = {
            "host": rank_metadata.local_ip,
            "engine_id": rank_metadata.engine_id,
            "handshake_port": kv_port + port_offset,
        }

    self.multi_nodes_meta_mapping.update(updated_mapping)
    logger.info(
        "MooncakeHybridConnector set_xfer_handshake_metadata: "
        "worker_count=%d, updated=%s, multi_nodes_meta_mapping=%s.",
        len(metadata),
        updated_mapping,
        self.multi_nodes_meta_mapping,
    )


def _patched_worker_init(
    self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
):
    self._get_prefill_decode_size(vllm_config)
    os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
    # ADXL link establishment defaults to only 10s.  Heterogeneous P/D
    # pools start many TransferEngines concurrently (15 producer + 16
    # consumer workers), so the first cross-node HcclCommPrepare can
    # exceed the default and surface as E19999/HCCL_E_TIMEOUT (ret 0x9)
    # even though the peer is healthy.  Keep an explicit user-provided
    # value, otherwise align the connect timeout with the transfer
    # timeout.
    os.environ.setdefault(
        "ASCEND_CONNECT_TIMEOUT", str(get_transfer_timeout_value())
    )
    # Mooncake's Python wrapper caps every sync batch at
    # MC_TRANSFER_TIMEOUT seconds (default 30s), independent of the
    # ASCEND_* timeouts.  The first cross-node ADXL connection can take
    # longer than that (see ASCEND_CONNECT_TIMEOUT above), so keep the
    # wrapper deadline above the link-establishment budget when the
    # deployment has not set MC_TRANSFER_TIMEOUT itself.
    connect_timeout_ms = int(os.environ["ASCEND_CONNECT_TIMEOUT"])
    os.environ.setdefault(
        "MC_TRANSFER_TIMEOUT",
        str(max(60, connect_timeout_ms // 1000 + 60)),
    )
    if self._prefill_tp_size < self._decode_tp_size:
        raise ValueError(
            f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
            f" or equal to the decode_tp_size: {self._decode_tp_size}"
        )

    # Metadata.
    self.vllm_config = vllm_config
    self.ascend_config = get_ascend_config()
    self.engine_id = engine_id
    self.tp_rank = get_tensor_model_parallel_rank()
    self.tp_size = vllm_config.parallel_config.tensor_parallel_size
    self.tp_group = get_tp_group()
    self.pp_rank = get_pp_group().rank_in_group
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
    self.dp_size = vllm_config.parallel_config.data_parallel_size_local
    self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
    self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    assert self.pcp_size * self.dcp_size == 1, "Mooncake Hybrid Connector only support cp_world_size == 1. "
    self.kv_caches: dict[str, torch.Tensor] = {}
    self.side_channel_host = get_ip()

    parallel_config = vllm_config.parallel_config
    if parallel_config.is_heterogeneous_tp:
        self.max_device_id = parallel_config.world_size_across_dp
    else:
        self.max_device_id = self.tp_size * self.dp_size * self.pp_size
    self.kv_role = vllm_config.kv_transfer_config.kv_role
    self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

    # kv cache config
    self.kv_cache_config = kv_cache_config
    self.use_hybrid = (
        not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups)
        and len(kv_cache_config.kv_cache_groups) > 1
    )
    self.hma_group_size = len(kv_cache_config.kv_cache_groups)

    # Mamba metadata
    self._is_mamba_group = [isinstance(group.kv_cache_spec, MambaSpec) for group in kv_cache_config.kv_cache_groups]
    mamba_ssm_size = (0, 0)
    self.use_mamba = any(self._is_mamba_group)
    if self.use_mamba:
        assert self.use_hybrid
        assert self._prefill_tp_size == self._decode_tp_size, (
            "Mooncake connector does not support different TP size with Mamba."
        )
        self.layer_specs = {
            layer: group.kv_cache_spec for group in kv_cache_config.kv_cache_groups for layer in group.layer_names
        }
        mamba_spec = next(spec for spec in self.layer_specs.values() if isinstance(spec, MambaSpec))
        conv_nbytes, ssm_nbytes = (
            torch.tensor([], dtype=mamba_spec.dtypes[0]).element_size(),  # type: ignore[misc]
            torch.tensor([], dtype=mamba_spec.dtypes[1]).element_size(),  # type: ignore[misc]
        )
        conv_shape, ssm_shape = (
            torch.Size(mamba_spec.shapes[0]),
            torch.Size(mamba_spec.shapes[1]),
        )
        mamba_ssm_size = (
            conv_shape.numel() * conv_nbytes,
            ssm_shape.numel() * ssm_nbytes,
        )
    self._mamba_ssm_size = mamba_ssm_size
    self.use_compress = hasattr(self.vllm_config.model_config.hf_config, "compress_ratios")

    # Handshake base port.  Must match MooncakeConnectorScheduler
    # exactly; under heterogeneous TP that is the cumulative device
    # offset (0/3/7/11 for tp=[3,4,4,4]).
    dp_port_offset = get_dp_side_channel_port_offset(parallel_config)
    self.side_channel_port = (
        vllm_config.kv_transfer_config.kv_port + dp_port_offset
    )
    device_index = self.pp_rank * self.tp_size + self.tp_rank
    self.handshake_port = self.side_channel_port + device_index
    logger.debug(
        "MooncakeHybridConnector worker dp_rank=%d tp_rank=%d "
        "dp_port_offset=%d side_channel_port=%d handshake_port=%d.",
        parallel_config.data_parallel_rank,
        self.tp_rank,
        dp_port_offset,
        self.side_channel_port,
        self.handshake_port,
    )
    self.sockets: dict = {}
    self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
    self.te_rpc_port = self.engine.get_rpc_port()

    # Background thread for sending or receiving KV caches.
    self.kv_send_thread: KVCacheSendingThread | None = None
    self.kv_recv_thread: KVCacheRecvingThread | None = None

    # Handshake metadata of this worker
    self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None

    # kv_transfer variables
    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size
    if self.vllm_config.model_config.is_deepseek_mla:
        self.tp_num_need_pulls = 1
    else:
        num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
        # On a kv_producer the ranks are selected from THIS instance's
        # per-DP tp group (tp_size can differ per DP rank under
        # heterogeneous TP).  ``_prefill_tp_size`` is only the remote
        # pool descriptor and is only used by consumers.
        producer_prefill_tp_size = (
            self.tp_size
            if self.kv_role == "kv_producer"
            else self._prefill_tp_size
        )
        num_p_block_heads = max(
            1, self.num_key_value_heads // producer_prefill_tp_size
        )
        self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
    self.local_remote_block_port_mapping: dict[str, list[list[int]] | None] = {}
    self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}


def _patched_get_tp_num_need_pulls(self, prefill_tp_size=None):
    if self.use_mamba:
        assert prefill_tp_size == self.tp_size, "Mooncake connector does not support different TP size with Mamba."
        return prefill_tp_size
    if prefill_tp_size is None:
        prefill_tp_size = self._prefill_tp_size

    # On a heterogeneous kv_producer the rank set is built from this
    # instance's per-DP tp_size (3 for DP0), while ``_prefill_tp_size``
    # is the remote pool descriptor from the extra config (4).
    parallel_config = self.vllm_config.parallel_config
    expected_prefill_tp_size = self._prefill_tp_size
    if self.kv_role == "kv_producer" and parallel_config.is_heterogeneous_tp:
        expected_prefill_tp_size = self.tp_size
    if prefill_tp_size == expected_prefill_tp_size:
        return self.tp_num_need_pulls

    if self.vllm_config.model_config.is_deepseek_mla:
        tp_num_need_pulls = 1
    else:
        num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
        num_p_block_heads = max(1, self.num_key_value_heads // prefill_tp_size)
        tp_num_need_pulls = num_d_block_heads // num_p_block_heads
    return tp_num_need_pulls


def _patched_get_remote_host_info_by_port(
    self,
    base_port: int,
    remote_handshake_port: int,
    remote_host: str,
    remote_engine_id: str,
    remote_multi_nodes_meta_mapping: dict,
):
    if remote_multi_nodes_meta_mapping is None:
        return remote_host, remote_engine_id

    # Producers publish the absolute handshake_port in their metadata.
    # Match on it first: producer and consumer kv_port values are allowed
    # to differ (e.g. prefill 36000 vs decode 36200), and under
    # heterogeneous TP only the absolute port identifies a worker
    # unambiguously across DP ranks.
    for info in remote_multi_nodes_meta_mapping.values():
        if (
            isinstance(info, dict)
            and info.get("handshake_port") == remote_handshake_port
        ):
            return (
                info.get("host", remote_host),
                info.get("engine_id", remote_engine_id),
            )

    # Legacy mappings are keyed by the producer's kv_port offset or by
    # the DP-local rank.
    kv_port = self.vllm_config.kv_transfer_config.kv_port
    rank = str(remote_handshake_port - kv_port)
    info = remote_multi_nodes_meta_mapping.get(rank)
    if info is None:
        rank = str(remote_handshake_port - base_port)
        info = remote_multi_nodes_meta_mapping.get(rank)
    if info is None:
        return remote_host, remote_engine_id
    return (
        info.get("host", remote_host),
        info.get("engine_id", remote_engine_id),
    )


def _patched_prefill_get_remote_rank(self, req_id: str) -> list[int]:
    # This method is only used on the producer side (see start_load_kv).
    # Select from the local per-DP TP group: under heterogeneous TP the
    # extra-config prefill tp_size may be larger than the local tp_size
    # (4 vs 3 for DP0), and hashing the wrong population would pick a
    # nonexistent local rank or disagree with the consumer's
    # remote_ptp_size-based selection.
    prefill_tp_size = self.tp_size
    return sum(self._get_remote_ranks_for_req(req_id, prefill_tp_size), [])


def _patched_start_load_kv(self, metadata: MooncakeConnectorMetadata):
    """Start loading KV blocks from remote engine."""
    for req_id, meta in metadata.requests.items():
        logger.debug(
            "start_load_kv for request %s from remote engine %s. "
            "Num local_block_ids: %s. Num remote_block_ids: %s. ",
            req_id,
            meta.remote_engine_id,
            len(meta.local_block_ids),
            len(meta.remote_block_ids),
        )

        prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        remote_req_id = meta.remote_request_id
        logger.debug(
            "start_load_kv request %s: remote_ptp_size=%s "
            "tp_num_need_pulls=%s remote_port=%s remote_host=%s.",
            remote_req_id,
            prefill_tp_size,
            tp_num_need_pulls,
            meta.remote_port,
            meta.remote_host,
        )

        def _validate_chosen_ranks(chosen_rank_list: list[int]) -> None:
            if any(
                rank < 0 or rank >= prefill_tp_size
                for rank in chosen_rank_list
            ):
                raise RuntimeError(
                    f"MooncakeHybridConnector selected invalid prefill TP "
                    f"rank {chosen_rank_list} for prefill_tp_size="
                    f"{prefill_tp_size}, request={remote_req_id}."
                )
            expected_pulls = (
                1
                if self.use_mamba
                else tp_num_need_pulls * self._prefill_pp_size
            )
            if len(chosen_rank_list) < expected_pulls:
                raise RuntimeError(
                    "MooncakeHybridConnector selected "
                    f"{len(chosen_rank_list)} prefill TP ranks but "
                    f"expected {expected_pulls} pulls for prefill_tp_size="
                    f"{prefill_tp_size}, decode_tp_size={self.tp_size}, "
                    f"request={remote_req_id}."
                )

        if self.use_mamba:
            assert self.kv_recv_thread is not None
            chosen_rank_list = self._get_remote_rank(remote_req_id, prefill_tp_size)
            _validate_chosen_ranks(chosen_rank_list)
            remote_handshake_port_list = [[x + meta.remote_port] for x in chosen_rank_list]
            remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                meta.remote_port,
                remote_handshake_port_list[0][0],
                meta.remote_host,
                meta.remote_engine_id,
                meta.remote_multi_nodes_meta_mapping,
            )
            self.kv_recv_thread.add_request(
                request_id=req_id,
                remote_request_id=remote_req_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_engine_id=remote_engine_id,
                remote_host=remote_host,
                remote_handshake_port=remote_handshake_port_list[0][0],
                offset=0,
                tp_num_need_pulls=tp_num_need_pulls,
                all_task_done=True,
            )
        else:  # TODO: support prefill context parallel and pipeline parallel open at the same time
            chosen_rank_list = self._get_remote_rank(remote_req_id, prefill_tp_size)
            _validate_chosen_ranks(chosen_rank_list)
            remote_handshake_port_list = [[x + meta.remote_port] for x in chosen_rank_list]
            for i in range(tp_num_need_pulls * self._prefill_pp_size):
                assert self.kv_recv_thread is not None
                remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                    meta.remote_port,
                    remote_handshake_port_list[i][0],
                    meta.remote_host,
                    meta.remote_engine_id,
                    meta.remote_multi_nodes_meta_mapping,
                )
                self.kv_recv_thread.add_request(
                    request_id=req_id,
                    remote_request_id=remote_req_id,
                    local_block_ids=meta.local_block_ids,
                    remote_block_ids=meta.remote_block_ids,
                    remote_engine_id=remote_engine_id,
                    remote_host=remote_host,
                    remote_handshake_port=remote_handshake_port_list[i][0],
                    offset=i,
                    tp_num_need_pulls=tp_num_need_pulls,
                    all_task_done=(i == tp_num_need_pulls * self._prefill_pp_size - 1),
                )

    for req_id in metadata.reqs_in_batch:
        if self.kv_send_thread is not None:
            self.kv_send_thread.task_tracker.add_req_to_process(req_id)
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

    if self.kv_send_thread is not None:
        for req_id, delay_start_time in metadata.requests_to_send.items():
            if self.tp_rank in self._prefill_get_remote_rank(req_id):
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
            else:
                self.kv_send_thread.add_not_transfer_request(req_id)


def _code_shape_compatible(target_func, new_func) -> bool:
    """Return True when ``new_func.__code__`` can safely replace ``target_func.__code__``.

    ``__code__`` replacement keeps the target function's own globals,
    defaults, kwdefaults and annotations, so the positional argument names
    must line up exactly for any caller that uses keyword arguments.
    """
    old_code = target_func.__code__
    new_code = new_func.__code__
    if (
        old_code.co_argcount != new_code.co_argcount
        or old_code.co_kwonlyargcount != new_code.co_kwonlyargcount
        or old_code.co_posonlyargcount != new_code.co_posonlyargcount
    ):
        return False
    old_args = old_code.co_varnames[: old_code.co_argcount]
    new_args = new_code.co_varnames[: new_code.co_argcount]
    return old_args == new_args


_ORIG_KV_RECV_INIT = None


def _patched_kv_recv_thread_init(self, *args, **kwargs):
    _ORIG_KV_RECV_INIT(self, *args, **kwargs)
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        SizedDict,
    )

    self.remote_block_lens: dict[str, dict[int, list[int]]] = SizedDict()
    self.remote_block_strides: dict[str, dict[int, list[int]]] = SizedDict()


def _patched_get_remote_metadata(self, remote_host, remote_handshake_port):
    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector as mod

    sock = None
    try:
        sock = self._get_remote_socket(remote_host, remote_handshake_port)
        mod.ensure_zmq_send(
            sock,
            self.encoder.encode((mod.GET_META_MSG, "")),
            f"{remote_host}:{remote_handshake_port}",
        )
        metadata_bytes = mod.ensure_zmq_recv(
            sock, f"{remote_host}:{remote_handshake_port}"
        )
        agent_meta = self.decoder.decode(metadata_bytes)
        engine_id = agent_meta.engine_id
        assert engine_id != self.local_engine_id, (
            f"Conflict engine id {engine_id} with local engine id "
            f"{self.local_engine_id}."
        )
        with self.remote_metadata_lock:
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = (
                agent_meta.kv_caches_base_addr
            )
            self.remote_te_port[engine_id][remote_handshake_port] = (
                agent_meta.te_rpc_port
            )
            self.remote_block_lens[engine_id][remote_handshake_port] = (
                agent_meta.block_lens
            )
            self.remote_block_strides[engine_id][remote_handshake_port] = (
                agent_meta.block_strides
            )
    except Exception:
        if isinstance(sock, mod.zmq.Socket):
            sock.close()
            sock = None
        raise
    finally:
        if sock is not None:
            self._return_remote_socket(sock, remote_host, remote_handshake_port)


def _patched_transfer_kv_cache_all_groups(self, req_meta):
    import time

    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector as mod

    remote_request_id = req_meta["remote_request_id"]
    remote_block_ids = req_meta["remote_block_ids"]
    local_block_ids = req_meta["local_block_ids"]
    remote_engine_id = req_meta["remote_engine_id"]
    remote_host = req_meta["remote_host"]
    remote_handshake_port = req_meta["remote_handshake_port"]

    num_local_blocks = sum(
        len(group_block_ids) for group_block_ids in local_block_ids
    )
    if num_local_blocks == 0:
        return

    with self.remote_metadata_lock:
        has_remote_metadata = (
            remote_engine_id in self.kv_caches_base_addr
            and remote_handshake_port in self.kv_caches_base_addr[remote_engine_id]
        )
    if not has_remote_metadata:
        self._get_remote_metadata(remote_host, remote_handshake_port)
    with self.remote_metadata_lock:
        remote_kv_caches_base_addrs = self.kv_caches_base_addr[
            remote_engine_id
        ][remote_handshake_port]
        local_kv_caches_base_addrs = self.kv_caches_base_addr[
            self.local_engine_id
        ][self.local_handshake_port]
        remote_transfer_port = self.remote_te_port[remote_engine_id][
            remote_handshake_port
        ]
        remote_block_lens = self.remote_block_lens[remote_engine_id][
            remote_handshake_port
        ]
        remote_block_strides = self.remote_block_strides[remote_engine_id][
            remote_handshake_port
        ]
    session_id = f"{remote_host}:{remote_transfer_port}"

    if len(local_kv_caches_base_addrs) != len(remote_kv_caches_base_addrs):
        raise RuntimeError(
            "Mooncake hybrid KV metadata mismatch: local KV cache address "
            f"count {len(local_kv_caches_base_addrs)} != remote KV cache "
            f"address count {len(remote_kv_caches_base_addrs)} for "
            f"remote_engine_id={remote_engine_id} "
            f"remote_handshake_port={remote_handshake_port}."
        )

    req_start_time = time.perf_counter()
    src_list, dst_list, length_list = [], [], []
    for i in range(self.hma_group_size):
        if not remote_block_ids[i] or not local_block_ids[i]:
            continue
        cur_remote_block_ids = remote_block_ids[i]
        cur_local_block_ids = local_block_ids[i]
        if (
            not isinstance(self.kv_cache_specs[i], mod.MambaSpec)
            and len(cur_local_block_ids) < len(cur_remote_block_ids)
        ):
            cur_remote_block_ids = cur_remote_block_ids[
                -len(cur_local_block_ids):
            ]
        grouped_remote_block_ids, grouped_local_block_ids = (
            mod.group_concurrent_contiguous(
                cur_remote_block_ids, cur_local_block_ids
            )
        )
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
            zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)
        ):
            if self.addr_group_idx and i not in self.addr_group_idx[k]:
                continue
            block_len = (
                remote_block_lens[k]
                if k < len(remote_block_lens)
                else self.block_len_per_addr[k]
            )
            block_stride = (
                remote_block_strides[k]
                if k < len(remote_block_strides)
                else self.block_stride_per_addr[k]
            )
            local_block_stride = (
                self.block_stride_per_addr[k]
                if k < len(self.block_stride_per_addr)
                else block_stride
            )
            for remote_block_id, local_block_id in zip(
                grouped_remote_block_ids, grouped_local_block_ids
            ):
                src = (
                    src_layer_base_addr
                    + local_block_id[0] * local_block_stride
                )
                dst = (
                    dst_layer_base_addr
                    + remote_block_id[0] * block_stride
                )
                length = block_len * len(local_block_id)
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)

    ret = self.engine.batch_transfer_sync_read(
        session_id, src_list, dst_list, length_list
    )
    if ret < 0:
        mod.logger.error(
            "Mooncake transfer failed for request. remote_request_id=%s, ret=%d. ",
            req_meta["remote_request_id"],
            ret,
        )
        raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

    req_end_time = time.perf_counter()
    req_transfer_elapsed = (req_end_time - req_start_time) * 1000
    mod.logger.info(
        "KV cache transfer for request %s took %.2f ms. local_ip %s "
        "local_device_id %s remote_session_id %s",
        remote_request_id,
        req_transfer_elapsed,
        mod.get_ip(),
        self.tp_rank,
        session_id,
    )


def _patch_method_code(cls, method_name: str, new_func) -> tuple[str, list[str]]:
    """Patch ``cls.method_name`` with ``new_func.__code__`` and report status."""
    target = getattr(cls, method_name, None)
    symbol = f"{cls.__name__}.{method_name}"
    if target is None:
        return symbol, [f"{symbol}: target method not found"]
    if not _code_shape_compatible(target, new_func):
        return symbol, [
            f"{symbol}: signature mismatch - target={target.__code__.co_varnames[: target.__code__.co_argcount]} "
            f"patch={new_func.__code__.co_varnames[: new_func.__code__.co_argcount]}"
        ]
    target.__code__ = new_func.__code__
    return symbol, []


def apply_hetero_mooncake_patch():
    """Apply the heterogeneous-TP Mooncake hybrid connector patch.

    Idempotent: re-running simply re-applies the same ``__code__`` objects.

    Returns:
        dict with ``patched`` (list of patched symbols) and ``mismatches``
        (list of signature/missing-target problems).
    """
    import msgspec

    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector as mod

    patched: list[str] = []
    mismatches: list[str] = []

    # The stock v0.23 MooncakeAgentMetadata lacks the fields introduced by
    # hetero_cp.  Replace the module-level msgspec struct with an extended one
    # so worker-constructed metadata carries handshake_port/block_strides and
    # scheduler-side decodes keep working.
    class PatchedMooncakeAgentMetadata(
        msgspec.Struct, omit_defaults=True, dict=True
    ):
        engine_id: str
        te_rpc_port: int
        block_size: int
        kv_caches_base_addr: list[int]
        num_blocks: int
        block_lens: list[int]
        ssm_sizes: tuple[int, int]
        local_ip: str = ""
        handshake_port: int = 0
        block_strides: list[int] = msgspec.field(default_factory=list)

    mod.MooncakeAgentMetadata = PatchedMooncakeAgentMetadata
    patched.append("mooncake_hybrid_connector.MooncakeAgentMetadata")

    # Module-level helper.  Inject into the target module's globals so the
    # copied method bodies resolve it exactly like the hetero final code does.
    mod.__dict__["get_dp_side_channel_port_offset"] = get_dp_side_channel_port_offset
    patched.append("mooncake_hybrid_connector.get_dp_side_channel_port_offset")

    # KVCacheRecvingThread requires two extra dicts and hetero-aware remote
    # block geometry.  Its __init__ is wrapped (whole-method replacement) to
    # keep the origin constructor untouched.
    global _ORIG_KV_RECV_INIT
    if _ORIG_KV_RECV_INIT is None:
        _ORIG_KV_RECV_INIT = mod.KVCacheRecvingThread.__init__
    mod.KVCacheRecvingThread.__init__ = _patched_kv_recv_thread_init
    patched.append("mooncake_hybrid_connector.KVCacheRecvingThread.__init__")
    _patch_method_code(
        mod.KVCacheRecvingThread,
        "_get_remote_metadata",
        _patched_get_remote_metadata,
    )
    _patch_method_code(
        mod.KVCacheRecvingThread,
        "_transfer_kv_cache_all_groups",
        _patched_transfer_kv_cache_all_groups,
    )

    method_targets = [
        (mod.MooncakeConnectorScheduler, "__init__", _patched_scheduler_init),
        (mod.MooncakeConnectorScheduler, "set_xfer_handshake_metadata", _patched_set_xfer_handshake_metadata),
        (mod.MooncakeConnectorWorker, "__init__", _patched_worker_init),
        (mod.MooncakeConnectorWorker, "_get_tp_num_need_pulls", _patched_get_tp_num_need_pulls),
        (mod.MooncakeConnectorWorker, "_get_remote_host_info_by_port", _patched_get_remote_host_info_by_port),
        (mod.MooncakeConnectorWorker, "_prefill_get_remote_rank", _patched_prefill_get_remote_rank),
        (mod.MooncakeConnectorWorker, "start_load_kv", _patched_start_load_kv),
    ]
    for cls, method_name, new_func in method_targets:
        symbol, problems = _patch_method_code(cls, method_name, new_func)
        if problems:
            mismatches.extend(problems)
            continue
        patched.append(f"mooncake_hybrid_connector.{symbol}")

    # Optional sibling patch: the plain mooncake_connector worker has the same
    # legacy handshake-port lookup and benefits from the absolute-port fix.
    try:
        import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector as plain_mod
    except ImportError:
        plain_mod = None
        logger.debug("plain mooncake_connector not importable; skipping optional patch")
    if plain_mod is not None:
        worker_cls = getattr(plain_mod, "MooncakeConnectorWorker", None)
        if worker_cls is not None and hasattr(worker_cls, "_get_remote_host_info_by_port"):
            symbol, problems = _patch_method_code(
                worker_cls, "_get_remote_host_info_by_port", _patched_get_remote_host_info_by_port
            )
            if problems:
                mismatches.extend(problems)
            else:
                patched.append(f"mooncake_connector.{symbol}")

    if mismatches:
        logger.warning("hetero mooncake patch mismatches: %s", mismatches)
    logger.info("hetero mooncake patch applied. patched=%s", patched)
    return {"patched": patched, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# MooncakeAgentMetadata notes
# ---------------------------------------------------------------------------
# The stock v0.23 MooncakeAgentMetadata lacks the ``handshake_port`` and
# ``block_strides`` fields introduced by hetero_cp.  apply_hetero_mooncake_patch
# replaces the module-level msgspec struct with an extended subclass-style
# struct (see PatchedMooncakeAgentMetadata above), so worker-constructed
# metadata and scheduler-side decodes both carry the extra fields.
