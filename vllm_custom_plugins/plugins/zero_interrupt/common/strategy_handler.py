#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Strategy Handler for ITS Worker.

This module handles deployment strategy execution in the worker process,
including PD chain rebuild for P/D分离 scenarios.
"""

import os
import zmq

from typing import Any

from vllm.logger import logger
from vllm_custom_plugins.plugins.zero_interrupt.common.constants import VLLM_ITS_ENABLE_PD_REBUILD
from vllm_custom_plugins.plugins.zero_interrupt.common.types import (
    DeployStrategy,
    DeployType,
    EngineNPUHealthyState,
    EngineParallelConfig,
)

class StrategyHandler:
    """Handler for deployment strategies in worker process.

    This handler processes deployment strategies:
    - STOP/DEGRADE/RECOVER: Worker restart handled by Executor (no-op here)
    - PD_REBUILD: Online P/D KV-Cache chain rebuild (no worker restart)
    """

    def __init__(
        self,
        worker,
        pd_rebuild_enabled: bool = VLLM_ITS_ENABLE_PD_REBUILD,
    ):
        """Initialize strategy handler.

        Args:
            worker: Worker instance
            pd_rebuild_enabled: Enable PD chain rebuild
        """
        self._worker = worker
        self._pd_rebuild_enabled = pd_rebuild_enabled

        # KV Connector reference (set during initialization)
        self._kv_connector = None
        self._transfer_engine = None

    def set_kv_connector(self, kv_connector: Any) -> None:
        """Set the KV connector reference.

        Args:
            kv_connector: MooncakeConnector or similar KV connector
        """
        self._kv_connector = kv_connector

    def _get_pd_role(self) -> str:
        """获取当前实例的 P/D 角色

        通过 kv_transfer_config.kv_role 判断:
        - kv_producer = P_INSTANCE (Prefill)
        - kv_consumer = D_INSTANCE (Decode)

        Returns:
            P_INSTANCE 或 D_INSTANCE
        """
        # 尝试从 worker 获取 kv_transfer_config
        kv_role = None
        if self._worker is not None:
            vllm_config = getattr(self._worker, "vllm_config", None)
            if vllm_config is not None:
                kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
                if kv_transfer_config is not None:
                    kv_role = getattr(kv_transfer_config, "kv_role", None)

        if kv_role == "kv_producer":
            return "P_INSTANCE"
        elif kv_role == "kv_consumer":
            return "D_INSTANCE"

        # 备用：从环境变量获取
        return os.getenv("VLLM_PD_ROLE", "D_INSTANCE")

    def execute_strategy(self, strategy: DeployStrategy) -> bool:
        """Execute deployment strategy.

        Note: For STOP/DEGRADE/RECOVER strategies, the Worker is already
        restarted by Executor via _cleanup_and_restart_workers(), so model
        and communication groups are already re-initialized.

        Only PD_REBUILD requires actual work as it doesn't restart Worker.

        Args:
            strategy: Deployment strategy to execute

        Returns:
            True if successful
        """
        logger.info(f"Executing strategy: {strategy.deploy_type.value}")

        # STOP/DEGRADE/RECOVER: Worker already restarted by Executor
        if strategy.deploy_type in (DeployType.STOP, DeployType.DEGRADE, DeployType.RECOVER):
            logger.info(f"Strategy {strategy.deploy_type.value}: Worker restart handled by Executor")
            return True

        # PD_REBUILD: Online KV chain rebuild (no worker restart)
        if strategy.deploy_type == DeployType.PD_REBUILD:
            return self._execute_pd_rebuild(strategy)

        logger.warning(f"Unknown deploy type: {strategy.deploy_type}")
        return False

    def _execute_pd_rebuild(self, strategy: DeployStrategy) -> bool:
        """Execute PD chain rebuild strategy.

        This handles the P/D分离 scenario where KV-Cache transfer paths
        need to be rebuilt due to instance availability changes.

        对于 D 实例，需要处理非对称 TP 场景：
        - P 实例 TP=3，D 实例 TP=4
        - D-rank-0 → P-rank-0 拉取
        - D-rank-1 → P-rank-1 拉取
        - D-rank-2 → P-rank-2 拉取
        - D-rank-3 → P-rank-0 拉取（循环）

        Args:
            strategy: PD rebuild strategy

        Returns:
            True if successful
        """
        if not self._pd_rebuild_enabled:
            logger.warning("PD rebuild not enabled")
            return False

        logger.info("Starting PD chain rebuild")

        try:
            # 获取当前实例的 P/D 角色
            pd_role = self._get_pd_role()
            logger.info(f"Current instance role: {pd_role}")

            # 获取对端 P 实例配置（用于计算映射）
            p_configs = self._get_peer_configs(strategy, "P_INSTANCE")
            if not p_configs:
                logger.warning("No P instance configs found in strategy")

            # 获取当前 worker 的 rank
            worker_rank = getattr(self._worker, "rank", 0)

            # 计算非对称映射（仅 D 实例需要）
            kv_fetch_mapping = None
            if pd_role == "D_INSTANCE" and p_configs:
                kv_fetch_mapping = self._calculate_kv_fetch_mapping(
                    p_configs, worker_rank
                )
                logger.info(
                    f"D instance rank={worker_rank} mapping to P rank={kv_fetch_mapping.get('target_p_rank')}, "
                    f"P TP size={kv_fetch_mapping.get('p_tp_size')}"
                )

            # Get healthy NPU state from strategy
            npu_state = strategy.engine_npu_healthy_state

            # Step 1: Detect available peer instances from healthy state
            # 注意：这里返回所有健康的实例，P 和 D 实例都能检测到
            # _establish_peer_connections 会根据 kv_fetch_mapping 过滤
            peer_instances = self._detect_available_peers(npu_state)
            logger.info(f"Detected {len(peer_instances)} available peer instances")

            # Step 2: Cleanup existing KV transfer connections
            self._cleanup_kv_connections()

            # Step 3: Reinitialize Transfer Engine if needed
            self._reinit_transfer_engine()

            # Step 4: Rebuild KV-Cache transfer chain with new peers
            self._rebuild_kv_chain(peer_instances, kv_fetch_mapping=kv_fetch_mapping)

            # Step 5: Re-register KV caches
            self._reregister_kv_caches()

            logger.info("PD chain rebuild completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in PD chain rebuild: {e}")
            return False

    def _get_peer_configs(
        self, strategy: DeployStrategy, role: str
    ) -> list[EngineParallelConfig]:
        """获取指定角色的对端实例配置

        通过 executor_id 区分 P/D 实例：
        - 偶数 executor_id = P 实例 (Prefill)
        - 奇数 executor_id = D 实例 (Decode)

        Args:
            strategy: PD rebuild strategy
            role: P_INSTANCE 或 D_INSTANCE

        Returns:
            指定角色的 EngineParallelConfig 列表
        """
        target_is_p = (role == "P_INSTANCE")
        configs = []

        for config in strategy.engine_parallel_config:
            executor_id = getattr(config, "executor_id", 0)
            # 偶数 executor_id = P 实例，奇数 = D 实例
            is_p = (executor_id % 2 == 0)

            if is_p == target_is_p:
                configs.append(config)

        logger.debug(f"Found {len(configs)} configs for role={role}")
        return configs

    def _calculate_kv_fetch_mapping(
        self, p_configs: list[EngineParallelConfig], d_local_rank: int
    ) -> dict[str, Any]:
        """计算 D 实例从 P 实例拉取 KV 的地址映射

        场景：P 实例 TP=3，D 实例 TP=4
        - D-rank-0 → P-rank-0 拉取
        - D-rank-1 → P-rank-1 拉取
        - D-rank-2 → P-rank-2 拉取
        - D-rank-3 → P-rank-0 拉取（循环）

        Args:
            p_configs: P 实例的并行配置列表
            d_local_rank: D 实例的本地 rank

        Returns:
            映射信息字典
        """
        if not p_configs:
            logger.warning("No P configs provided, using default mapping")
            return {"target_p_rank": 0, "p_tp_size": 1, "kv_offset_per_rank": 1}

        # 获取 P 实例的 TP 大小（优先使用 new_tp，其次使用 tp）
        first_config = p_configs[0]
        p_tp_size = getattr(first_config, "new_tp", None)
        if p_tp_size is None or p_tp_size == 0:
            p_tp_size = first_config.tp if hasattr(first_config, "tp") else 1

        # 轮询映射：D rank 对 P rank 做模运算
        target_p_rank = d_local_rank % p_tp_size if p_tp_size > 0 else 0

        return {
            "target_p_rank": target_p_rank,
            "p_tp_size": p_tp_size,
            "kv_offset_per_rank": p_tp_size,  # 每个 P rank 负责的 KV 范围
        }

    def _detect_available_peers(
        self, npu_state: list[EngineNPUHealthyState] | None
    ) -> list[dict[str, Any]]:
        """Detect available P/D instances from NPU healthy state.

        Args:
            npu_state: NPU healthy state list from strategy

        Returns:
            List of available peer instance information
        """
        peer_instances = []

        if npu_state is None:
            logger.warning("No NPU state provided, cannot detect peers")
            return peer_instances

        # Iterate over the list of NPU healthy states
        for state in npu_state:
            for server in state.server_list:
                # Check if server has healthy NPUs
                healthy_devices = [d for d in server.device if d.healthy]
                if not healthy_devices:
                    continue

                peer_instances.append({
                    "server_id": server.server_id,
                    "host_ip": server.host_ip,
                    "healthy_devices": [d.npu_id for d in healthy_devices],
                    "healthy_ranks": [d.rank_id for d in healthy_devices],
                })

        return peer_instances

    def _cleanup_kv_connections(self) -> None:
        """Cleanup existing KV transfer connections.

        This includes:
        - Closing sockets
        - Stopping send/receive threads
        - Clearing transfer engine state
        """
        logger.info("Cleaning up existing KV connections")

        try:
            if self._kv_connector is not None:
                # Get worker side connector
                worker = getattr(self._kv_connector, "connector_worker", None)
                if worker is not None:
                    # Close sockets
                    if hasattr(worker, "sockets"):
                        for sock in worker.sockets.values():
                            try:
                                sock.close()
                            except Exception as e:
                                logger.warning(f"Error closing socket: {e}")
                        worker.sockets.clear()

                    # Stop KV threads
                    if hasattr(worker, "kv_send_thread") and worker.kv_send_thread:
                        worker.kv_send_thread = None
                    if hasattr(worker, "kv_recv_thread") and worker.kv_recv_thread:
                        worker.kv_recv_thread = None

                    # Clear metadata
                    if hasattr(worker, "xfer_handshake_metadata"):
                        worker.xfer_handshake_metadata = None
                    if hasattr(worker, "local_remote_block_port_mapping"):
                        worker.local_remote_block_port_mapping.clear()
                    if hasattr(worker, "remote_port_send_num"):
                        worker.remote_port_send_num.clear()

            logger.info("KV connections cleanup completed")
        except Exception as e:
            logger.warning(f"Error during KV connection cleanup: {e}")

    def _reinit_transfer_engine(self) -> None:
        """Reinitialize the Transfer Engine.

        This is needed when:
        - Peer instances change
        - Network topology changes
        - Previous engine connection is stale
        """
        logger.info("Reinitializing Transfer Engine")

        try:
            # Import the global transfer engine
            from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
                global_te,
            )

            # Get the transfer engine instance
            engine = global_te.get_transfer_engine(
                hostname="",  # Will use existing or reinitialize
                device_name=None,
            )

            self._transfer_engine = engine
            logger.info("Transfer Engine reinitialized")

        except ImportError as e:
            logger.warning(f"Mooncake not available: {e}")
        except Exception as e:
            logger.error(f"Error reinitializing Transfer Engine: {e}")
            raise

    def _rebuild_kv_chain(
        self,
        peer_instances: list[dict[str, Any]],
        kv_fetch_mapping: dict[str, Any] | None = None,
    ) -> None:
        """Rebuild KV-Cache transfer chain with new peer instances.

        This establishes new KV-Cache transfer paths to available
        P/D instances based on the healthy NPU state.

        For D instances with asymmetric TP (e.g., P TP=3, D TP=4),
        uses kv_fetch_mapping to determine which P rank to fetch from.

        Args:
            peer_instances: List of available peer instances
            kv_fetch_mapping: D 实例的 KV 拉取映射（可选）
        """
        logger.info(f"Rebuilding KV-Cache chain with {len(peer_instances)} peers")

        try:
            if self._kv_connector is None:
                logger.warning("No KV connector available, attempting to get from worker")
                self._kv_connector = self._get_kv_connector_from_worker()

            if self._kv_connector is None:
                logger.error("Cannot rebuild KV chain without KV connector")
                return

            # Get the worker side connector
            worker = getattr(self._kv_connector, "connector_worker", None)
            if worker is None:
                logger.error("Cannot get worker connector")
                return

            # Rebuild handshake metadata with new peers
            self._rebuild_handshake_metadata(worker, peer_instances)

            # Establish new connections to peers (with asymmetric mapping if provided)
            self._establish_peer_connections(worker, peer_instances, kv_fetch_mapping)

            logger.info("KV-Cache chain rebuilt successfully")

        except Exception as e:
            logger.error(f"Error rebuilding KV chain: {e}")
            raise

    def _get_kv_connector_from_worker(self) -> Any:
        """Get KV connector from worker instance.

        Returns:
            KV connector instance or None
        """
        # Try to get from worker
        if hasattr(self._worker, "kv_connector"):
            return self._worker.kv_connector

        # Try to get from worker's model
        if hasattr(self._worker, "model") and hasattr(self._worker.model, "kv_connector"):
            return self._worker.model.kv_connector

        return None

    def _rebuild_handshake_metadata(
        self, worker: Any, peer_instances: list[dict[str, Any]]
    ) -> None:
        """Rebuild handshake metadata for P/D connection.

        Args:
            worker: Worker connector instance
            peer_instances: Available peer instances
        """
        logger.info("Rebuilding handshake metadata")

        try:
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
                MooncakeAgentMetadata,
            )

            # Get current worker info
            side_channel_host = getattr(worker, "side_channel_host", None)
            handshake_port = getattr(worker, "handshake_port", None)

            if side_channel_host is None or handshake_port is None:
                logger.warning("Cannot get worker handshake info")
                return

            # Build new metadata for each peer
            # The actual metadata structure depends on P/D role
            metadata = MooncakeAgentMetadata(
                host=side_channel_host,
                port=handshake_port,
                engine_id=getattr(worker, "engine_id", "0"),
                tp_rank=getattr(worker, "tp_rank", 0),
                pp_rank=getattr(worker, "pp_rank", 0),
                dp_rank=getattr(worker, "dp_rank", 0),
            )

            worker.xfer_handshake_metadata = metadata
            logger.info("Handshake metadata rebuilt")

        except ImportError:
            logger.warning("Mooncake connector not available for metadata rebuild")
        except Exception as e:
            logger.warning(f"Error rebuilding handshake metadata: {e}")

    def _establish_peer_connections(
        self,
        worker: Any,
        peer_instances: list[dict[str, Any]],
        kv_fetch_mapping: dict[str, Any] | None = None,
    ) -> None:
        """Establish new connections to peer instances.

        For D instances with asymmetric TP, uses kv_fetch_mapping to determine
        which P rank to connect to.

        Args:
            worker: Worker connector instance
            peer_instances: Available peer instances
            kv_fetch_mapping: D 实例的 KV 拉取映射（可选）
        """
        logger.info("Establishing peer connections")

        try:
            # Get port configuration
            base_port = getattr(worker, "side_channel_port", None)
            if base_port is None:
                logger.warning("Cannot get base port for peer connections")
                return

            # 如果有非对称映射，计算目标 P rank
            target_p_rank = None
            if kv_fetch_mapping is not None:
                target_p_rank = kv_fetch_mapping.get("target_p_rank")
                logger.info(f"Using asymmetric mapping: target P rank = {target_p_rank}")

            # Build socket connections to each peer
            for peer in peer_instances:
                peer_host = peer["host_ip"]
                # Calculate peer port based on device index
                for npu_id in peer["healthy_devices"]:
                    # 如果有非对称映射，只连接目标 P rank
                    if target_p_rank is not None and npu_id != target_p_rank:
                        continue

                    peer_port = base_port + npu_id
                    self._create_peer_socket(worker, peer_host, peer_port, npu_id)

            logger.info(f"Established connections to {len(peer_instances)} peers")

        except Exception as e:
            logger.error(f"Error establishing peer connections: {e}")
            raise

    def _create_peer_socket(
        self, worker: Any, host: str, port: int, npu_id: int
    ) -> None:
        """Create a socket connection to a peer instance.

        Args:
            worker: Worker connector instance
            host: Peer host IP
            port: Peer port
            npu_id: Device ID for this connection
        """
        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PAIR)
            sock.setsockopt(zmq.LINGER, 0)

            # Connect to peer
            connection_str = f"tcp://{host}:{port}"
            sock.connect(connection_str)

            # Store socket
            if not hasattr(worker, "sockets"):
                worker.sockets = {}

            worker.sockets[npu_id] = sock
            logger.debug(f"Connected to peer at {connection_str}")

        except Exception as e:
            logger.warning(f"Error creating socket to {host}:{port}: {e}")

    def _reregister_kv_caches(self) -> None:
        """Re-register KV caches after chain rebuild."""
        logger.info("Re-registering KV caches")

        try:
            if self._kv_connector is None:
                return

            # Get worker connector
            worker = getattr(self._kv_connector, "connector_worker", None)
            if worker is None or not hasattr(worker, "kv_caches"):
                return

            # Re-register KV caches if they exist
            kv_caches = getattr(worker, "kv_caches", None)
            if kv_caches and hasattr(worker, "register_kv_caches"):
                worker.register_kv_caches(kv_caches)
                logger.info("KV caches re-registered")

        except Exception as e:
            logger.warning(f"Error re-registering KV caches: {e}")


class ITSMooncakeConnectorV1:
    """ITS-specific MooncakeConnector for PD rebuild support.

    This class extends the standard MooncakeConnector to provide
    additional methods for PD chain rebuild during deployment.
    """

    def __init__(self, original_connector: Any):
        """Initialize with original connector.

        Args:
            original_connector: Original MooncakeConnector instance
        """
        self._connector = original_connector

    def rebuild_for_deployment(self, peer_instances: list[dict[str, Any]]) -> bool:
        """Rebuild connector for new deployment configuration.

        Args:
            peer_instances: New list of peer instances

        Returns:
            True if successful
        """
        logger.info("Rebuilding MooncakeConnector for deployment")

        try:
            # Cleanup existing connections
            self._cleanup()

            # Rebuild connections
            worker = getattr(self._connector, "connector_worker", None)
            if worker is not None:
                # Re-establish connections
                for peer in peer_instances:
                    self._connect_to_peer(worker, peer)

            logger.info("MooncakeConnector rebuild completed")
            return True

        except Exception as e:
            logger.error(f"Error rebuilding connector: {e}")
            return False

    def _cleanup(self) -> None:
        """Cleanup existing connections."""
        try:
            worker = getattr(self._connector, "connector_worker", None)
            if worker is not None and hasattr(worker, "sockets"):
                for sock in worker.sockets.values():
                    sock.close()
                worker.sockets.clear()
        except Exception as e:
            logger.warning(f"Error in cleanup: {e}")

    def _connect_to_peer(self, worker: Any, peer: dict[str, Any]) -> None:
        """Connect to a peer instance.

        Args:
            worker: Worker connector
            peer: Peer instance info
        """
        peer_host = peer["host_ip"]
        base_port = getattr(worker, "side_channel_port", 0)

        for npu_id in peer.get("healthy_devices", []):
            port = base_port + npu_id
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PAIR)
            sock.connect(f"tcp://{peer_host}:{port}")

            if not hasattr(worker, "sockets"):
                worker.sockets = {}
            worker.sockets[npu_id] = sock