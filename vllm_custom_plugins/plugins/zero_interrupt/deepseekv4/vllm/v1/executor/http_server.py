#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""HTTP Server for ITS plugin using FastAPI.

This module provides an HTTP server for receiving deployment strategies
from the decision center and exposing status endpoints.
"""

import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request

from vllm.logger import logger
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.constants import (
    VLLM_ITS_HTTP_SERVER_PORT_START,
)
from vllm_custom_plugins.plugins.zero_interrupt.deepseekv4.common.types import DeployStrategy, DeployType, EngineParallelConfig, ServerInfo, NPUInfo, EngineNPUHealthyState, UpdateEngineInfo


class ITSHttpServer:
    """HTTP server for ITS plugin using FastAPI.

    Provides REST API endpoints for:
    - Receiving deployment strategies (POST /api/v1/executor/deploy)
    - Health check (GET /health)

    The strategy receiving is被动接收 (passive receive) - the decision center
    sends a POST request to this endpoint, and we process and notify the
    strategy sync thread.
    """

    def __init__(
            self,
            port: int = VLLM_ITS_HTTP_SERVER_PORT_START,
            strategy_sync_thread: Optional[Any] = None,
            expected_executor_id: Optional[str] = None,
            status_provider: Optional[Any] = None,
    ):
        """Initialize the HTTP server.

        Args:
            port: Port number to listen on
            strategy_sync_thread: Reference to StrategySyncThread for passive receiving
            expected_executor_id: If set, POST /deploy strategies whose
                top-level executor_id does not match are rejected with 400.
            status_provider: Optional zero-argument callable returning a dict
                of executor runtime fields (e.g. world_size) merged into the
                /status response.
        """
        self.port = port
        self.strategy_sync_thread = strategy_sync_thread
        self.status_provider = status_provider
        self.expected_executor_id = (
            None
            if expected_executor_id is None
            else str(expected_executor_id)
        )
        # DecisionMakingCenter assigns ``exe-<service>-<engine>-<n>`` during
        # /init_executor_state registration. Strategies delivered by the
        # center use that id, while the test scripts keep posting the local
        # numeric data_parallel_rank. Accept both.
        self.accepted_executor_ids: set[str] = set()
        if self.expected_executor_id is not None:
            self.accepted_executor_ids.add(self.expected_executor_id)
        self._app = FastAPI(title="ITS Executor API", version="1.0.0")
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._shutdown_event = threading.Event()
        # Reference to the running uvicorn Server so stop() can request a
        # clean exit of server.run() (which otherwise never returns and
        # keeps serving on the port after shutdown).
        self._uvicorn_server: Any = None

        self._setup_routes()

    def set_strategy_sync_thread(self, strategy_sync_thread: "Any") -> None:
        """Set the strategy sync thread reference.

        This allows the HTTP server to notify the strategy sync thread
        when a deployment strategy is received.

        Args:
            strategy_sync_thread: StrategySyncThread instance
        """
        self.strategy_sync_thread = strategy_sync_thread
        logger.info("Strategy sync thread reference set")

    def add_expected_executor_id(self, executor_id: Any) -> None:
        """Accept strategies addressed to an additional executor id.

        DecisionMakingCenter registers every executor and returns a generated
        id such as ``exe-<service_id>-<engine_uid>-<count>``. After the init
        report completes, this id must be accepted by the deploy endpoint in
        addition to the local numeric data_parallel_rank.
        """
        if executor_id is None:
            return
        self.accepted_executor_ids.add(str(executor_id))
        logger.info(
            "ITS HTTP server now accepts executor_id=%s (all accepted: %s)",
            executor_id,
            sorted(self.accepted_executor_ids),
        )

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self._app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "its-executor"}

        @self._app.get("/api/v1/executor/status")
        async def status():
            """Get executor status."""
            extra: dict[str, Any] = {}
            if self.status_provider is not None:
                try:
                    provided = self.status_provider()
                    if isinstance(provided, dict):
                        extra.update(provided)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "status_provider failed while building /status "
                        "response: %s",
                        exc,
                    )
            return {
                "status": "running",
                "port": self.port,
                "strategy_sync_configured": self.strategy_sync_thread is not None,
                **extra,
            }

        @self._app.get("/api/v1/executor/strategy")
        async def get_strategy():
            """Get current strategy status."""
            if self.strategy_sync_thread:
                current = self.strategy_sync_thread.get_current_strategy()
                if current:
                    return {
                        "status": "has_strategy",
                        "deploy_type": current.deploy_type.value,
                        "executor_id": current.executor_id,
                    }
                return {"status": "no_strategy"}
            return {"status": "not_configured"}

        @self._app.post("/api/v1/executor/deploy")
        async def deploy(request: Request):
            """Receive deployment strategy from decision center.

            This is the被动接收 (passive receive) endpoint.
            The decision center sends a POST request with the deployment strategy.
            """
            try:
                # Handle empty body or invalid JSON
                try:
                    data = await request.json()
                except Exception:
                    # Try to get body directly if JSON parsing fails
                    body = await request.body()
                    if not body:
                        raise HTTPException(status_code=400, detail="No data provided or invalid JSON")
                    # A non-empty body that is not valid JSON must be
                    # rejected.  Falling back to data={} would parse it as a
                    # default DEGRADE strategy with an empty executor_id and
                    # forward that to the strategy sync thread.
                    raise HTTPException(
                        status_code=400, detail="Invalid JSON body"
                    )

                # Parse the strategy from request
                strategy = self._parse_deploy_request(data)

                # Reject strategies addressed to another executor.  A wrong
                # top-level executor_id would otherwise make
                # _get_engine_parallel_config silently pick another DP's
                # topology and restart with the wrong tp/dp/rank offset.
                if (
                    self.accepted_executor_ids
                    and str(strategy.executor_id) not in self.accepted_executor_ids
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "executor_id mismatch: strategy is for "
                            f"'{strategy.executor_id}', this executor accepts "
                            f"{sorted(self.accepted_executor_ids)}"
                        ),
                    )

                logger.info(
                    f"Received deployment strategy from decision center: "
                    f"deploy_type={strategy.deploy_type.value}, executor_id={strategy.executor_id}"
                )

                # Notify the strategy sync thread (passive receiving)
                if self.strategy_sync_thread:
                    self.strategy_sync_thread.on_strategy_received(strategy)
                else:
                    logger.warning("No strategy sync thread configured, strategy not notified")

                return {
                    "status": "success",
                    "message": "Strategy received",
                    "deploy_type": strategy.deploy_type.value,
                    "executor_id": strategy.executor_id,
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing deploy request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _parse_deploy_request(self, data: dict[str, Any]) -> DeployStrategy:
        """Parse deployment request data.

        Args:
            data: Request JSON data

        Returns:
            DeployStrategy object
        """
        # Reject invalid deploy_type instead of silently degrading to DEGRADE.
        logger.info(f"parsing deploy request: {data}")
        try:
            deploy_type = DeployType(data.get("deploy_type", "DEGRADE"))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid deploy_type: {data.get('deploy_type')!r}",
            ) from e
        executor_id = data.get("executor_id", "")

        # Convert raw dict list to EngineParallelConfig objects
        config_list = data.get("engine_parallel_config", [])
        engine_parallel_config = [
            EngineParallelConfig(
                executor_id=config.get("executor_id", ""),
                dp=config.get("dp", 1),
                tp=config.get("tp", 1),
                data_parallel_rank=config.get("data_parallel_rank", -1),
                enable_expert_parallel=config.get("enable_expert_parallel", False),
                new_dp=config.get("new_dp"),
                new_tp=config.get("new_tp"),
                tp_asymmetric_shardings=config.get(
                    "tp_asymmetric_shardings", None
                ),
            )
            for config in config_list
        ]

        npu_state_data_list = data.get("engine_npu_healthy_state", [])

        def parse_server_list(npu_info_data: dict[str, Any]) -> list[Any]:
            """
            Parse server list from request data
            """
            server_list = []
            for server in npu_info_data.get("server_list", []):
                devices = [
                    NPUInfo(
                        npu_id=d["npu_id"],
                        device_ip=d["device_ip"],
                        rank_id=d["rank_id"],
                        healthy=d["npu_healthy"],
                    )
                    for d in server.get("device", [])
                ]
                server_list.append(
                    ServerInfo(
                        server_id=server["server_id"],
                        host_ip=server["host_ip"],
                        device=devices,
                    )
                )
            return server_list

        engine_npu_healthy_state = [
            EngineNPUHealthyState(
                server_count=npu_state_data.get("server_count", "1"),
                status=npu_state_data.get("status", "completed"),
                version=npu_state_data.get("version", "1.0"),
                server_list=parse_server_list(npu_state_data),
            )
            for npu_state_data in npu_state_data_list
        ]

        update_engine_info = None
        raw_update_info = data.get("update_engine_info")
        if isinstance(raw_update_info, dict):
            orig_engine_id = raw_update_info.get("orig_engine_id")
            new_engine_id = raw_update_info.get("new_engine_id")
            if orig_engine_id is not None and new_engine_id is not None:
                update_engine_info = UpdateEngineInfo(
                    orig_engine_id=str(orig_engine_id),
                    new_engine_id=str(new_engine_id),
                )

        strategy_generation = data.get("strategy_generation")
        if strategy_generation is not None:
            strategy_generation = str(strategy_generation)

        barrier_master_port = data.get("barrier_master_port")
        if barrier_master_port is not None:
            try:
                barrier_master_port = int(barrier_master_port)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "barrier_master_port must be an integer, got "
                        f"{barrier_master_port!r}"
                    ),
                )

        return DeployStrategy(
            deploy_type=deploy_type,
            executor_id=executor_id,
            engine_parallel_config=engine_parallel_config,
            engine_npu_healthy_state=engine_npu_healthy_state,
            update_engine_info=update_engine_info,
            strategy_generation=strategy_generation,
            barrier_master_port=barrier_master_port,
        )

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._running:
            logger.warning("HTTP server already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="ITS-HttpServer",
        )
        self._server_thread.start()
        logger.info(f"HTTP server started on port {self.port}, waiting for strategy from decision center")

    def _run_server(self) -> None:
        """Run the FastAPI application using uvicorn."""
        try:
            import uvicorn

            # Configure uvicorn
            config = uvicorn.Config(
                self._app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(config)

            # stop() may race with thread startup (for example when the
            # executor shuts down immediately after a failed init).  If a
            # stop was already requested, do not run the server: it would
            # otherwise keep the port bound forever because uvicorn's
            # run() loop never observes _running/_shutdown_event.
            if not self._running or self._shutdown_event.is_set():
                return

            # Run server
            self._uvicorn_server.run()

        except Exception as e:
            logger.error(f"HTTP server error: {e}")
        finally:
            self._running = False
            self._uvicorn_server = None

    def stop(self) -> None:
        """Stop the HTTP server."""
        self._running = False
        self._shutdown_event.set()
        # server.run() does not observe _running/_shutdown_event; request a
        # clean uvicorn exit so the port is actually released.  Do not early
        # return when _running is already False: stop() can race with the
        # server thread creating the uvicorn Server, and in that case the
        # thread must still observe should_exit / _shutdown_event after it
        # finishes constructing the server.
        server = self._uvicorn_server
        if server is not None:
            server.should_exit = True
        logger.info("HTTP server stopped")

    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if running, False otherwise
        """
        return self._running

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance.

        Returns:
            FastAPI application
        """
        return self._app