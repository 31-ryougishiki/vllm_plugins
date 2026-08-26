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
from vllm_custom_plugins.plugins.zero_interrupt.common.constants import (
    VLLM_ITS_HTTP_SERVER_PORT_START,
)
from vllm_custom_plugins.plugins.zero_interrupt.common.types import DeployStrategy, DeployType, EngineParallelConfig, ServerInfo, NPUInfo, EngineNPUHealthyState


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
    ):
        """Initialize the HTTP server.

        Args:
            port: Port number to listen on
            strategy_sync_thread: Reference to StrategySyncThread for passive receiving
        """
        self.port = port
        self.strategy_sync_thread = strategy_sync_thread
        self._app = FastAPI(title="ITS Executor API", version="1.0.0")
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._shutdown_event = threading.Event()

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

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self._app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "its-executor"}

        @self._app.get("/api/v1/executor/status")
        async def status():
            """Get executor status."""
            return {
                "status": "running",
                "port": self.port,
                "strategy_sync_configured": self.strategy_sync_thread is not None,
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
                    data = {}

                # Parse the strategy from request
                strategy = self._parse_deploy_request(data)

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
        # Handle invalid deploy type gracefully
        logger.info(f"parsing deploy request: {data}")
        try:
            deploy_type = DeployType(data.get("deploy_type", "DEGRADE"))
        except ValueError:
            logger.warning(f"Invalid deploy_type: {data.get('deploy_type')}, using DEGRADE")
            deploy_type = DeployType.DEGRADE
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

        return DeployStrategy(
            deploy_type=deploy_type,
            executor_id=executor_id,
            engine_parallel_config=engine_parallel_config,
            engine_npu_healthy_state=engine_npu_healthy_state,
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
            server = uvicorn.Server(config)

            # Run server
            server.run()

        except Exception as e:
            logger.error(f"HTTP server error: {e}")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()
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