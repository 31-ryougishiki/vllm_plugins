#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Test cases for ITS HTTP Server using FastAPI."""

import json
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Add parent directories to path for imports
# Use absolute path to ensure it works from any directory
_plugin_dir = r"D:\code\python_code\fork\vLLM\plugins\vllm_custom_plugins\vllm_custom_plugins\plugins\zero_interrupt"
_plugins_dir = r"D:\code\python_code\fork\vLLM\plugins\vllm_custom_plugins\vllm_custom_plugins\plugins"
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)
if _plugins_dir not in sys.path:
    sys.path.insert(0, _plugins_dir)

from fastapi.testclient import TestClient
from zero_interrup.executor.http_server import ITSHttpServer


class TestITSHttpServer:
    """Test cases for ITSHttpServer with FastAPI."""

    @pytest.fixture
    def http_server(self):
        """Create HTTP server instance for testing."""
        server = ITSHttpServer(port=18001)
        yield server
        if server.is_running():
            server.stop()

    @pytest.fixture
    def client(self, http_server):
        """Create FastAPI test client."""
        return TestClient(http_server._app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "its-executor"

    def test_status_endpoint_no_strategy_sync(self, client):
        """Test status endpoint without strategy sync thread."""
        response = client.get("/api/v1/executor/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["port"] == 18001
        assert data["strategy_sync_configured"] is False

    def test_status_endpoint_with_strategy_sync(self, http_server, client):
        """Test status endpoint with strategy sync thread."""
        # Create mock strategy sync thread
        class MockStrategySyncThread:
            def get_current_strategy(self):
                return None

        http_server.set_strategy_sync_thread(MockStrategySyncThread())

        response = client.get("/api/v1/executor/status")

        assert response.status_code == 200
        data = response.json()
        assert data["strategy_sync_configured"] is True

    def test_strategy_endpoint_not_configured(self, client):
        """Test strategy endpoint when not configured."""
        response = client.get("/api/v1/executor/strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_configured"

    def test_strategy_endpoint_no_strategy(self, http_server, client):
        """Test strategy endpoint when no strategy is set."""
        class MockStrategySyncThread:
            def get_current_strategy(self):
                return None

        http_server.set_strategy_sync_thread(MockStrategySyncThread())

        response = client.get("/api/v1/executor/strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_strategy"

    def test_strategy_endpoint_with_strategy(self, http_server, client):
        """Test strategy endpoint when strategy is set."""
        from zero_interrupt.common.types import DeployStrategy, DeployType, EngineParallelConfig

        class MockStrategySyncThread:
            def get_current_strategy(self):
                return DeployStrategy(
                    deploy_type=DeployType.DEGRADE,
                    executor_id=0,
                    engine_parallel_config=[
                        EngineParallelConfig(
                            executor_id=0, dp=2, tp=4, data_parallel_rank=0, new_tp=2, new_dp=2
                        )
                    ],
                    engine_npu_healthy_state=[],
                )

        http_server.set_strategy_sync_thread(MockStrategySyncThread())

        response = client.get("/api/v1/executor/strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "has_strategy"
        assert data["deploy_type"] == "DEGRADE"
        assert data["executor_id"] == 0

    def test_deploy_endpoint_no_data(self, client):
        """Test deploy endpoint with no data."""
        response = client.post(
            "/api/v1/executor/deploy",
            content="",
            headers={"Content-Type": "application/json"}
        )

        # FastAPI will return 422 for empty body with required fields
        assert response.status_code in [400, 422]

    def test_deploy_endpoint_invalid_json(self, client):
        """Test deploy endpoint with invalid JSON.

        Note: FastAPI is more lenient with invalid JSON than Flask.
        It accepts any body as long as the content-type is correct.
        """
        response = client.post(
            "/api/v1/executor/deploy",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        # FastAPI returns 200 because it treats non-JSON body as empty and uses defaults
        assert response.status_code == 200

    def test_deploy_endpoint_minimal_data(self, client):
        """Test deploy endpoint with minimal required data."""
        data = {"deploy_type": "DEGRADE", "executor_id": 0}

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["deploy_type"] == "DEGRADE"
        assert result["executor_id"] == 0

    def test_deploy_endpoint_with_parallel_config(self, client):
        """Test deploy endpoint with parallel config."""
        data = {
            "deploy_type": "DEGRADE",
            "executor_id": 1,
            "engine_parallel_config": [
                {"executor_id": 0, "dp": 2, "tp": 4, "new_tp": 2, "new_dp": 2},
                {"executor_id": 1, "dp": 2, "tp": 4, "new_tp": 2, "new_dp": 2},
            ],
        }

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

    def test_deploy_endpoint_all_deploy_types(self, client):
        """Test deploy endpoint with all deploy types."""
        deploy_types = ["STOP", "DEGRADE", "RECOVER", "PD_REBUILD"]

        for deploy_type in deploy_types:
            data = {"deploy_type": deploy_type, "executor_id": 0}

            response = client.post(
                "/api/v1/executor/deploy",
                json=data,
            )

            assert response.status_code == 200
            result = response.json()
            assert result["deploy_type"] == deploy_type

    def test_deploy_endpoint_with_npu_healthy_state(self, client):
        """Test deploy endpoint with NPU healthy state (PD_REBUILD scenario)."""
        data = {
            "deploy_type": "PD_REBUILD",
            "executor_id": 0,
            "engine_parallel_config": [{"executor_id": 0, "dp": 2, "tp": 4}],
            "engine_npu_healthy_state": [
                {
                    "server_count": "1",
                    "status": "completed",
                    "version": "1.0",
                    "server_list": [
                        {
                            "server_id": "0",
                            "host_ip": "192.168.1.10",
                            "device": [
                                {
                                    "npu_id": 0,
                                    "device_ip": "192.168.1.10",
                                    "rank_id": "0",
                                    "healthy": True,
                                },
                                {
                                    "npu_id": 1,
                                    "device_ip": "192.168.1.10",
                                    "rank_id": "1",
                                    "healthy": False,
                                },
                            ],
                        }
                    ],
                }
            ],
        }

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["deploy_type"] == "PD_REBUILD"

    def test_deploy_endpoint_with_strategy_notification(self, http_server, client):
        """Test deploy endpoint triggers strategy notification."""
        received_strategy = []

        class MockStrategySyncThread:
            def on_strategy_received(self, strategy):
                received_strategy.append(strategy)

        http_server.set_strategy_sync_thread(MockStrategySyncThread())

        data = {"deploy_type": "DEGRADE", "executor_id": 0}

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200
        assert len(received_strategy) == 1
        assert received_strategy[0].deploy_type.value == "DEGRADE"
        assert received_strategy[0].executor_id == 0

    def test_deploy_endpoint_without_strategy_sync(self, client):
        """Test deploy endpoint when strategy sync not configured."""
        data = {"deploy_type": "DEGRADE", "executor_id": 0}

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        # Should still return success but log warning
        assert response.status_code == 200

    def test_deploy_endpoint_invalid_deploy_type(self, client):
        """Test deploy endpoint with invalid deploy type."""
        data = {"deploy_type": "INVALID_TYPE", "executor_id": 0}

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        # DeployType will handle invalid type with fallback to DEGRADE
        assert response.status_code == 200
        result = response.json()
        assert result["deploy_type"] == "DEGRADE"  # Falls back to DEGRADE

    def test_deploy_endpoint_with_enable_expert_parallel(self, client):
        """Test deploy endpoint with expert parallel enabled."""
        data = {
            "deploy_type": "DEGRADE",
            "executor_id": 0,
            "engine_parallel_config": [
                {"executor_id": 0, "dp": 2, "tp": 4, "enable_expert_parallel": True}
            ],
        }

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200

    def test_deploy_endpoint_empty_parallel_config(self, client):
        """Test deploy endpoint with empty parallel config."""
        data = {"deploy_type": "STOP", "executor_id": 0, "engine_parallel_config": []}

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200


class TestITSHttpServerEdgeCases:
    """Test edge cases for ITS HTTP Server with FastAPI."""

    @pytest.fixture
    def http_server(self):
        """Create HTTP server instance for testing."""
        server = ITSHttpServer(port=18002)
        yield server
        if server.is_running():
            server.stop()

    @pytest.fixture
    def client(self, http_server):
        """Create FastAPI test client."""
        return TestClient(http_server._app)

    def test_health_with_query_params(self, client):
        """Test health endpoint ignores query params."""
        response = client.get("/health?extra=param")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_deploy_with_extra_fields(self, client):
        """Test deploy endpoint ignores extra fields."""
        data = {
            "deploy_type": "DEGRADE",
            "executor_id": 0,
            "extra_field": "should_be_ignored",
            "another_field": {"nested": "value"},
        }

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200

    def test_deploy_with_none_values(self, client):
        """Test deploy endpoint with None values in optional fields."""
        data = {
            "deploy_type": "DEGRADE",
            "executor_id": 0,
            "engine_parallel_config": [
                {"executor_id": 0, "dp": 2, "tp": 4, "new_tp": None, "new_dp": None}
            ],
        }

        response = client.post(
            "/api/v1/executor/deploy",
            json=data,
        )

        assert response.status_code == 200

    def test_status_returns_correct_port(self, http_server, client):
        """Test status returns the correct configured port."""
        response = client.get("/api/v1/executor/status")

        data = response.json()
        assert data["port"] == 18002


class TestITSHttpServerFastAPIFeatures:
    """Test FastAPI-specific features."""

    @pytest.fixture
    def http_server(self):
        """Create HTTP server instance for testing."""
        server = ITSHttpServer(port=18003)
        yield server
        if server.is_running():
            server.stop()

    @pytest.fixture
    def client(self, http_server):
        """Create FastAPI test client."""
        return TestClient(http_server._app)

    def test_openapi_schema_generated(self, http_server):
        """Test that OpenAPI schema is generated."""
        # Access the app's openapi schema
        schema = http_server._app.openapi()
        assert "openapi" in schema
        assert schema["openapi"].startswith("3.")

    def test_api_info(self, http_server):
        """Test API info in OpenAPI schema."""
        schema = http_server._app.openapi()
        assert "info" in schema
        assert schema["info"]["title"] == "ITS Executor API"
        assert schema["info"]["version"] == "1.0.0"

    def test_endpoints_in_schema(self, http_server):
        """Test that all endpoints are in OpenAPI schema."""
        schema = http_server._app.openapi()
        paths = schema.get("paths", {})
        assert "/health" in paths
        assert "/api/v1/executor/status" in paths
        assert "/api/v1/executor/strategy" in paths
        assert "/api/v1/executor/deploy" in paths

    def test_deploy_endpoint_method(self, http_server):
        """Test that deploy endpoint is POST."""
        schema = http_server._app.openapi()
        paths = schema.get("paths", {})
        deploy_spec = paths.get("/api/v1/executor/deploy", {})
        assert "post" in deploy_spec

    def test_health_endpoint_method(self, http_server):
        """Test that health endpoint is GET."""
        schema = http_server._app.openapi()
        paths = schema.get("paths", {})
        health_spec = paths.get("/health", {})
        assert "get" in health_spec

    def test_multiple_requests_concurrent(self, http_server, client):
        """Test handling multiple concurrent requests."""
        # Send multiple requests concurrently
        import concurrent.futures

        def make_request():
            return client.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in futures]

        # All should succeed
        for response in results:
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])