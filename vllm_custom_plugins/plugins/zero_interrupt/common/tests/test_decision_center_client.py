#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Test cases for Decision Center Client."""

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Add paths
_plugin_dir = r"D:\code\python_code\fork\vLLM\plugins\vllm_custom_plugins\vllm_custom_plugins\plugins\zero_interrup"
_plugins_dir = r"D:\code\python_code\fork\vLLM\plugins\vllm_custom_plugins\vllm_custom_plugins\plugins"
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)
if _plugins_dir not in sys.path:
    sys.path.insert(0, _plugins_dir)

from zero_interrup.communication.decision_center_client import DecisionCenterClient
from zero_interrup.common.types import DeployState, EngineParallelConfig, InitExecutorStateRequest, ModelInfo


class TestDecisionCenterClientInit:
    """Test DecisionCenterClient initialization."""

    def test_client_init_default(self):
        """Test client initialization with defaults."""
        client = DecisionCenterClient()

        assert client.base_url == "http://127.0.0.1:8080"
        assert client.token == ""
        assert client.timeout == 30
        assert client.max_retries == 3

    def test_client_init_custom(self):
        """Test client initialization with custom values."""
        client = DecisionCenterClient(
            base_url="http://192.168.1.100:9090",
            token="test_token_123",
            timeout=60,
            max_retries=5,
        )

        assert client.base_url == "http://192.168.1.100:9090"
        assert client.token == "test_token_123"
        assert client.timeout == 60
        assert client.max_retries == 5

    def test_client_base_url_strip_trailing_slash(self):
        """Test that base URL trailing slash is stripped."""
        client = DecisionCenterClient(base_url="http://example.com/")
        assert client.base_url == "http://example.com"


class TestDecisionCenterClientHeaders:
    """Test DecisionCenterClient headers."""

    def test_headers_without_token(self):
        """Test headers without token."""
        client = DecisionCenterClient(token="")
        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_with_token(self):
        """Test headers with token."""
        client = DecisionCenterClient(token="my_token")
        headers = client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer my_token"


class TestDecisionCenterClientReportInitState:
    """Test report_init_state method."""

    @patch("requests.post")
    def test_report_init_state_success(self, mock_post):
        """Test successful init state reporting."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = DecisionCenterClient(base_url="http://test.com")

        request = InitExecutorStateRequest(
            service_id="test_service",
            model_name="test_model",
            engine_id="0",
            engine_parallel_config=EngineParallelConfig(dp=2, tp=4),
            model_info=ModelInfo(),
            engine_pd_role="PD_MIX",
            executor_state="RUNNING",
            executor_ip_port="192.168.1.10:29500",
            data_parallel_ip_port="192.168.1.10:29501",
            data_parallel_rank=0,
            node_ip="192.168.1.10",
            node_hbm=68828198400,
            npu_id=["0", "1", "2", "3"],
            npu_rank_id=["0", "1", "2", "3"],
            npu_healthy=[True, True, True, True],
        )

        result = client.report_init_state(request)

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_report_init_state_failure(self, mock_post):
        """Test failed init state reporting."""
        import requests as req
        mock_post.side_effect = req.RequestException("Connection error")

        client = DecisionCenterClient(base_url="http://test.com")

        request = InitExecutorStateRequest(
            service_id="test_service",
            model_name="test_model",
            engine_id="0",
            engine_parallel_config=EngineParallelConfig(dp=2, tp=4),
            model_info=ModelInfo(),
            engine_pd_role="PD_MIX",
            executor_state="RUNNING",
            executor_ip_port="192.168.1.10:29500",
            data_parallel_ip_port="192.168.1.10:29501",
            data_parallel_rank=0,
            node_ip="192.168.1.10",
            node_hbm=68828198400,
            npu_id=["0", "1", "2", "3"],
            npu_rank_id=["0", "1", "2", "3"],
            npu_healthy=[True, True, True, True],
        )

        result = client.report_init_state(request)

        assert result is False


class TestDecisionCenterClientReportDeployStatus:
    """Test report_deploy_status method."""

    @patch("requests.post")
    def test_report_deploy_status_success(self, mock_post):
        """Test successful deploy status reporting."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = DecisionCenterClient(base_url="http://test.com")

        result = client.report_deploy_status("0", DeployState.EXECUTOR_DEPLOY_SUCCESS)

        assert result is True
        mock_post.assert_called_once()

        # Verify the payload
        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["executor_id"] == "0"
        assert call_args.kwargs["json"]["deploy_state"] == "EXECUTOR_DEPLOY_SUCCESS"

    @patch("requests.post")
    def test_report_deploy_status_all_states(self, mock_post):
        """Test all deploy states."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = DecisionCenterClient(base_url="http://test.com")

        states = [
            DeployState.EXECUTOR_DEPLOY_SUCCESS,
            DeployState.EXECUTOR_STOP,
            DeployState.EXECUTOR_DEPLOY_FAIL,
        ]

        for state in states:
            result = client.report_deploy_status("0", state)
            assert result is True

        assert mock_post.call_count == 3

    @patch("requests.post")
    def test_report_deploy_status_failure(self, mock_post):
        """Test failed deploy status reporting."""
        import requests as req
        mock_post.side_effect = req.RequestException("Connection error")

        client = DecisionCenterClient(base_url="http://test.com")

        result = client.report_deploy_status("0", DeployState.EXECUTOR_DEPLOY_FAIL)

        assert result is False


class TestDecisionCenterClientMakeRequest:
    """Test _make_request method."""

    @patch("requests.get")
    def test_make_request_get(self, mock_get):
        """Test GET request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = DecisionCenterClient(base_url="http://test.com")

        result = client._make_request("GET", "/api/test", {"key": "value"})

        assert result == {"data": "test"}
        mock_get.assert_called_once()

    @patch("requests.post")
    def test_make_request_post(self, mock_post):
        """Test POST request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = DecisionCenterClient(base_url="http://test.com")

        result = client._make_request("POST", "/api/test", {"key": "value"})

        assert result == {"status": "ok"}
        mock_post.assert_called_once()

    def test_make_request_invalid_method(self):
        """Test invalid HTTP method."""
        client = DecisionCenterClient(base_url="http://test.com")

        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            client._make_request("PUT", "/api/test", {})


class TestDecisionCenterClientUrl:
    """Test URL construction."""

    def test_url_construction(self):
        """Test URL is constructed correctly."""
        client = DecisionCenterClient(base_url="http://example.com:8080")

        # Test that URL is constructed from base_url + endpoint
        url = f"{client.base_url}/api/v1/test"
        assert url == "http://example.com:8080/api/v1/test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])