#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""ITS 插件决策中心客户端。

本模块提供与决策中心的通信能力，用于上报执行器状态和接收部署策略。
"""

import time
from dataclasses import asdict
from typing import Any

import requests

from vllm.logger import logger
from vllm_custom_plugins.plugins.zero_interrupt.common.constants import (
    API_INIT_EXECUTOR_STATE,
    API_REPORT_DEPLOY_STATUS,
    VLLM_ITS_DECISION_CENTER_TOKEN,
    VLLM_ITS_DECISION_CENTER_URL,
    VLLM_ITS_MAX_RETRY_COUNT,
)
from vllm_custom_plugins.plugins.zero_interrupt.common.types import DeployState, InitExecutorStateRequest, UpdateEngineInfo


class DecisionCenterClient:
    """与决策中心通信的客户端。

    此客户端处理：
    - 上报初始执行器状态
    - 上报部署策略执行结果
    - 轮询部署策略
    """

    def __init__(
            self,
            base_url: str = VLLM_ITS_DECISION_CENTER_URL,
            token: str = VLLM_ITS_DECISION_CENTER_TOKEN,
            timeout: int = 30,
            max_retries: int = VLLM_ITS_MAX_RETRY_COUNT,
    ):
        """初始化决策中心客户端。

        Args:
            base_url: 决策中心基础 URL
            token: 认证令牌
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_headers(self) -> dict[str, str]:
        """获取带认证的请求头。

        Returns:
            HTTP 头部字典
        """
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _make_request(
            self,
            method: str,
            endpoint: str,
            data: dict[str, Any] | None = None,
            retries: int | None = None,
    ) -> dict[str, Any]:
        """带重试逻辑的 HTTP 请求。

        Args:
            method: HTTP 方法（GET、POST 等）
            endpoint: API 端点
            data: 请求数据
            retries: 当前重试次数

        Returns:
            响应数据字典

        Raises:
            requests.RequestException: 如果所有重试都失败则抛出异常
        """
        url = f"{self.base_url}{endpoint}"
        retries = retries or 0

        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                    params=data,
                )
            elif method.upper() == "POST":
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=self.timeout,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            if retries < self.max_retries:
                wait_time = 2 ** retries
                logger.warning(
                    f"Request failed (retry {retries + 1}/{self.max_retries}): {e}. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                return self._make_request(method, endpoint, data, retries + 1)
            logger.error(f"Request to {url} failed after {self.max_retries} retries: {e}")
            raise

    def _post_init_executor_state(
        self,
        init_executor_state_request: InitExecutorStateRequest,
    ) -> dict[str, Any] | None:
        """POST /init_executor_state and return the response body.

        ``None`` means the request failed (the exception is already logged).
        """
        request_data = asdict(init_executor_state_request)
        logger.debug(f"Request payload: {request_data}")
        try:
            return self._make_request(
                "POST",
                API_INIT_EXECUTOR_STATE,
                request_data,
            )
        except requests.RequestException as e:
            logger.error(
                "Failed to report init state to decision center: %s",
                e,
                exc_info=True,
            )
            return None

    def report_init_state(
            self,
            init_executor_state_request: InitExecutorStateRequest,
    ) -> bool:
        """上报初始执行器状态到决策中心。

        上报内容：
        - service_id: 服务实例唯一标识
        - engine_id: KV Cache Engine ID
        - engine_parallel_config: 并行配置（dp/tp/enable_expert_parallel）
        - engine_pd_role: PD 角色
        - executor_state: 执行器状态（RUNNING）
        - data_parallel_ip_port: data_parallel地址
        - data_parallel_rank: DP 组内 rank
        - node_ip: 节点 IP
        - node_hbm: 节点 HBM 总容量（Byte），用于决策中心估算 HBM 使用率
        - npu_id: NPU 物理 ID 列表
        - npu_rank_id: NPU 全局 rank 列表
        - npu_healthy: NPU 健康状态列表
        - model_info: HBM 估算所需模型信息
        - executor_ip_port: 执行器地址
        Args:
            init_executor_state_request: 初始化执行器状态请求对象

        Returns:
            bool: 上报是否成功（True 成功，False 失败）

        Raises:
            无异常，上报失败返回 False 而非抛出异常
        """
        logger.info("Reporting init state to decision center")
        logger.debug(f"Init executor state request: {init_executor_state_request}")

        response = self._post_init_executor_state(init_executor_state_request)
        if response is None:
            return False

        logger.info("Successfully reported init state to decision center")
        logger.debug(f"Response from decision center: {response}")
        return True

    def report_init_state_with_executor_id(
        self,
        init_executor_state_request: InitExecutorStateRequest,
    ) -> str | None:
        """Report init state and return the executor_id assigned by the center.

        DecisionMakingCenter generates ``exe-<service_id>-<engine_uid>-<n>``
        and returns it in the JSON body of ``/init_executor_state``. Every
        strategy and deploy-status report must use this assigned id; the
        local numeric ``data_parallel_rank`` is not known to the center.

        Returns:
            The assigned executor id, or ``None`` when the request failed or
            the response has no executor_id.
        """
        response = self._post_init_executor_state(init_executor_state_request)
        if not response:
            return None
        executor_id = response.get("executor_id")
        if executor_id is None:
            logger.warning(
                "Decision center response has no executor_id: %s",
                response,
            )
            return None
        logger.info(
            "Decision center assigned executor_id=%s",
            executor_id,
        )
        return str(executor_id)

    def report_deploy_status(
            self,
            executor_id: str,
            deploy_state: DeployState,
            update_engine_info: UpdateEngineInfo | None = None
    ) -> bool:
        """上报部署状态到决策中心。

        Args:
            executor_id: Executor 标识符
            deploy_state: 部署状态
            update_engine_info: 更新引擎信息

        Returns:
            成功返回 True，否则返回 False
        """
        payload = {
            "executor_id": executor_id,
            "deploy_state": deploy_state.value,
        }

        if update_engine_info:
            payload["update_engine_info"] = asdict(update_engine_info)

        try:
            response = self._make_request("POST", API_REPORT_DEPLOY_STATUS, payload)
            logger.info(f"Reported deploy status: {response}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to report deploy status: {e}")
            return False
