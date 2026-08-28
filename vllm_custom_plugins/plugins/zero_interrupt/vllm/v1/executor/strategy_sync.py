#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""Strategy Sync Thread for ITS plugin.

This module provides a simple callback mechanism for receiving deployment
strategies from the decision center via HTTP.

Note: The actual strategy receiving is done by HTTP server directly calling
the executor's callback. This class is kept for compatibility.
"""

from typing import Callable

from vllm.logger import logger
from vllm_custom_plugins.plugins.zero_interrupt.common.constants import (
    VLLM_ITS_STRATEGY_TIMEOUT,
)
from vllm_custom_plugins.plugins.zero_interrupt.common.types import DeployStrategy


class StrategySyncThread:
    """Thread for coordination of strategy receipt.

    This class acts as a bridge between HTTP server and executor.
    The HTTP server calls on_strategy_received() which triggers the callback.
    """

    def __init__(
        self,
        strategy_callback: Callable[[DeployStrategy], None],
        timeout: int = VLLM_ITS_STRATEGY_TIMEOUT,
    ):
        """Initialize the strategy sync thread.

        Args:
            strategy_callback: Callback when strategy is received
            timeout: Strategy execution timeout in seconds
        """
        self.strategy_callback = strategy_callback
        self.timeout = timeout

        self._running = False
        self._current_strategy: DeployStrategy | None = None

    def start(self) -> None:
        """Start the strategy sync thread."""
        self._running = True
        logger.info("Strategy sync thread initialized")

    def on_strategy_received(self, strategy: DeployStrategy) -> None:
        """Called by HTTP server when a deployment strategy is received.

        This directly triggers the callback to process the strategy.

        Args:
            strategy: The deployment strategy received from decision center
        """
        logger.info(f"Strategy received via HTTP: {strategy.deploy_type.value}")

        if self._current_strategy == strategy:
            # A previous attempt may have recorded this exact strategy but
            # failed to execute (or only some DPs received it).  Returning
            # silently here makes the HTTP endpoint reply 200 while the
            # executor does nothing, which is indistinguishable from a lost
            # request.  Forward it again: the caller is explicit about the
            # deployment intent, and the executor strategy path is the
            # idempotency boundary.
            logger.warning(
                "Duplicate deployment strategy received; forwarding it to "
                "the executor again instead of dropping it silently."
            )
        else:
            self._current_strategy = strategy

        # Directly invoke callback (no need for separate thread)
        if self.strategy_callback:
            try:
                self.strategy_callback(strategy)
            except Exception as e:
                logger.error(f"Error in strategy callback: {e}")

    def stop(self) -> None:
        """Stop the strategy sync thread."""
        self._running = False
        logger.info("Strategy sync thread stopped")

    def is_running(self) -> bool:
        """Check if the thread is running."""
        return self._running

    def get_current_strategy(self) -> DeployStrategy | None:
        """Get the current strategy."""
        return self._current_strategy