import importlib
import os
import sys
import logging
from typing import Dict, List

# Use fixed logger name for consistent configuration

def _setup_logger():
    """Configure vLLM Custom Plugins logging system."""
    _logger = logging.getLogger("vllm_custom_plugins")

    # Check if already configured (avoid duplicate handlers)
    if _logger.handlers:
        return _logger

    # Default to DEBUG level
    log_level = os.environ.get('VLLM_PATCH_LOG_LEVEL', 'DEBUG')
    log_level = getattr(logging, log_level.upper(), logging.DEBUG)

    _logger.setLevel(log_level)

    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # Format output
    formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    _logger.addHandler(handler)
    _logger.propagate = False  # Avoid duplicate logs

    return _logger


logger = _setup_logger()


class PatchManager:
    """Manage vLLM patch registration and application."""

    def __init__(self):
        self.available_patches: Dict[str, type] = {}
        self.applied_patches: List[str] = []

    def register(self, name: str, patch_class: type):
        """Register a patch for later application."""
        self.available_patches[name] = patch_class
        logger.info(f"Registered patch: {name}")

    def apply_patch(self, name: str) -> bool:
        """Apply a single patch by name."""
        if name not in self.available_patches:
            logger.error(f"Unknown patch: {name}")
            return False

        try:
            self.available_patches[name].apply()
            self.applied_patches.append(name)
            return True
        except Exception as e:
            logger.exception(f"Failed to apply patch {name}:")
            return False

    def apply_from_env(self):
        """
        Apply patches specified in VLLM_CUSTOM_PATCHES env var.

        Format: VLLM_CUSTOM_PATCHES="PatchOne,PatchTwo"
        """
        # 获取环境变量
        env_patches = os.environ.get('VLLM_CUSTOM_PATCHES', '').strip()

        if not env_patches:
            logger.info("No custom patches specified (VLLM_CUSTOM_PATCHES not set or empty)")
            return

        patch_names = [p.strip() for p in env_patches.split(',') if p.strip()]
        logger.info(f"Applying patches: {patch_names}")

        for name in patch_names:
            success = self.apply_patch(name)
            # zero_interrupt patches are load-bearing for the DeepSeek-V4 /
            # 0829 zero-interruption scenarios: its own apply() deliberately
            # raises RuntimeError to fail closed.  Swallowing that here would
            # let vLLM start with a half-patched executor/model path and fail
            # much later (wrong weights, dead strategy handling) instead of at
            # startup, so abort immediately for this plugin.
            if name == "zero_interrupt" and not success:
                raise RuntimeError(
                    "Failed to apply required patch 'zero_interrupt'; "
                    "refusing to continue with a half-patched service."
                )

        logger.info(f"Successfully applied: {self.applied_patches}")


# Global manager instance
manager = PatchManager()


def register_patches():
    """
    Main entry point called by vLLM plugin system.
    This function is automatically called on vLLM startup.
    """
    logger.info("=" * 60)
    logger.info("Initializing vLLM Custom Plugins")
    logger.info("=" * 60)

    # Import and register all available plugins
    from vllm_custom_plugins.plugins import PLUGINS

    for plugin_name in PLUGINS:
        plugin_module = f'vllm_custom_plugins.plugins.{plugin_name}'
        try:
            module = importlib.import_module(plugin_module)
            module.register(manager)
            logger.info(f"Loaded plugin: {plugin_name}")
        except Exception as e:  # Catch all exceptions
            # license_verify failures must propagate to abort startup
            if plugin_name == 'license_verify':
                logger.error(f"Critical error in {plugin_name}: {e}")
                # Exit without traceback to avoid leaking verification logic
                sys.exit(1)
            logger.warning(f"Failed to load plugin {plugin_name}: {e}")

    # Apply patches based on environment config
    manager.apply_from_env()

    logger.info("=" * 60)