"""
License verification plugin for vLLM.

This plugin intercepts vLLM startup and performs license verification.
If verification fails, the startup process will be aborted.
"""
import os
import signal
import sys
import threading
import time
import logging

# Plugin metadata
__version__ = "1.0.0"

# License check interval: 1 hour
LICENSE_CHECK_INTERVAL = 3600

# Expiry warning interval: 24 hours before expiry
LICENSE_EXPIRY_WARNING_THRESHOLD = 86400


class LicenseVerifyError(Exception):
    """License verification failed."""
    pass



def setup_fallback_logging():
    """设置 fallback 日志配置，确保在 vLLM 启动后仍能工作"""
    # 重新配置根 logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ],
        force=True  # Python 3.8+ 强制重新配置
    )

    # 重新配置你的 logger
    logger = logging.getLogger("vllm_custom_plugins.license_verify")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 添加处理器
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
logger = setup_fallback_logging()


def check_license() -> bool:
    from .license_validator import LicenseValidator
    validator = LicenseValidator()
    status = validator.validate()
    if status == LicenseValidator.VALID:
        logger.info("License validation passed.")
    elif status == LicenseValidator.LICENSE_INVALID:
        logger.error("License invalid.")
        return False
    elif status == LicenseValidator.ESN_MISMATCH:
        logger.error("License ESN mismatch.")
        return False
    elif status == LicenseValidator.EXPIRED:
        logger.error("License expired,please apply for a new license.")
        return False
    elif status == LicenseValidator.EXPIRED_BUT_CONTINUE:
        logger.warning("Service will continue running but license is invalid.")
    elif status == LicenseValidator.SIGNATURE_INVALID:
        logger.error("License signature verification failed.")
        return False
    return True


def verify_license() -> bool:
    """
    Verify the vLLM license.

    This function performs license verification during vLLM startup.
    If verification fails, a LicenseVerifyError will be raised.

    Raises:
        LicenseVerifyError: If license verification fails.

    Returns:
        True if license verification succeeds.
    """

    if check_license():
        return True
    logger.error("License verification failed. vllm engine start up failed.")
    raise LicenseVerifyError("License check failed. vllm engine start up failed.")


def _license_monitor():
    """
    Background thread to monitor license status periodically.
    Checks license every LICENSE_CHECK_INTERVAL seconds.
    """

    logger.info(f"License monitor started (check every {LICENSE_CHECK_INTERVAL}s)")

    while True:
        time.sleep(LICENSE_CHECK_INTERVAL)

        try:
            if not check_license():
                logger.error("=" * 60)
                logger.error("LICENSE VERIFICATION FAILED: Periodic license check failed.")
                logger.error("Terminating inference process.")
                logger.error("=" * 60)
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logger.error(f"License check error: {e}")
            os.kill(os.getpid(), signal.SIGTERM)


def register(manager):
    """
    Register the license verification plugin.

    This function is called during vLLM plugin initialization.
    It performs license verification before any patches are applied.
    """

    logger.info("Running license verification...")

    try:
        verify_license()
        logger.info("License verification passed.")
    except LicenseVerifyError as e:
        logger.error("=" * 60)
        logger.error(f"LICENSE VERIFICATION FAILED: {e}")
        logger.error("Please apply for a license: contact your vendor.")
        logger.error("=" * 60)
        # Raise without traceback to avoid leaking verification logic
        raise LicenseVerifyError(str(e)) from None

    # Start background license monitor thread
    monitor_thread = threading.Thread(target=_license_monitor, daemon=True)
    monitor_thread.start()
    logger.info("License monitor thread started.")