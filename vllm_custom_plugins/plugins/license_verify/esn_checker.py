import subprocess
import platform
from vllm_custom_plugins import logger


class EsnChecker:
    @staticmethod
    def get_machine_esn() -> str:
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ['sudo', "cat", "/sys/class/dmi/id/product_serial"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    esn = result.stdout.strip()
                    if esn:
                        return esn
        except Exception as e:
            logger.warning(f"Failed to get ESN: {e}")
        return ""
