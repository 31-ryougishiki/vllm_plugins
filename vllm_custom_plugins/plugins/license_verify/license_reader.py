import os
from vllm_custom_plugins import logger


class LicenseReader:
    def __init__(self, license_path: str):
        self.license_path = license_path
        self._header: dict = {}
        self._service: dict = {}
        self._feature: dict = {}
        self._copyright: dict = {}

    def read(self) -> bool:
        if not os.path.exists(self.license_path):
            logger.error(f"The path specified by the environment variable LICENSE_PATH is incorrect. Please check.")
            return False
        try:
            with open(self.license_path, "r", encoding="latin-1") as f:
                content = f.read()
            sections = content.strip().split("\n\n")
            if len(sections) < 4:
                logger.error("Invalid license file format: expected 4 sections")
                return False
            self._copyright = self._parse_section(sections[0])
            self._header = self._parse_section(sections[1])
            self._service = self._parse_section(sections[2])
            self._feature = self._parse_section(sections[3])
            return True
        except Exception as e:
            logger.error(f"Failed to read license file: {e}")
            return False

    def _parse_section(self, section: str) -> dict:
        result = {}
        for line in section.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
        return result

    @property
    def header(self) -> dict:
        return self._header

    @property
    def service(self) -> dict:
        return self._service

    @property
    def feature(self) -> dict:
        return self._feature
