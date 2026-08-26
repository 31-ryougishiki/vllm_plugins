import os
from datetime import datetime, timedelta
from vllm_custom_plugins import logger


LICENSE_PATH = os.getenv("LICENSE_PATH")
CERT_PATH = os.getenv("CERT_PATH")
PRODUCT_KEY_PATH = os.getenv("PRODUCT_KEY_PATH")


class LicenseValidator:
    LICENSE_INVALID = "LICENSE_INVALID"
    ESN_MISMATCH = "ESN_MISMATCH"
    EXPIRED = "EXPIRED"
    EXPIRED_BUT_CONTINUE = "EXPIRED_BUT_CONTINUE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    VALID = "VALID"

    CONTROL_ITEM_NAME = "inferenceEngine=1"
    SERVICE_TEST = "DEMO"
    SERVICE_CONTRACT_DELIVERY = "COMM"
    COOLING_PERIOD = 60

    def __init__(self):
        self.license_path = LICENSE_PATH
        self.cert_path = CERT_PATH
        self.product_key_path = PRODUCT_KEY_PATH
        from .license_reader import LicenseReader
        from .esn_checker import EsnChecker
        from .signature_verifier import SignatureVerifier
        self.reader = LicenseReader(self.license_path)
        self.esn_checker = EsnChecker()
        self.sig_verifier = SignatureVerifier(self.cert_path, self.product_key_path)

    def validate(self):
        if not self.reader.read():
            logger.error("License file read failed.")
            return self.LICENSE_INVALID
        logger.info("License file read successfully.")
        if not self._verify_signatures():
            logger.error("License signature verification failed.")
            return self.SIGNATURE_INVALID
        logger.info("License signature verification passed.")
        feature = self.reader.feature
        if not self._check_feature_valid(feature):
            logger.error("License control item name invalid.")
            return self.LICENSE_INVALID
        logger.info("License feature validation passed.")
        service = self.reader.service
        if not self._check_esn_match(service):
            machine_esn = self.esn_checker.get_machine_esn()
            logger.error(f"License hardware ESN mismatch: machine ESN={machine_esn}.")
            return self.ESN_MISMATCH
        logger.info("License ESN validation passed.")
        is_expiry, scenario, cooling_days_remaining = self._check_expiry_scenario(feature)
        if scenario not in [self.SERVICE_TEST, self.SERVICE_CONTRACT_DELIVERY]:
            logger.error(f"The scenario: {scenario} is not supported.")
            return self.LICENSE_INVALID
        if is_expiry:
            if cooling_days_remaining is not None:
                if scenario == self.SERVICE_TEST:
                    logger.error(f"License expired.")
                    return self.EXPIRED
                elif scenario == self.SERVICE_CONTRACT_DELIVERY:
                    logger.warning("License expired.")
                    return self.EXPIRED_BUT_CONTINUE
            return self.LICENSE_INVALID
        return self.VALID

    def _verify_signatures(self) -> bool:
        if not self.sig_verifier.verify_header_signature(self.license_path, self.reader.header):
            logger.error("Header signature verification failed.")
            return False
        if not self.sig_verifier.verify_service_signature(self.reader.service):
            logger.error("Service signature verification failed.")
            return False
        if not self.sig_verifier.verify_feature_signature(self.reader.feature):
            logger.error("Feature signature verification failed.")
            return False
        return True

    def _check_feature_valid(self, feature: dict) -> bool:
        function_field = feature.get("Function", "").strip('"')
        return self.CONTROL_ITEM_NAME in function_field

    def _check_esn_match(self, service: dict) -> bool:
        license_esn = service.get("Esn", "").strip('"')
        machine_esn = self.esn_checker.get_machine_esn()
        if not machine_esn:
            logger.warning("Could not obtain machine ESN.")
            return False
        if license_esn:
            esn_list = [esn.strip() for esn in license_esn.split(";") if esn.strip()]
        else:
            return False
        return machine_esn in esn_list

    def _check_expiry_scenario(self, feature: dict) -> tuple:
        attrib = feature.get("Attrib", "").strip('"')
        parts = attrib.split(",")
        if len(parts) < 2:
            logger.warning("Attrib does not contain expiry date.")
            return True, None, None
        expiry_str = parts[1].strip()
        scenario = parts[0].strip()
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            current_date = datetime.now().date()

            # 过期检查逻辑
            if current_date > expiry_date:
                # 计算冷却期剩余天数
                cooling_end_date = expiry_date + timedelta(days=self.COOLING_PERIOD)
                cooling_days_remaining = (cooling_end_date - current_date).days

                if cooling_days_remaining > 0:
                    # 冷却期内：提醒到期，返回剩余冷却天数
                    logger.warning(f"License will expire soon, remaining {cooling_days_remaining} days.")
                    return False, scenario, cooling_days_remaining
                else:
                    # 超过冷却期：正式过期
                    return True, scenario, 0
            else:
                # 未过期
                return False, scenario, None
        except ValueError:
            logger.warning(f"Invalid expiry date format: {expiry_str}.")
            return True, scenario, None

