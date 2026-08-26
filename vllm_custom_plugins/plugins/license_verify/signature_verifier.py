import traceback

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
from vllm_custom_plugins import logger


class SignatureVerifier:
    def __init__(self, cert_path: str, product_key_path: str):
        self.cert_path = cert_path
        self.product_key_path = product_key_path
        self._root_public_key = None
        self._product_public_key = None

    def _load_x509_cert(self):
        with open(self.cert_path, "r", encoding="utf-8") as f:
            cert_pem = f.read()
        return load_pem_x509_certificate(cert_pem.encode("utf-8"), default_backend())

    def _load_root_public_key(self):
        if self._root_public_key is not None:
            return self._root_public_key
        try:
            cert = self._load_x509_cert()
            self._root_public_key = cert.public_key()
            return self._root_public_key
        except Exception as e:
            logger.error(f"Failed to load root certificate: {e}")
            return None

    def _load_product_public_key(self):
        if self._product_public_key is not None:
            return self._product_public_key
        try:
            with open(self.product_key_path, "r", encoding="utf-8") as f:
                product_key_data = f.read().strip()
            parts = product_key_data.split(":")
            if len(parts) < 4:
                logger.error("Invalid product key format")
                return None
            key_hex = parts[3]
            key_bytes = bytes.fromhex(key_hex)
            self._product_public_key = serialization.load_der_public_key(
                key_bytes,
                backend=default_backend()
            )
            # 确保返回的是RSA公钥
            if isinstance(self._product_public_key, rsa.RSAPublicKey):
                return self._product_public_key
            else:
                logger.error("Product key is not an RSA public key")
                return None
        except Exception as e:
            logger.error(f"Failed to load product key: {e}")
            return None

    def _dict_to_str(self, data: dict) -> str:
        lines = []
        for key, value in data.items():
            if key != "Sign":
                lines.append(f"{key}={value}")
        return "\n".join(lines)

    def verify_header_signature(self, license_path: str, header_data: dict) -> bool:
        sign_value = header_data.get("Sign", "")
        if not sign_value:
            logger.error("No signature found in header")
            return False
        try:
            public_key = self._load_root_public_key()
            if public_key is None:
                return False
            signature = bytes.fromhex(sign_value)
            with open(license_path, "r", encoding="latin-1") as f:
                content = f.read()
            sections = content.split("\n\n")
            processed_lines = []
            for section in sections:
                section_lines = []
                for line in section.split("\n"):
                    if line.startswith("Sign="):
                        continue
                    section_lines.append(line)
                if section_lines:
                    processed_lines.append("\n".join(section_lines))
            original_text = ("\n\n\n".join(processed_lines) + "\n\n").encode("latin-1").rstrip(b"\n")
            public_key.verify(
                signature,
                original_text,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=32
                ),
                hashes.SHA256()
            )
            logger.info("Header signature verified successfully")
            return True
        except Exception as e:
            logger.error(f"Header signature verification failed: {e} \n{traceback.format_exc()}")
            return False

    def verify_service_signature(self, service_data: dict) -> bool:
        sign_value = service_data.get("Sign", "")
        if not sign_value:
            logger.error("No signature found in service section")
            return False
        try:
            public_key = self._load_product_public_key()
            if public_key is None:
                return False
            signature = bytes.fromhex(sign_value)
            original_text = self._dict_to_str(service_data).encode("latin-1").rstrip(b"\n")
            public_key.verify(
                signature,
                original_text,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=32
                ),
                hashes.SHA256()
            )
            logger.info("Service signature verified successfully")
            return True
        except Exception as e:
            logger.error(f"Service signature verification failed: {e}")
            return False

    def verify_feature_signature(self, feature_data: dict) -> bool:
        sign_value = feature_data.get("Sign", "")
        if not sign_value:
            logger.error("No signature found in feature section")
            return False
        try:
            public_key = self._load_product_public_key()
            if public_key is None:
                return False
            signature = bytes.fromhex(sign_value)
            original_text = self._dict_to_str(feature_data).encode("latin-1").rstrip(b"\n")
            public_key.verify(
                signature,
                original_text,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=32
                ),
                hashes.SHA256()
            )
            logger.info("Feature signature verified successfully")
            return True
        except Exception as e:
            logger.error(f"Feature signature verification failed: {e}")
            return False

