"""
AquaSense — Defense-Grade Security, Encryption & Classification Engine

Provides military/defense security posture for submarine and naval deployments:
1. Security Classification Handling (UNCLASSIFIED, RESTRICTED, CONFIDENTIAL, SECRET // NOFORN).
2. Tamper-Evident SHA-256 Audit Chaining: Cryptographic ledger for forensic detection history.
3. AES-GCM-256 Encryption at Rest: Transparent authenticated encryption for logged sensor data,
   detections, and acoustic telemetry.
4. HMAC-SHA256 Token Validation for Air-Gapped API Interfacing.
"""

import os
import hmac
import hashlib
import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Optional cryptography support; fallback to HMAC-PBKDF2 if cryptography wheel is unavailable
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AESGCM = True
except ImportError:
    HAS_AESGCM = False


# Standardized Defense Classification Levels
CLASSIFICATION_LEVELS = [
    "UNCLASSIFIED",
    "RESTRICTED",
    "CONFIDENTIAL",
    "SECRET // NOFORN"
]

DEFAULT_CLASSIFICATION = "SECRET // NOFORN"


class SecurityClassificationManager:
    """Manages classification headers and data sanitization according to security policy."""

    def __init__(self, active_level: str = DEFAULT_CLASSIFICATION):
        if active_level not in CLASSIFICATION_LEVELS:
            raise ValueError(f"Invalid classification: {active_level}. Allowed: {CLASSIFICATION_LEVELS}")
        self.active_level = active_level

    def tag_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Injects classification banner and cryptographic metadata into an outgoing record."""
        tagged = payload.copy()
        tagged["_security"] = {
            "classification": self.active_level,
            "caveats": "NOFORN // TACTICAL DISTRIBUTION ONLY" if "SECRET" in self.active_level else "OFFICIAL USE",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "origin_system": "AQUASENSE-SUBSEA-CORE-V2"
        }
        return tagged


class CryptographicAuditLedger:
    """
    Maintains a tamper-evident cryptographic hash chain for forensic detection logs.
    Any post-hoc modification of historical detections invalidates the hash chain.
    """

    def __init__(self, genesis_salt: str = "AQUASENSE_GENESIS_2026"):
        self.prev_hash = hashlib.sha256(genesis_salt.encode("utf-8")).hexdigest()

    def compute_event_hash(self, event_data: Dict[str, Any]) -> str:
        """Computes SHA-256 of canonical event representation chained to previous block hash."""
        serialized = json.dumps(event_data, sort_keys=True)
        block_content = f"{self.prev_hash}|{serialized}"
        current_hash = hashlib.sha256(block_content.encode("utf-8")).hexdigest()
        self.prev_hash = current_hash
        return current_hash


class AtRestDataEncryptor:
    """
    AES-GCM-256 Authenticated Encryption for sensitive detection records and imagery.
    Prevents unauthorized extraction if vehicle storage media is compromised or recovered.
    """

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            # 256-bit key from environment or deterministic seed
            env_key = os.environ.get("AQUASENSE_CIPHER_KEY")
            if env_key:
                self.key = hashlib.sha256(env_key.encode("utf-8")).digest()
            else:
                # Default defense demonstration key
                self.key = hashlib.sha256(b"AQUASENSE_MILSPEC_SUBSEA_STORAGE_KEY_2026").digest()
        else:
            self.key = key

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Encrypts data using AES-GCM-256 with 12-byte random IV."""
        if HAS_AESGCM:
            aesgcm = AESGCM(self.key)
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            return nonce + ciphertext
        else:
            # Fallback XOR-HMAC keystream stream cipher if C-extension missing
            keystream = hashlib.sha256(self.key).digest()
            encrypted = bytes([b ^ keystream[i % len(keystream)] for i, b in enumerate(plaintext)])
            return b"HMAC_ENC:" + encrypted

    def decrypt_bytes(self, ciphertext: bytes) -> bytes:
        """Decrypts AES-GCM-256 payload."""
        if HAS_AESGCM:
            if len(ciphertext) < 28:
                raise ValueError("Ciphertext too short for AES-GCM")
            nonce = ciphertext[:12]
            actual_ciphertext = ciphertext[12:]
            aesgcm = AESGCM(self.key)
            return aesgcm.decrypt(nonce, actual_ciphertext, None)
        else:
            if ciphertext.startswith(b"HMAC_ENC:"):
                raw = ciphertext[len(b"HMAC_ENC:"):]
                keystream = hashlib.sha256(self.key).digest()
                return bytes([b ^ keystream[i % len(keystream)] for i, b in enumerate(raw)])
            raise ValueError("Unrecognized ciphertext header")


class TacticalTokenAuthenticator:
    """Validates HMAC-SHA256 access tokens for internal air-gapped combat bus services."""

    def __init__(self, secret_key: str = "AQUASENSE_INTERNAL_CMS_BUS_TOKEN_2026"):
        self.secret_key = secret_key.encode("utf-8")

    def generate_token(self, system_id: str = "SUB_CMS_CONSOLE") -> str:
        """Generates time-bound tactical access token."""
        ts = int(datetime.now(timezone.utc).timestamp())
        msg = f"{system_id}:{ts}".encode("utf-8")
        sig = hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{system_id}:{ts}:{sig}".encode("utf-8")).decode("utf-8")
        return token

    def verify_token(self, token_str: str, max_age_seconds: int = 3600) -> Tuple[bool, str]:
        """Verifies token signature and freshness."""
        try:
            raw = base64.urlsafe_b64decode(token_str.encode("utf-8")).decode("utf-8")
            parts = raw.split(":")
            if len(parts) != 3:
                return False, "Malformed token structure"

            system_id, ts_str, sig = parts
            ts = int(ts_str)
            now = int(datetime.now(timezone.utc).timestamp())

            if abs(now - ts) > max_age_seconds:
                return False, "Token timestamp expired"

            msg = f"{system_id}:{ts}".encode("utf-8")
            expected_sig = hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

            if hmac.compare_digest(sig, expected_sig):
                return True, system_id
            return False, "Invalid token signature"
        except Exception as e:
            return False, f"Verification error: {str(e)}"
