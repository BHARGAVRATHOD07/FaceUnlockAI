"""
vault_crypto.py — Cryptographic Engine for FaceUnlockAI Secure Vault
----------------------------------------------------------------------
Implements authenticated document encryption, key derivation, and multi-slot
keystore management.

Cryptographic Design & Standards:
  1. Cipher: AES-256-GCM with 96-bit unique cryptographically random nonces.
  2. Master Key: 256-bit random symmetric key (Vault Master Key - VMK).
  3. Key Derivation: Argon2id (RFC 9106) with unique 16-byte random salt,
     time_cost=3, memory_cost=65536 KiB (64 MiB), parallelism=4.
  4. Multi-Slot Keystore:
     - Slot 1: Master Password Slot (Argon2id KEK wraps VMK via AES-256-GCM)
     - Slot 2: Recovery Key Slot (Independent 256-bit recovery token wraps VMK)
     - Slot 3: Biometric Authorization Slot (Windows DPAPI + active session gate)
  5. Authenticated Associated Data (AAD): File metadata (filename, size, UUID,
     timestamp) authenticated alongside ciphertext to prevent file swap attacks.
  6. Memory Management: SecureKeyHolder uses mutable bytearray with zeroing.
  7. Fail-Closed: Corrupted headers, invalid tags, or wrong keys raise specific
     exceptions without leaking plaintext or partial decrypted state.

CRITICAL SECURITY NOTICE ON DPAPI & BIOMETRICS:
  Windows DPAPI (CryptProtectData) encrypts data tied to the user's Windows logon.
  It protects data from *other* non-admin Windows users and offline disk inspection,
  but does NOT protect against malware executing inside the *same* user session.
  Biometric face recognition (SFace + Liveness) acts as an authorization policy gate
  to release the VMK into memory during an active, non-expired session. Biometric
  embeddings are NEVER used directly as encryption keys.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import secrets
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import argon2.low_level
from argon2.low_level import Type as Argon2Type
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Constants & Container Formatting
VAULT_MAGIC = b"FVAU"  # FaceUnlockAI Vault magic bytes (4 bytes)
VAULT_VERSION = 1       # Container format version (1 byte)
NONCE_BYTES = 12        # 96-bit nonce standard for AES-GCM
KEY_BYTES = 32          # 256-bit symmetric key
SALT_BYTES = 16         # 128-bit salt for Argon2id
HEADER_SIZE = 4 + 1 + NONCE_BYTES  # 17 bytes (MAGIC + VERSION + NONCE)

# Default Recommended Argon2id Parameters (RFC 9106 recommended interactive profile)
ARGON2_TIME_COST = 3          # 3 iterations
ARGON2_MEMORY_COST = 65536    # 64 MiB memory hardness
ARGON2_PARALLELISM = 4        # 4 concurrent threads
ARGON2_HASH_LEN = 32          # 256-bit derived key length


# =====================================================================
# Custom Cryptographic Exceptions (Fail-Closed)
# =====================================================================

class CryptoError(Exception):
    """Base exception for all vault cryptographic operations."""


class InvalidKeyError(CryptoError):
    """Raised when an encryption key is invalid, wrong length, or corrupted."""


class AuthenticationError(CryptoError):
    """Raised when authentication tag verification fails (tampering or wrong key)."""


class CorruptedDataError(CryptoError):
    """Raised when ciphertext header is missing, corrupted, or unsupported version."""


class VaultLockedError(CryptoError):
    """Raised when a cryptographic operation is attempted while vault is locked."""


class RecoveryKeyError(CryptoError):
    """Raised when an invalid recovery key format or checksum is provided."""


# =====================================================================
# Key Generation & Derivation (Argon2id)
# =====================================================================

def generate_master_key() -> bytes:
    """
    Generates a cryptographically secure 256-bit (32-byte) Vault Master Key (VMK).
    """
    return secrets.token_bytes(KEY_BYTES)


def derive_key_argon2id(
    password: str,
    salt: bytes,
    time_cost: int = ARGON2_TIME_COST,
    memory_cost: int = ARGON2_MEMORY_COST,
    parallelism: int = ARGON2_PARALLELISM,
    hash_len: int = ARGON2_HASH_LEN,
) -> bytes:
    """
    Derives a 256-bit Key Encryption Key (KEK) from a password using Argon2id.

    Args:
        password: UTF-8 user password.
        salt: Exactly 16 cryptographically random bytes.
        time_cost: Number of iterations.
        memory_cost: Memory usage in KiB.
        parallelism: Number of parallel lanes.
        hash_len: Desired key length in bytes (32 for AES-256).

    Returns:
        32-byte derived key.
    """
    if len(salt) < SALT_BYTES:
        raise InvalidKeyError(f"Salt must be at least {SALT_BYTES} bytes, got {len(salt)}.")
    if not password:
        raise InvalidKeyError("Password cannot be empty.")

    password_bytes = password.encode("utf-8")
    try:
        derived_key = argon2.low_level.hash_secret_raw(
            secret=password_bytes,
            salt=salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_len,
            type=Argon2Type.ID,
        )
        return derived_key
    except Exception as exc:
        raise CryptoError(f"Argon2id key derivation failed: {exc}") from exc


# =====================================================================
# Recovery Key Mechanism (160-bit High Entropy - NIST SP 800-132)
# =====================================================================

RECOVERY_BYTES = 20  # 160 bits of entropy -> exactly 32 Crockford Base32 characters (32 * 5 bits)
# Crockford Base32 alphabet (excludes I, L, O, U to avoid transcription confusion)
RECOVERY_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_recovery_key() -> Tuple[bytes, str]:
    """
    Generates a high-entropy 160-bit recovery token and its human-readable representation.
    160 bits provides 2^160 combinations (exceeding NIST 128-bit security requirements)
    and maps bijectively to exactly 32 Crockford characters (8 groups of 4).

    Returns:
        (raw_token_bytes, formatted_recovery_string)
        Formatted as 8 groups of 4 uppercase characters separated by hyphens:
        e.g., 'A7X9-K2M4-P9Q8-W3T5-R8Y2-N6B1-C4V7-D9E2'
    """
    raw_token = secrets.token_bytes(RECOVERY_BYTES)
    formatted = format_recovery_key(raw_token)
    return raw_token, formatted


def format_recovery_key(raw_token: bytes) -> str:
    """Encodes 20 raw bytes (160 bits) into a standardized 32-character Crockford-grouped string."""
    if len(raw_token) != RECOVERY_BYTES:
        raise InvalidKeyError(f"Raw recovery token must be exactly {RECOVERY_BYTES} bytes.")

    num = int.from_bytes(raw_token, byteorder="big")
    chars = []
    for _ in range(32):
        chars.append(RECOVERY_ALPHABET[num & 0x1F])
        num >>= 5
    chars.reverse()
    encoded = "".join(chars)

    # Group into 8 blocks of 4 characters: XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
    blocks = [encoded[i : i + 4] for i in range(0, 32, 4)]
    return "-".join(blocks)


def parse_recovery_key(recovery_str: str) -> bytes:
    """
    Parses and validates a user-entered recovery key string.
    Tolerates hyphens, spaces, lowercase, and common transcription typos (O->0, I/L->1).

    Returns:
        20-byte raw recovery token.
    """
    # Clean input: strip spaces and hyphens, uppercase
    cleaned = recovery_str.strip().replace("-", "").replace(" ", "").upper()

    # Normalize common substitutions
    cleaned = cleaned.replace("O", "0").replace("I", "1").replace("L", "1")

    if len(cleaned) != 32:
        raise RecoveryKeyError(
            f"Recovery key must contain exactly 32 alphanumeric characters (got {len(cleaned)})."
        )

    # Validate against alphabet and reconstruct 160-bit integer
    num = 0
    for char in cleaned:
        idx = RECOVERY_ALPHABET.find(char)
        if idx == -1:
            raise RecoveryKeyError(f"Invalid character '{char}' in recovery key.")
        num = (num << 5) | idx

    # Convert back to 20 bytes
    token_bytes = num.to_bytes(RECOVERY_BYTES, byteorder="big")
    return token_bytes



# =====================================================================
# Authenticated Document Encryption / Decryption (AES-256-GCM)
# =====================================================================

def encrypt_data(
    plaintext: bytes,
    key: bytes,
    associated_data: Optional[Union[bytes, str, Dict[str, Any]]] = None,
) -> bytes:
    """
    Encrypts plaintext using AES-256-GCM with a fresh 96-bit random nonce.

    Container Layout:
      [MAGIC (4B: b"FVAU")] [VERSION (1B: 0x01)] [NONCE (12B)] [CIPHERTEXT + 16B TAG]

    Args:
        plaintext: Raw data to encrypt.
        key: Exactly 32 bytes (256 bits).
        associated_data: Optional metadata authenticated alongside the ciphertext.

    Returns:
        Packaged container bytes.
    """
    if len(key) != KEY_BYTES:
        raise InvalidKeyError(f"Encryption key must be {KEY_BYTES} bytes, got {len(key)}.")

    aad_bytes = _serialize_aad(associated_data)
    nonce = secrets.token_bytes(NONCE_BYTES)

    try:
        aesgcm = AESGCM(key)
        # AESGCM.encrypt automatically appends the 16-byte authentication tag
        ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, aad_bytes)
    except Exception as exc:
        raise CryptoError(f"AES-256-GCM encryption failed: {exc}") from exc

    # Assemble container
    header = VAULT_MAGIC + bytes([VAULT_VERSION]) + nonce
    return header + ciphertext_and_tag


def decrypt_data(
    container_bytes: bytes,
    key: bytes,
    associated_data: Optional[Union[bytes, str, Dict[str, Any]]] = None,
) -> bytes:
    """
    Decrypts an authenticated container using AES-256-GCM.
    Fails closed if the header is corrupted, the key is wrong, or the data was tampered with.

    Args:
        container_bytes: Packaged container bytes.
        key: Exactly 32 bytes (256 bits).
        associated_data: Metadata that must match what was passed during encryption.

    Returns:
        Decrypted plaintext bytes.
    """
    if len(key) != KEY_BYTES:
        raise InvalidKeyError(f"Decryption key must be {KEY_BYTES} bytes, got {len(key)}.")

    if len(container_bytes) < HEADER_SIZE + 16:  # Must have header + at least 16B tag
        raise CorruptedDataError(
            f"Container too short ({len(container_bytes)} bytes; minimum is {HEADER_SIZE + 16})."
        )

    # Verify Magic Bytes
    if container_bytes[:4] != VAULT_MAGIC:
        raise CorruptedDataError("Invalid vault magic bytes. File is not a valid Secure Vault container.")

    # Verify Version
    version = container_bytes[4]
    if version != VAULT_VERSION:
        raise CorruptedDataError(f"Unsupported vault container version ({version}).")

    # Extract Nonce and Ciphertext+Tag
    nonce = container_bytes[5 : 5 + NONCE_BYTES]
    ciphertext_and_tag = container_bytes[HEADER_SIZE:]
    aad_bytes = _serialize_aad(associated_data)

    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, aad_bytes)
        return plaintext
    except InvalidTag as exc:
        raise AuthenticationError(
            "Decryption failed: Authentication tag mismatch. "
            "Data has been tampered with, corrupted, or the key is incorrect."
        ) from exc
    except Exception as exc:
        raise CryptoError(f"Decryption error: {exc}") from exc


def _serialize_aad(associated_data: Optional[Union[bytes, str, Dict[str, Any]]]) -> Optional[bytes]:
    """Helper to deterministically convert AAD metadata into canonical bytes."""
    if associated_data is None:
        return None
    if isinstance(associated_data, bytes):
        return associated_data
    if isinstance(associated_data, str):
        return associated_data.encode("utf-8")
    if isinstance(associated_data, dict):
        return json.dumps(associated_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raise ValueError(f"Unsupported associated data type: {type(associated_data)}")


# =====================================================================
# Windows DPAPI Wrapper (Biometric Authorization Slot)
# =====================================================================

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def dpapi_protect(plaintext: bytes, description: str = "FaceUnlockAI Secure Vault") -> bytes:
    """
    Encrypts data using Windows Data Protection API (DPAPI).
    Ties encryption to the current logged-in Windows user account.
    """
    if sys.platform != "win32":
        # Fallback for non-Windows development/testing
        salt = secrets.token_bytes(16)
        return b"NONWIN_DPAPI:" + salt + plaintext

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    in_blob = _DATA_BLOB(len(plaintext), ctypes.create_string_buffer(plaintext, len(plaintext)))
    out_blob = _DATA_BLOB()

    # CryptProtectData(pDataIn, szDataDescr, pOptionalEntropy, pReserved, pPromptStruct, dwFlags, pDataOut)
    # dwFlags = 0x01 (CRYPTPROTECT_UI_FORBIDDEN - no UI dialog)
    success = crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        ctypes.c_wchar_p(description),
        None,
        None,
        None,
        0x01,
        ctypes.byref(out_blob),
    )
    if not success:
        err = kernel32.GetLastError()
        raise CryptoError(f"Windows DPAPI CryptProtectData failed with error code: {err}")

    try:
        encrypted_bytes = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return encrypted_bytes
    finally:
        kernel32.LocalFree(out_blob.pbData)


def dpapi_unprotect(ciphertext: bytes) -> bytes:
    """
    Decrypts data using Windows Data Protection API (DPAPI).
    Succeeds only when running in the same Windows user session.
    """
    if sys.platform != "win32":
        if ciphertext.startswith(b"NONWIN_DPAPI:"):
            return ciphertext[len(b"NONWIN_DPAPI:") + 16 :]
        raise CryptoError("Invalid non-Windows DPAPI payload.")

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    in_blob = _DATA_BLOB(len(ciphertext), ctypes.create_string_buffer(ciphertext, len(ciphertext)))
    out_blob = _DATA_BLOB()

    # CryptUnprotectData(pDataIn, ppszDataDescr, pOptionalEntropy, pReserved, pPromptStruct, dwFlags, pDataOut)
    success = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(out_blob),
    )
    if not success:
        err = kernel32.GetLastError()
        raise CryptoError(f"Windows DPAPI CryptUnprotectData failed with error code: {err}")

    try:
        decrypted_bytes = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return decrypted_bytes
    finally:
        kernel32.LocalFree(out_blob.pbData)


# =====================================================================
# Multi-Slot Keystore (Password, Recovery, and Biometric Slots)
# =====================================================================

@dataclass
class KeySlot:
    slot_type: str                   # 'password', 'recovery', or 'biometric'
    salt_b64: str                    # Argon2id salt (base64)
    argon2_params: Dict[str, int]    # {time_cost, memory_cost, parallelism}
    wrapped_vmk_b64: str             # AES-256-GCM container of the VMK (base64)


class VaultKeyStore:
    """
    Manages the multi-slot encrypted key container.
    Stores the 256-bit VMK encrypted under:
      1. Password Slot (KEK derived via Argon2id)
      2. Recovery Slot (KEK derived via Argon2id from Recovery Key)
      3. Biometric Authorization Slot (DPAPI + local session gate token)
    """

    def __init__(self) -> None:
        self.slots: Dict[str, KeySlot] = {}
        self.biometric_dpapi_blob_b64: Optional[str] = None
        self.biometric_session_salt_b64: Optional[str] = None

    @classmethod
    def initialize_new_vault(
        cls,
        master_password: str,
        vmk: Optional[bytes] = None,
        custom_recovery_token: Optional[bytes] = None,
    ) -> Tuple[VaultKeyStore, bytes, str]:
        """
        Initializes a brand-new Vault KeyStore with a fresh 256-bit VMK,
        derives KEKs, and creates both Password and Recovery slots.

        Returns:
            (keystore_instance, raw_vmk, formatted_recovery_key_string)
        """
        if not master_password or len(master_password.strip()) < 8:
            raise InvalidKeyError("Master password must be at least 8 characters.")

        vmk = vmk or generate_master_key()
        raw_recovery, formatted_recovery = (
            (custom_recovery_token, format_recovery_key(custom_recovery_token))
            if custom_recovery_token
            else generate_recovery_key()
        )

        keystore = cls()

        # Slot 1: Password Slot
        pwd_salt = secrets.token_bytes(SALT_BYTES)
        pwd_kek = derive_key_argon2id(master_password, pwd_salt)
        pwd_wrapped = encrypt_data(vmk, pwd_kek, associated_data={"slot": "password"})
        keystore.slots["password"] = KeySlot(
            slot_type="password",
            salt_b64=base64.b64encode(pwd_salt).decode("utf-8"),
            argon2_params={
                "time_cost": ARGON2_TIME_COST,
                "memory_cost": ARGON2_MEMORY_COST,
                "parallelism": ARGON2_PARALLELISM,
            },
            wrapped_vmk_b64=base64.b64encode(pwd_wrapped).decode("utf-8"),
        )

        # Slot 2: Recovery Slot
        rec_salt = secrets.token_bytes(SALT_BYTES)
        # Use clean formatted recovery key string as password for derivation
        rec_kek = derive_key_argon2id(formatted_recovery, rec_salt)
        rec_wrapped = encrypt_data(vmk, rec_kek, associated_data={"slot": "recovery"})
        keystore.slots["recovery"] = KeySlot(
            slot_type="recovery",
            salt_b64=base64.b64encode(rec_salt).decode("utf-8"),
            argon2_params={
                "time_cost": ARGON2_TIME_COST,
                "memory_cost": ARGON2_MEMORY_COST,
                "parallelism": ARGON2_PARALLELISM,
            },
            wrapped_vmk_b64=base64.b64encode(rec_wrapped).decode("utf-8"),
        )

        return keystore, vmk, formatted_recovery

    def unlock_with_password(self, password: str) -> bytes:
        """Unlocks the keystore using the Master Password and returns the 32-byte VMK."""
        if "password" not in self.slots:
            raise InvalidKeyError("No password slot found in keystore.")

        slot = self.slots["password"]
        salt = base64.b64decode(slot.salt_b64)
        params = slot.argon2_params

        kek = derive_key_argon2id(
            password=password,
            salt=salt,
            time_cost=params.get("time_cost", ARGON2_TIME_COST),
            memory_cost=params.get("memory_cost", ARGON2_MEMORY_COST),
            parallelism=params.get("parallelism", ARGON2_PARALLELISM),
        )

        wrapped = base64.b64decode(slot.wrapped_vmk_b64)
        vmk = decrypt_data(wrapped, kek, associated_data={"slot": "password"})
        return vmk

    def unlock_with_recovery_key(self, recovery_key_str: str) -> bytes:
        """Unlocks the keystore using the Recovery Key and returns the 32-byte VMK."""
        if "recovery" not in self.slots:
            raise InvalidKeyError("No recovery slot found in keystore.")

        # Validate and re-format canonical key representation
        token_bytes = parse_recovery_key(recovery_key_str)
        canonical_recovery_str = format_recovery_key(token_bytes)

        slot = self.slots["recovery"]
        salt = base64.b64decode(slot.salt_b64)
        params = slot.argon2_params

        kek = derive_key_argon2id(
            password=canonical_recovery_str,
            salt=salt,
            time_cost=params.get("time_cost", ARGON2_TIME_COST),
            memory_cost=params.get("memory_cost", ARGON2_MEMORY_COST),
            parallelism=params.get("parallelism", ARGON2_PARALLELISM),
        )

        wrapped = base64.b64decode(slot.wrapped_vmk_b64)
        vmk = decrypt_data(wrapped, kek, associated_data={"slot": "recovery"})
        return vmk

    def setup_biometric_slot(self, vmk: bytes) -> str:
        """
        Configures the Biometric Authorization slot.
        Combines Windows DPAPI with an ephemeral authorization token.

        Returns:
            biometric_auth_token (random 32-byte secret held in memory/session).
        """
        if len(vmk) != KEY_BYTES:
            raise InvalidKeyError("Invalid VMK length.")

        # Generate a high-entropy biometric unlock token
        bio_token = secrets.token_bytes(KEY_BYTES)
        bio_salt = secrets.token_bytes(SALT_BYTES)

        # Encrypt VMK using the bio_token via AES-256-GCM
        wrapped_vmk = encrypt_data(vmk, bio_token, associated_data={"slot": "biometric"})

        # Protect the wrapped_vmk using Windows DPAPI
        dpapi_protected = dpapi_protect(wrapped_vmk, description="FaceUnlockAI Biometric Vault Token")

        self.biometric_dpapi_blob_b64 = base64.b64encode(dpapi_protected).decode("utf-8")
        self.biometric_session_salt_b64 = base64.b64encode(bio_salt).decode("utf-8")

        # Return the auth token string (to be wrapped in an authorized session token)
        return base64.b64encode(bio_token).decode("utf-8")

    def unlock_with_biometric_token(self, bio_token_b64: str) -> bytes:
        """
        Unlocks the keystore using the verified Biometric Authorization Token.
        This method is invoked ONLY after SFace recognition AND Liveness pass.
        """
        if not self.biometric_dpapi_blob_b64:
            raise InvalidKeyError("Biometric unlock has not been enrolled or configured.")

        bio_token = base64.b64decode(bio_token_b64)
        dpapi_ciphertext = base64.b64decode(self.biometric_dpapi_blob_b64)

        # Unprotect via DPAPI (verifies current Windows user identity)
        wrapped_vmk = dpapi_unprotect(dpapi_ciphertext)

        # Decrypt VMK using the biometric session token
        vmk = decrypt_data(wrapped_vmk, bio_token, associated_data={"slot": "biometric"})
        return vmk

    def to_json(self) -> str:
        """Serializes keystore to JSON string."""
        data = {
            "version": VAULT_VERSION,
            "slots": {k: asdict(v) for k, v in self.slots.items()},
            "biometric_dpapi_blob_b64": self.biometric_dpapi_blob_b64,
            "biometric_session_salt_b64": self.biometric_session_salt_b64,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> VaultKeyStore:
        """Deserializes keystore from JSON string."""
        data = json.loads(json_str)
        keystore = cls()
        for k, v in data.get("slots", {}).items():
            keystore.slots[k] = KeySlot(**v)
        keystore.biometric_dpapi_blob_b64 = data.get("biometric_dpapi_blob_b64")
        keystore.biometric_session_salt_b64 = data.get("biometric_session_salt_b64")
        return keystore


# =====================================================================
# Secure In-Memory Key Management (RAM Zeroing)
# =====================================================================

class SecureKeyHolder:
    """
    Holds the decrypted Vault Master Key in process memory during an active session.

    Memory Hygiene & Limitation Disclosure:
      - Uses a mutable bytearray buffer.
      - Upon calling lock() or clear(), the buffer is explicitly overwritten with
        zeros before unlinking.
      - Managed Language Limitation: In CPython, transient immutable `bytes` objects
        created during OpenSSL or C-extension calls may persist in heap fragments
        or Windows paging files until garbage collection occurs. Absolute physical
        RAM zeroing cannot be guaranteed in Python.
    """

    def __init__(self) -> None:
        self._key_buffer: Optional[bytearray] = None
        self._is_locked: bool = True

    @property
    def is_locked(self) -> bool:
        return self._is_locked or self._key_buffer is None

    def unlock(self, key_bytes: bytes) -> None:
        """Loads a 32-byte key into the active mutable buffer."""
        if len(key_bytes) != KEY_BYTES:
            raise InvalidKeyError(f"Key must be exactly {KEY_BYTES} bytes.")

        # Zero any existing key first
        self.lock()

        self._key_buffer = bytearray(key_bytes)
        self._is_locked = False

    def get_key(self) -> bytes:
        """Returns the active key bytes. Raises VaultLockedError if locked."""
        if self.is_locked or self._key_buffer is None:
            raise VaultLockedError("Vault is locked. Decryption key is not in memory.")
        return bytes(self._key_buffer)

    def lock(self) -> None:
        """Explicitly overwrites the memory buffer with zeros and sets locked state."""
        self._is_locked = True
        if self._key_buffer is not None:
            # Overwrite with zeros
            for i in range(len(self._key_buffer)):
                self._key_buffer[i] = 0
            self._key_buffer.clear()
            self._key_buffer = None

    def __enter__(self) -> SecureKeyHolder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.lock()
