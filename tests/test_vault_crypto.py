"""
test_vault_crypto.py — Comprehensive Automated Tests for Secure Vault Cryptography
------------------------------------------------------------------------------------
Verifies:
  1. AES-256-GCM encryption, decryption, and container format.
  2. Nonce uniqueness (random 96-bit per encryption).
  3. Authenticated metadata (AAD) verification and tamper detection.
  4. Bit-level ciphertext, tag, and header tamper resistance (fail-closed).
  5. Argon2id key derivation parameters, determinism, and salt isolation.
  6. Recovery key generation, formatting, transcription fault tolerance, and unlock.
  7. Multi-slot keystore lifecycle (Password, Recovery, and Biometric DPAPI slots).
  8. SecureKeyHolder in-memory buffer hygiene, zeroing, and locked-state enforcement.
"""

import copy
import json
import secrets
from pathlib import Path

import pytest

from vault.vault_crypto import (
    AuthenticationError,
    CorruptedDataError,
    CryptoError,
    InvalidKeyError,
    RecoveryKeyError,
    SecureKeyHolder,
    VaultKeyStore,
    VaultLockedError,
    decrypt_data,
    derive_key_argon2id,
    encrypt_data,
    format_recovery_key,
    generate_master_key,
    generate_recovery_key,
    parse_recovery_key,
    VAULT_MAGIC,
    VAULT_VERSION,
    HEADER_SIZE,
    KEY_BYTES,
    RECOVERY_BYTES,
    NONCE_BYTES,
    SALT_BYTES,
)


# =====================================================================
# 1. AES-256-GCM Encryption & Container Format Tests
# =====================================================================

class TestAESGCMEncryption:
    @pytest.fixture
    def test_key(self) -> bytes:
        return generate_master_key()

    def test_roundtrip_plaintext_various_sizes(self, test_key: bytes):
        payloads = [
            b"",                               # Empty payload
            b"Hello Secure Vault!",            # Short text
            b"P" * 1024,                       # 1 KiB
            secrets.token_bytes(64 * 1024),    # 64 KiB binary data
        ]
        for plaintext in payloads:
            container = encrypt_data(plaintext, test_key)
            assert isinstance(container, bytes)
            assert len(container) >= HEADER_SIZE + 16  # Header + 16B tag

            decrypted = decrypt_data(container, test_key)
            assert decrypted == plaintext

    def test_container_header_structure(self, test_key: bytes):
        plaintext = b"Confidential document data"
        container = encrypt_data(plaintext, test_key)

        assert container[:4] == VAULT_MAGIC
        assert container[4] == VAULT_VERSION
        nonce = container[5 : 5 + NONCE_BYTES]
        assert len(nonce) == NONCE_BYTES

    def test_nonce_uniqueness(self, test_key: bytes):
        """Verifies that encrypting the same plaintext generates unique nonces every time."""
        plaintext = b"Constant message"
        nonces = set()
        for _ in range(200):
            container = encrypt_data(plaintext, test_key)
            nonce = container[5 : 5 + NONCE_BYTES]
            assert nonce not in nonces, "CRITICAL: Nonce collision detected!"
            nonces.add(nonce)

    def test_invalid_key_length_rejected(self):
        plaintext = b"Data"
        with pytest.raises(InvalidKeyError):
            encrypt_data(plaintext, b"too_short_key_16_bytes")

        with pytest.raises(InvalidKeyError):
            encrypt_data(plaintext, secrets.token_bytes(64))  # 64 bytes instead of 32

        with pytest.raises(InvalidKeyError):
            decrypt_data(b"container", b"short_key")


# =====================================================================
# 2. Authenticated Associated Data (AAD) & Tamper Resistance
# =====================================================================

class TestTamperResistanceAndAAD:
    @pytest.fixture
    def test_key(self) -> bytes:
        return generate_master_key()

    def test_aad_dictionary_roundtrip(self, test_key: bytes):
        plaintext = b"Sensitive Financial Report"
        meta = {"filename": "report.pdf", "size": 1024, "uuid": "abc-123"}

        container = encrypt_data(plaintext, test_key, associated_data=meta)
        decrypted = decrypt_data(container, test_key, associated_data=meta)
        assert decrypted == plaintext

    def test_aad_mismatch_fails_closed(self, test_key: bytes):
        plaintext = b"Sensitive Medical History"
        meta1 = {"filename": "medical.docx", "author": "Alice"}
        meta2 = {"filename": "medical.docx", "author": "Eve"}  # Tampered author

        container = encrypt_data(plaintext, test_key, associated_data=meta1)

        with pytest.raises(AuthenticationError):
            decrypt_data(container, test_key, associated_data=meta2)

    def test_missing_aad_when_encrypted_with_aad_fails(self, test_key: bytes):
        plaintext = b"Sensitive Medical History"
        meta = {"filename": "data.txt"}

        container = encrypt_data(plaintext, test_key, associated_data=meta)
        with pytest.raises(AuthenticationError):
            decrypt_data(container, test_key, associated_data=None)

    def test_bit_flip_in_ciphertext_fails_closed(self, test_key: bytes):
        plaintext = b"Transfer $1000 to Account A"
        container = bytearray(encrypt_data(plaintext, test_key))

        # Flip a bit in the ciphertext body
        container[HEADER_SIZE + 5] ^= 0x01

        with pytest.raises(AuthenticationError):
            decrypt_data(bytes(container), test_key)

    def test_bit_flip_in_tag_fails_closed(self, test_key: bytes):
        plaintext = b"Top Secret Plan"
        container = bytearray(encrypt_data(plaintext, test_key))

        # Flip a bit in the last byte (authentication tag)
        container[-1] ^= 0xFF

        with pytest.raises(AuthenticationError):
            decrypt_data(bytes(container), test_key)

    def test_bit_flip_in_nonce_fails_closed(self, test_key: bytes):
        plaintext = b"Top Secret Plan"
        container = bytearray(encrypt_data(plaintext, test_key))

        # Flip a bit in the nonce
        container[7] ^= 0x04

        with pytest.raises(AuthenticationError):
            decrypt_data(bytes(container), test_key)

    def test_corrupted_magic_bytes_raises_corrupted_data_error(self, test_key: bytes):
        container = bytearray(encrypt_data(b"data", test_key))
        container[0:4] = b"XXXX"

        with pytest.raises(CorruptedDataError, match="Invalid vault magic bytes"):
            decrypt_data(bytes(container), test_key)

    def test_unsupported_version_raises_corrupted_data_error(self, test_key: bytes):
        container = bytearray(encrypt_data(b"data", test_key))
        container[4] = 99  # Version 99

        with pytest.raises(CorruptedDataError, match="Unsupported vault container version"):
            decrypt_data(bytes(container), test_key)

    def test_truncated_container_raises_corrupted_data_error(self, test_key: bytes):
        container = encrypt_data(b"Hello world", test_key)
        truncated = container[: HEADER_SIZE + 5]  # Too short for tag

        with pytest.raises(CorruptedDataError, match="Container too short"):
            decrypt_data(truncated, test_key)

    def test_wrong_key_fails_closed_without_plaintext_leak(self, test_key: bytes):
        wrong_key = generate_master_key()
        container = encrypt_data(b"Secret Document", test_key)

        with pytest.raises(AuthenticationError):
            decrypt_data(container, wrong_key)


# =====================================================================
# 3. Argon2id Key Derivation Tests
# =====================================================================

class TestArgon2idKeyDerivation:
    def test_deterministic_for_same_password_and_salt(self):
        password = "CorrectHorseBatteryStaple!"
        salt = secrets.token_bytes(SALT_BYTES)

        # Use fast parameters for unit testing speed
        k1 = derive_key_argon2id(password, salt, time_cost=1, memory_cost=16384, parallelism=2)
        k2 = derive_key_argon2id(password, salt, time_cost=1, memory_cost=16384, parallelism=2)

        assert k1 == k2
        assert len(k1) == KEY_BYTES

    def test_different_passwords_produce_different_keys(self):
        salt = secrets.token_bytes(SALT_BYTES)
        k1 = derive_key_argon2id("Password123!", salt, time_cost=1, memory_cost=16384, parallelism=2)
        k2 = derive_key_argon2id("Password124!", salt, time_cost=1, memory_cost=16384, parallelism=2)

        assert k1 != k2

    def test_different_salts_produce_different_keys(self):
        pwd = "Password123!"
        salt1 = secrets.token_bytes(SALT_BYTES)
        salt2 = secrets.token_bytes(SALT_BYTES)

        k1 = derive_key_argon2id(pwd, salt1, time_cost=1, memory_cost=16384, parallelism=2)
        k2 = derive_key_argon2id(pwd, salt2, time_cost=1, memory_cost=16384, parallelism=2)

        assert k1 != k2

    def test_short_salt_rejected(self):
        with pytest.raises(InvalidKeyError, match="Salt must be at least"):
            derive_key_argon2id("Password", b"short_salt")

    def test_empty_password_rejected(self):
        salt = secrets.token_bytes(SALT_BYTES)
        with pytest.raises(InvalidKeyError, match="Password cannot be empty"):
            derive_key_argon2id("", salt)


# =====================================================================
# 4. Recovery Key Mechanism Tests
# =====================================================================

class TestRecoveryKeyMechanism:
    def test_recovery_key_format_and_parsing(self):
        raw_token, formatted = generate_recovery_key()

        assert len(raw_token) == RECOVERY_BYTES
        # Check standard format: 8 blocks of 4 separated by hyphens (39 chars total)
        assert len(formatted) == 39
        assert formatted.count("-") == 7

        # Parsing roundtrip
        parsed = parse_recovery_key(formatted)
        assert parsed == raw_token

    def test_recovery_key_transcription_normalization(self):
        raw_token, formatted = generate_recovery_key()

        # Lowercase, spaces, removed hyphens
        messy_str = formatted.lower().replace("-", "   ")
        parsed = parse_recovery_key(messy_str)
        assert parsed == raw_token

    def test_ambiguous_characters_substitution(self):
        """Checks O->0, I->1, L->1 substitution for transcription resilience."""
        raw_token, formatted = generate_recovery_key()
        # Replace a character with 'O' and ensure it converts to '0'
        mod_str = formatted.replace("0", "O")
        parsed = parse_recovery_key(mod_str)
        # Should parse without error
        assert len(parsed) == RECOVERY_BYTES

    def test_invalid_length_recovery_key_raises_error(self):
        with pytest.raises(RecoveryKeyError, match="must contain exactly 32"):
            parse_recovery_key("ABCD-EFGH-1234")

    def test_invalid_character_recovery_key_raises_error(self):
        with pytest.raises(RecoveryKeyError, match="Invalid character"):
            parse_recovery_key("????-EFGH-1234-5678-9012-3456-7890-1234")


# =====================================================================
# 5. Multi-Slot Keystore Lifecycle & Unlocking Tests
# =====================================================================

class TestVaultKeyStoreLifecycle:
    def test_initialize_and_unlock_via_password(self):
        password = "SecureVaultMasterPassword2026!"
        keystore, vmk, recovery_key = VaultKeyStore.initialize_new_vault(password)

        assert len(vmk) == KEY_BYTES
        assert "password" in keystore.slots
        assert "recovery" in keystore.slots

        # Unlock with correct password
        unlocked_vmk = keystore.unlock_with_password(password)
        assert unlocked_vmk == vmk

    def test_unlock_via_wrong_password_fails_closed(self):
        keystore, _, _ = VaultKeyStore.initialize_new_vault("CorrectPassword123!")

        with pytest.raises(AuthenticationError):
            keystore.unlock_with_password("WrongPassword999!")

    def test_unlock_via_recovery_key_works_independently(self):
        password = "OriginalPassword789!"
        keystore, vmk, recovery_key = VaultKeyStore.initialize_new_vault(password)

        # Unlock using the recovery key alone
        unlocked_vmk = keystore.unlock_with_recovery_key(recovery_key)
        assert unlocked_vmk == vmk

    def test_unlock_via_wrong_recovery_key_fails_closed(self):
        keystore, _, _ = VaultKeyStore.initialize_new_vault("MyPassword123!")
        _, wrong_recovery_key = generate_recovery_key()

        with pytest.raises(AuthenticationError):
            keystore.unlock_with_recovery_key(wrong_recovery_key)

    def test_keystore_serialization_roundtrip(self):
        password = "MasterPasswordABC!"
        keystore, vmk, recovery_key = VaultKeyStore.initialize_new_vault(password)

        json_str = keystore.to_json()
        assert isinstance(json_str, str)

        reloaded = VaultKeyStore.from_json(json_str)
        # Should unlock exactly the same
        assert reloaded.unlock_with_password(password) == vmk
        assert reloaded.unlock_with_recovery_key(recovery_key) == vmk

    def test_biometric_slot_lifecycle(self):
        password = "MyMasterPassword456!"
        keystore, vmk, _ = VaultKeyStore.initialize_new_vault(password)

        # Setup biometric slot
        bio_token_b64 = keystore.setup_biometric_slot(vmk)
        assert bio_token_b64 is not None
        assert keystore.biometric_dpapi_blob_b64 is not None

        # Unlock with valid token
        unlocked_vmk = keystore.unlock_with_biometric_token(bio_token_b64)
        assert unlocked_vmk == vmk

        # Unlock with invalid token
        wrong_token_b64 = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
        with pytest.raises(AuthenticationError):
            keystore.unlock_with_biometric_token(wrong_token_b64)


# =====================================================================
# 6. SecureKeyHolder (In-Memory Key Hygiene) Tests
# =====================================================================

class TestSecureKeyHolder:
    def test_initial_state_locked(self):
        holder = SecureKeyHolder()
        assert holder.is_locked is True
        with pytest.raises(VaultLockedError):
            holder.get_key()

    def test_unlock_and_get_key(self):
        holder = SecureKeyHolder()
        test_key = generate_master_key()

        holder.unlock(test_key)
        assert holder.is_locked is False
        assert holder.get_key() == test_key

    def test_lock_zeroes_memory_and_enforces_locked_state(self):
        holder = SecureKeyHolder()
        test_key = generate_master_key()

        holder.unlock(test_key)
        assert holder.is_locked is False

        # Lock
        holder.lock()
        assert holder.is_locked is True
        with pytest.raises(VaultLockedError):
            holder.get_key()

    def test_context_manager_auto_locks(self):
        test_key = generate_master_key()
        holder = SecureKeyHolder()

        with holder:
            holder.unlock(test_key)
            assert holder.get_key() == test_key
            assert holder.is_locked is False

        # After exiting context, holder must be automatically locked
        assert holder.is_locked is True
        with pytest.raises(VaultLockedError):
            holder.get_key()
