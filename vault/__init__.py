"""
FaceUnlockAI Secure Vault — Cryptographic & Storage Subsystem
"""

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
    RECOVERY_BYTES,
)

from vault.vault_manager import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    FileSizeLimitExceededError,
    ManifestCorruptedError,
    SecurityValidationError,
    UnsupportedFormatError,
    VaultDocumentMetadata,
    VaultManager,
    VaultManagerError,
    inspect_file_type,
    sanitize_filename,
)

__all__ = [
    "CryptoError",
    "InvalidKeyError",
    "AuthenticationError",
    "CorruptedDataError",
    "VaultLockedError",
    "RecoveryKeyError",
    "SecureKeyHolder",
    "VaultKeyStore",
    "generate_master_key",
    "generate_recovery_key",
    "format_recovery_key",
    "parse_recovery_key",
    "derive_key_argon2id",
    "encrypt_data",
    "decrypt_data",
    "RECOVERY_BYTES",
    "VaultManager",
    "VaultDocumentMetadata",
    "VaultManagerError",
    "UnsupportedFormatError",
    "SecurityValidationError",
    "FileSizeLimitExceededError",
    "DocumentNotFoundError",
    "ManifestCorruptedError",
    "DuplicateDocumentError",
    "sanitize_filename",
    "inspect_file_type",
]


