"""
vault_manager.py — Vault Document Manager & Encrypted Container
-----------------------------------------------------------------
Manages secure storage, atomic file operations, content inspection,
indexing, import, and export of confidential documents.

Core Security & Architectural Principles:
  1. Authenticated Ciphertext Only: Document contents are encrypted with AES-256-GCM
     and stored on disk using unpredictable UUID identifiers (never plaintext filenames).
  2. Encrypted Manifest: Document metadata (original filename, file size, MIME type,
     SHA-256 integrity hash, timestamps) is stored in an encrypted index (manifest.enc).
  3. Metadata Authentication (AAD): Document AAD is tied to doc_id, filename, and size.
     Tampering with metadata or swapping ciphertext files triggers immediate decryption failure.
  4. Safe Atomic Writes: Files and manifests are staged via temporary files (.tmp.<hex>)
     and atomically swapped with os.replace + os.fsync to ensure crash resilience.
  5. Content Inspection (No blind extension trust): Inspects magic bytes to verify
     authentic PDF, DOCX, TXT, Markdown, CSV, JSON, and common image formats.
     Actively detects and rejects executable formats (Windows PE/MZ, Linux ELF, Mach-O).
  6. Source File Preservation: Original source files are strictly read; they are never
     automatically modified, truncated, or deleted.
  7. Path Traversal Defense: Strict filename sanitization rejects ../, absolute paths,
     null bytes, and Windows reserved characters.
  8. Memory Hygiene & Size Limits: Implements configurable file size limits (default 50 MiB)
     to prevent memory exhaustion / DoS attacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vault.vault_crypto import (
    AuthenticationError,
    CorruptedDataError,
    CryptoError,
    InvalidKeyError,
    SecureKeyHolder,
    VaultLockedError,
    decrypt_data,
    encrypt_data,
    KEY_BYTES,
)

# Limits & Defaults
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB maximum per document
MANIFEST_FILENAME = "manifest.enc"
MANIFEST_BACKUP_FILENAME = "manifest.enc.bak"
DOCUMENTS_DIRNAME = "documents"
STAGING_DIRNAME = ".tmp"


# =====================================================================
# Custom Exceptions (Fail-Closed)
# =====================================================================

class VaultManagerError(Exception):
    """Base exception for all document vault manager operations."""


class UnsupportedFormatError(VaultManagerError):
    """Raised when an imported file format is unsupported or dangerous."""


class SecurityValidationError(VaultManagerError):
    """Raised when path traversal, dangerous filenames, or executable code is detected."""


class FileSizeLimitExceededError(VaultManagerError):
    """Raised when an imported document exceeds the configured size limit."""


class DocumentNotFoundError(VaultManagerError):
    """Raised when a requested document ID does not exist in the manifest or storage."""


class ManifestCorruptedError(VaultManagerError):
    """Raised when the encrypted manifest cannot be decrypted or is malformed."""


class DuplicateDocumentError(VaultManagerError):
    """Raised when duplicate file collision policies are violated."""


# =====================================================================
# Document Metadata Model
# =====================================================================

@dataclass
class VaultDocumentMetadata:
    doc_id: str
    original_filename: str
    file_size: int
    mime_type: str
    sha256: str
    created_at: str
    updated_at: str
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VaultDocumentMetadata:
        return cls(
            doc_id=data["doc_id"],
            original_filename=data["original_filename"],
            file_size=int(data["file_size"]),
            mime_type=data["mime_type"],
            sha256=data["sha256"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            custom_metadata=data.get("custom_metadata", {}),
        )


# =====================================================================
# Safe Filename & Path Utilities
# =====================================================================

# Windows reserved device names
RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent path traversal, null-byte injection,
    and Windows device name conflicts.
    """
    if not filename:
        return f"document_{secrets.token_hex(4)}.dat"

    # Strip directory components (e.g. ../../../etc/passwd -> passwd)
    clean = Path(filename).name

    # Remove null bytes, control characters, and illegal Windows filesystem characters
    clean = re.sub(r'[\x00-\x1f\\/:*?"<>|]', "_", clean).strip(" .")

    # Check for reserved Windows filenames (CON, PRN, etc.)
    base_name = clean.split(".")[0].upper()
    if base_name in RESERVED_WINDOWS_NAMES:
        clean = f"safe_{clean}"

    if not clean or clean in (".", ".."):
        clean = f"document_{secrets.token_hex(4)}.dat"

    return clean


# =====================================================================
# Content Inspection & Format Validation
# =====================================================================

# Magic byte signatures
MAGIC_PDF = b"%PDF-"
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_JPG = b"\xff\xd8\xff"
MAGIC_BMP = b"BM"
MAGIC_ZIP = b"PK\x03\x04"  # Base signature for DOCX
MAGIC_PE = b"MZ"           # Windows executable / DLL
MAGIC_ELF = b"\x7fELF"     # Linux ELF binary
MAGIC_MACHO_LIST = (
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",   # Java class / Universal binary
)


def inspect_file_type(data: bytes, filename: str) -> str:
    """
    Inspects file content magic bytes and format structure to determine MIME type.
    Actively rejects executable binaries and unsupported formats.

    Returns:
        MIME type string (e.g., 'application/pdf', 'image/png', 'text/plain').

    Raises:
        SecurityValidationError: If executable code or dangerous header is found.
        UnsupportedFormatError: If the format is unsupported or inconsistent.
    """
    if not data:
        # Empty file
        ext = Path(filename).suffix.lower()
        if ext in (".txt", ".md", ".csv", ".json"):
            return "text/plain"
        raise UnsupportedFormatError(f"File '{filename}' is empty and not a valid text document.")

    # 1. Executable and binary signature checks (Strict Rejection)
    if data.startswith(MAGIC_PE):
        raise SecurityValidationError(
            f"Security Violation: '{filename}' contains a Windows executable (PE/MZ) header."
        )
    if data.startswith(MAGIC_ELF):
        raise SecurityValidationError(
            f"Security Violation: '{filename}' contains a Linux ELF executable header."
        )
    for macho_sig in MAGIC_MACHO_LIST:
        if data.startswith(macho_sig):
            raise SecurityValidationError(
                f"Security Violation: '{filename}' contains an executable binary header."
            )

    # Reject shebang scripts (#!)
    if data.startswith(b"#!"):
        raise SecurityValidationError(
            f"Security Violation: '{filename}' contains an executable script shebang."
        )

    # 2. PDF Detection
    if data.startswith(MAGIC_PDF):
        return "application/pdf"

    # 3. PNG Image Detection
    if data.startswith(MAGIC_PNG):
        return "image/png"

    # 4. JPEG Image Detection
    if data.startswith(MAGIC_JPG):
        return "image/jpeg"

    # 5. BMP Image Detection
    if data.startswith(MAGIC_BMP) and len(data) >= 14:
        return "image/bmp"

    # 6. WEBP Image Detection (RIFF....WEBP)
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # 7. DOCX Detection (ZIP container containing Word document markers)
    if data.startswith(MAGIC_ZIP):
        ext = Path(filename).suffix.lower()
        if ext == ".docx":
            # Verify it's an Office Open XML document
            if b"[Content_Types].xml" in data or b"word/" in data:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            raise UnsupportedFormatError(
                f"'{filename}' has ZIP header but lacks valid DOCX WordprocessingML structure."
            )
        raise UnsupportedFormatError(f"Generic ZIP archives are not supported for '{filename}'.")

    # 8. Text-Based Formats (TXT, Markdown, CSV, JSON)
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md", ".markdown", ".csv", ".json", ".rtf"):
        # Text files must not contain binary null bytes
        if b"\x00" in data[:4096]:
            raise SecurityValidationError(
                f"'{filename}' has text extension '{ext}' but contains binary null bytes."
            )
        try:
            # Validate UTF-8 encoding
            data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data.decode("latin-1")
            except Exception as exc:
                raise UnsupportedFormatError(f"Text encoding error in '{filename}': {exc}") from exc

        if ext == ".json":
            try:
                json.loads(data.decode("utf-8"))
                return "application/json"
            except Exception as exc:
                raise UnsupportedFormatError(f"Invalid JSON format in '{filename}': {exc}") from exc
        if ext in (".md", ".markdown"):
            return "text/markdown"
        if ext == ".csv":
            return "text/csv"
        return "text/plain"

    # If extension claims to be an image or PDF but magic bytes mismatched:
    if ext in (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".docx"):
        raise UnsupportedFormatError(
            f"Content mismatch: '{filename}' extension is '{ext}' but content signature does not match."
        )

    raise UnsupportedFormatError(
        f"Unsupported file format for '{filename}'. Allowed formats: PDF, DOCX, TXT, MD, CSV, JSON, PNG, JPG, BMP, WEBP."
    )


# =====================================================================
# Atomic File System Utilities
# =====================================================================

def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """
    Safely and atomically writes bytes to target_path using a temporary file
    and atomic os.replace with disk flush (fsync).
    """
    target_path = Path(target_path).resolve()
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Place temp file in the exact same directory to ensure same filesystem for os.replace
    temp_path = target_dir / f"{target_path.name}.tmp.{secrets.token_hex(6)}"
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# =====================================================================
# Vault Storage Manager
# =====================================================================

class VaultManager:
    """
    Manages encrypted document storage, atomic manifest persistence,
    document ingestion, authorized decryption, and secure metadata validation.
    """

    def __init__(
        self,
        vault_root: Union[str, Path],
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.documents_dir = self.vault_root / DOCUMENTS_DIRNAME
        self.manifest_file = self.vault_root / MANIFEST_FILENAME
        self.manifest_backup = self.vault_root / MANIFEST_BACKUP_FILENAME
        self.staging_dir = self.vault_root / STAGING_DIRNAME
        self.max_file_size = max_file_size

        self._ensure_storage_directories()

    def _ensure_storage_directories(self) -> None:
        """Creates vault directories and sets restrictive Windows ACLs if applicable."""
        self.vault_root.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Apply Windows ACL security to vault directory
        try:
            from face_registration import secure_directory_acls
            secure_directory_acls(self.vault_root)
        except Exception:
            pass

    @staticmethod
    def _resolve_key(key: Union[bytes, SecureKeyHolder]) -> bytes:
        """Extracts the 32-byte key from either raw bytes or a SecureKeyHolder."""
        if isinstance(key, SecureKeyHolder):
            return key.get_key()
        if isinstance(key, bytes):
            if len(key) != KEY_BYTES:
                raise InvalidKeyError(f"Vault key must be {KEY_BYTES} bytes, got {len(key)}.")
            return key
        raise InvalidKeyError(f"Unsupported key object type: {type(key)}")

    # -----------------------------------------------------------------
    # Manifest Management
    # -----------------------------------------------------------------

    def _load_manifest_dict(self, key: Union[bytes, SecureKeyHolder]) -> Dict[str, Any]:
        """Loads and decrypts the vault manifest, with automatic backup recovery."""
        resolved_key = self._resolve_key(key)

        if not self.manifest_file.exists():
            if self.manifest_backup.exists():
                # Attempt recovery from backup
                try:
                    return self._decrypt_manifest_file(self.manifest_backup, resolved_key)
                except Exception:
                    pass
            # Initialize brand new manifest structure
            return {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "documents": {},
            }

        try:
            return self._decrypt_manifest_file(self.manifest_file, resolved_key)
        except (AuthenticationError, CorruptedDataError) as exc:
            # Primary manifest failed; attempt backup recovery
            if self.manifest_backup.exists():
                try:
                    recovered = self._decrypt_manifest_file(self.manifest_backup, resolved_key)
                    # Restore primary from backup
                    atomic_write_bytes(self.manifest_file, self.manifest_backup.read_bytes())
                    return recovered
                except Exception:
                    pass
            raise ManifestCorruptedError(
                f"Failed to decrypt vault manifest: {exc}. Key may be invalid or data tampered."
            ) from exc

    def _decrypt_manifest_file(self, path: Path, resolved_key: bytes) -> Dict[str, Any]:
        """Decrypts a specific manifest file and returns parsed dict."""
        ciphertext = path.read_bytes()
        plaintext = decrypt_data(ciphertext, resolved_key, associated_data={"type": "vault_manifest"})
        return json.loads(plaintext.decode("utf-8"))

    def _save_manifest_dict(self, manifest_data: Dict[str, Any], key: Union[bytes, SecureKeyHolder]) -> None:
        """Encrypts and atomically saves the manifest, updating backup."""
        resolved_key = self._resolve_key(key)
        manifest_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        json_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")
        ciphertext = encrypt_data(json_bytes, resolved_key, associated_data={"type": "vault_manifest"})

        # Backup existing manifest if present (preserving previous snapshot)
        if self.manifest_file.exists():
            try:
                shutil.copy2(self.manifest_file, self.manifest_backup)
            except OSError:
                pass

        # Atomic write of new manifest
        atomic_write_bytes(self.manifest_file, ciphertext)

        # Ensure backup exists even on initial vault creation
        if not self.manifest_backup.exists() and self.manifest_file.exists():
            try:
                shutil.copy2(self.manifest_file, self.manifest_backup)
            except OSError:
                pass


    # -----------------------------------------------------------------
    # Document Operations
    # -----------------------------------------------------------------

    def list_documents(self, key: Union[bytes, SecureKeyHolder]) -> List[VaultDocumentMetadata]:
        """
        Lists all documents stored in the vault without decrypting document payloads.
        Returns a list of VaultDocumentMetadata objects sorted by date added.
        """
        manifest_data = self._load_manifest_dict(key)
        docs = []
        for doc_dict in manifest_data.get("documents", {}).values():
            docs.append(VaultDocumentMetadata.from_dict(doc_dict))
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs

    def get_document_metadata(
        self, doc_id: str, key: Union[bytes, SecureKeyHolder]
    ) -> VaultDocumentMetadata:
        """Retrieves metadata for a specific document without decrypting its contents."""
        manifest_data = self._load_manifest_dict(key)
        docs = manifest_data.get("documents", {})
        if doc_id not in docs:
            raise DocumentNotFoundError(f"Document ID '{doc_id}' not found in vault manifest.")
        return VaultDocumentMetadata.from_dict(docs[doc_id])

    def import_document(
        self,
        source_path: Union[str, Path],
        key: Union[bytes, SecureKeyHolder],
        custom_filename: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> VaultDocumentMetadata:
        """
        Imports and encrypts a file into the vault.

        Safety Guarantees:
          - Validates format via magic byte inspection.
          - Rejects executables.
          - Enforces maximum file size limit.
          - Preserves source file untouched.
          - Generates unpredictable UUID4 identifier.
          - Uses atomic writes to prevent partially written records.
        """
        resolved_key = self._resolve_key(key)
        source = Path(source_path).resolve()

        if not source.exists():
            raise FileNotFoundError(f"Source document '{source_path}' does not exist.")
        if not source.is_file():
            raise ValueError(f"Source path '{source_path}' is not a regular file.")

        # Check size before loading
        file_size = source.stat().st_size
        if file_size > self.max_file_size:
            raise FileSizeLimitExceededError(
                f"File '{source.name}' ({file_size} bytes) exceeds maximum vault limit of {self.max_file_size} bytes."
            )

        # Read source data
        raw_data = source.read_bytes()

        # Sanitize filename
        original_name = sanitize_filename(custom_filename or source.name)

        # Validate content and detect MIME type (content inspection)
        mime_type = inspect_file_type(raw_data, original_name)

        # Compute SHA-256 for integrity verification
        sha256_hash = hashlib.sha256(raw_data).hexdigest()

        # Generate unpredictable internal identifier
        doc_id = secrets.token_hex(16)  # 32-character UUID-like hex string
        target_cipher_file = self.documents_dir / f"{doc_id}.enc"

        # Build AAD metadata for cryptographic binding
        aad = {
            "doc_id": doc_id,
            "original_filename": original_name,
            "file_size": file_size,
            "sha256": sha256_hash,
        }

        # Encrypt payload with AES-256-GCM
        ciphertext = encrypt_data(raw_data, resolved_key, associated_data=aad)

        # Atomically write encrypted payload
        atomic_write_bytes(target_cipher_file, ciphertext)

        # Prepare metadata entry
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = VaultDocumentMetadata(
            doc_id=doc_id,
            original_filename=original_name,
            file_size=file_size,
            mime_type=mime_type,
            sha256=sha256_hash,
            created_at=now_iso,
            updated_at=now_iso,
            custom_metadata=custom_metadata or {},
        )

        # Update manifest atomically
        try:
            manifest_data = self._load_manifest_dict(resolved_key)
            manifest_data["documents"][doc_id] = metadata.to_dict()
            self._save_manifest_dict(manifest_data, resolved_key)
        except Exception as exc:
            # If manifest save fails, clean up the orphaned encrypted payload
            if target_cipher_file.exists():
                try:
                    target_cipher_file.unlink()
                except OSError:
                    pass
            raise VaultManagerError(f"Failed to update manifest during import: {exc}") from exc

        return metadata

    def read_document_bytes(
        self, doc_id: str, key: Union[bytes, SecureKeyHolder]
    ) -> Tuple[bytes, VaultDocumentMetadata]:
        """
        Decrypts and returns document plaintext bytes in memory for safe preview.
        Validates metadata and payload SHA-256 integrity.

        Returns:
            (plaintext_bytes, metadata)
        """
        resolved_key = self._resolve_key(key)
        manifest_data = self._load_manifest_dict(resolved_key)

        docs = manifest_data.get("documents", {})
        if doc_id not in docs:
            raise DocumentNotFoundError(f"Document ID '{doc_id}' not found in vault.")

        meta = VaultDocumentMetadata.from_dict(docs[doc_id])
        target_file = self.documents_dir / f"{doc_id}.enc"

        if not target_file.exists():
            raise DocumentNotFoundError(
                f"Ciphertext for document '{meta.original_filename}' ({doc_id}) is missing from disk."
            )

        # Reconstruct authenticated AAD
        aad = {
            "doc_id": meta.doc_id,
            "original_filename": meta.original_filename,
            "file_size": meta.file_size,
            "sha256": meta.sha256,
        }

        ciphertext = target_file.read_bytes()
        # Decrypt with AES-256-GCM + AAD validation
        plaintext = decrypt_data(ciphertext, resolved_key, associated_data=aad)

        # Verify integrity hash
        actual_hash = hashlib.sha256(plaintext).hexdigest()
        if actual_hash != meta.sha256:
            raise AuthenticationError(
                f"Integrity check failed: Plaintext SHA-256 ({actual_hash}) does not match manifest ({meta.sha256})."
            )

        return plaintext, meta

    def export_document(
        self,
        doc_id: str,
        destination_dir: Union[str, Path],
        key: Union[bytes, SecureKeyHolder],
        overwrite: bool = False,
        custom_export_filename: Optional[str] = None,
    ) -> Path:
        """
        Decrypts a document and exports it to the specified destination directory.
        Uses atomic file writing and filename sanitization to prevent path traversal.
        """
        dest_dir = Path(destination_dir).resolve()
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest_dir.is_dir():
            raise ValueError(f"Destination '{dest_dir}' is not a directory.")

        plaintext, meta = self.read_document_bytes(doc_id, key)

        out_name = sanitize_filename(custom_export_filename or meta.original_filename)
        dest_file = (dest_dir / out_name).resolve()

        # Prevent writing outside destination directory (Path Traversal defense)
        if dest_dir not in dest_file.parents and dest_file != dest_dir:
            raise SecurityValidationError("Path traversal detected: Destination path escapes target directory.")

        if dest_file.exists() and not overwrite:
            raise FileExistsError(
                f"Export file '{dest_file.name}' already exists in destination. Set overwrite=True to replace."
            )

        atomic_write_bytes(dest_file, plaintext)
        return dest_file

    def rename_document(
        self, doc_id: str, new_filename: str, key: Union[bytes, SecureKeyHolder]
    ) -> VaultDocumentMetadata:
        """
        Renames a document in the vault.
        Re-encrypts the ciphertext with the updated AAD to maintain metadata binding.
        """
        resolved_key = self._resolve_key(key)
        sanitized_new_name = sanitize_filename(new_filename)

        # Read current document
        plaintext, meta = self.read_document_bytes(doc_id, resolved_key)

        # Verify new filename extension matches content MIME type
        inspect_file_type(plaintext, sanitized_new_name)

        # Update metadata
        meta.original_filename = sanitized_new_name
        meta.updated_at = datetime.now(timezone.utc).isoformat()

        # Re-encrypt with new AAD
        new_aad = {
            "doc_id": meta.doc_id,
            "original_filename": meta.original_filename,
            "file_size": meta.file_size,
            "sha256": meta.sha256,
        }
        new_ciphertext = encrypt_data(plaintext, resolved_key, associated_data=new_aad)

        # Atomically write updated ciphertext
        cipher_file = self.documents_dir / f"{doc_id}.enc"
        atomic_write_bytes(cipher_file, new_ciphertext)

        # Update manifest
        manifest_data = self._load_manifest_dict(resolved_key)
        manifest_data["documents"][doc_id] = meta.to_dict()
        self._save_manifest_dict(manifest_data, resolved_key)

        return meta

    def delete_document(self, doc_id: str, key: Union[bytes, SecureKeyHolder]) -> bool:
        """
        Removes a document from the vault manifest and deletes its ciphertext file.

        Honest Deletion Disclosure:
          - Deletes the encrypted ciphertext file (.enc) from disk.
          - Note on Solid-State Drives (SSDs): Flash translation layers and wear-leveling
            prevent guaranteed physical zeroing of raw flash cells. However, since the
            file is encrypted with AES-256-GCM and its manifest entry is removed, residual
            ciphertext fragments on disk are cryptographically indecipherable without the key.
        """
        resolved_key = self._resolve_key(key)
        manifest_data = self._load_manifest_dict(resolved_key)

        docs = manifest_data.get("documents", {})
        if doc_id not in docs:
            raise DocumentNotFoundError(f"Document ID '{doc_id}' not found in vault.")

        # Remove from manifest
        del docs[doc_id]
        self._save_manifest_dict(manifest_data, resolved_key)

        # Remove ciphertext file
        cipher_file = self.documents_dir / f"{doc_id}.enc"
        if cipher_file.exists():
            try:
                cipher_file.unlink()
            except OSError:
                pass

        return True
