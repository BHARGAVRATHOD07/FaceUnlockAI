"""
test_vault_manager.py — Comprehensive Automated Tests for Vault Document Manager
-----------------------------------------------------------------------------------
Verifies:
  1. Document storage container, unpredictable UUIDs, and encrypted manifest.
  2. Multi-format import (PDF, DOCX, TXT, MD, CSV, JSON, PNG, JPG, BMP, WEBP).
  3. Content inspection & strict executable rejection (PE/MZ, ELF, shebangs).
  4. Misleading extension detection (e.g. EXE renamed to PDF, fake PNGs).
  5. Path traversal defense and filename sanitization (Windows reserved names).
  6. Source file preservation (zero modification of imported source files).
  7. Tamper resistance: modified ciphertext and AAD metadata mismatches.
  8. Atomic operations and manifest backup recovery.
  9. Document rename, authorized export, and deletion lifecycles.
  10. File size limit enforcement.
  11. Key resolution via raw 32-byte key or SecureKeyHolder.
"""

import json
import os
import secrets
from pathlib import Path

import pytest

from vault.vault_crypto import (
    AuthenticationError,
    CorruptedDataError,
    InvalidKeyError,
    SecureKeyHolder,
    VaultLockedError,
    generate_master_key,
)
from vault.vault_manager import (
    DEFAULT_MAX_FILE_SIZE,
    DocumentNotFoundError,
    FileSizeLimitExceededError,
    ManifestCorruptedError,
    SecurityValidationError,
    UnsupportedFormatError,
    VaultDocumentMetadata,
    VaultManager,
    inspect_file_type,
    sanitize_filename,
)


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    """Provides a fresh temporary directory for the vault."""
    v_dir = tmp_path / "test_vault"
    v_dir.mkdir(parents=True, exist_ok=True)
    return v_dir


@pytest.fixture
def manager(vault_dir: Path) -> VaultManager:
    return VaultManager(vault_root=vault_dir)


@pytest.fixture
def vmk() -> bytes:
    return generate_master_key()


# =====================================================================
# 1. Filename Sanitization & Path Traversal Tests
# =====================================================================

class TestFilenameSanitization:
    def test_path_traversal_stripped(self):
        malicious = "../../../../../etc/passwd"
        sanitized = sanitize_filename(malicious)
        assert sanitized == "passwd"
        assert "/" not in sanitized
        assert ".." not in sanitized

    def test_windows_path_traversal_stripped(self):
        malicious = r"..\..\Windows\System32\cmd.exe"
        sanitized = sanitize_filename(malicious)
        assert sanitized == "cmd.exe"
        assert "\\" not in sanitized

    def test_illegal_windows_characters_replaced(self):
        dirty = 'My:Report*<2026>?"data"|file.pdf'
        sanitized = sanitize_filename(dirty)
        for char in ':*?"<>|':
            assert char not in sanitized
        assert sanitized.endswith(".pdf")

    def test_reserved_windows_device_names(self):
        for name in ("CON", "PRN", "AUX", "NUL", "COM1", "LPT1"):
            filename = f"{name}.txt"
            sanitized = sanitize_filename(filename)
            assert sanitized.startswith("safe_")
            assert sanitized == f"safe_{name}.txt"

    def test_empty_or_whitespace_filename_handled(self):
        assert sanitize_filename("").startswith("document_")
        assert sanitize_filename("   ").startswith("document_")
        assert sanitize_filename(".").startswith("document_")
        assert sanitize_filename("..").startswith("document_")


# =====================================================================
# 2. Content Inspection & Executable Rejection Tests
# =====================================================================

class TestContentInspection:
    def test_valid_pdf_inspection(self):
        pdf_data = b"%PDF-1.7\nSample content..."
        assert inspect_file_type(pdf_data, "sample.pdf") == "application/pdf"

    def test_valid_png_inspection(self):
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        assert inspect_file_type(png_data, "image.png") == "image/png"

    def test_valid_jpeg_inspection(self):
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert inspect_file_type(jpeg_data, "photo.jpg") == "image/jpeg"

    def test_valid_bmp_inspection(self):
        bmp_data = b"BM" + b"\x00" * 20
        assert inspect_file_type(bmp_data, "graphic.bmp") == "image/bmp"

    def test_valid_webp_inspection(self):
        webp_data = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        assert inspect_file_type(webp_data, "banner.webp") == "image/webp"

    def test_valid_docx_inspection(self):
        # ZIP with [Content_Types].xml
        docx_data = b"PK\x03\x04\x00\x00[Content_Types].xml mock docx content"
        assert "wordprocessingml.document" in inspect_file_type(docx_data, "thesis.docx")

    def test_valid_text_formats(self):
        assert inspect_file_type(b"Plain text content", "notes.txt") == "text/plain"
        assert inspect_file_type(b"# Title\n\nMarkdown text", "readme.md") == "text/markdown"
        assert inspect_file_type(b'{"key": "value"}', "config.json") == "application/json"
        assert inspect_file_type(b"col1,col2\nval1,val2", "table.csv") == "text/csv"

    def test_reject_windows_pe_executable(self):
        exe_data = b"MZ\x90\x00\x03\x00\x00\x00This program cannot be run in DOS mode."
        with pytest.raises(SecurityValidationError, match="Windows executable"):
            inspect_file_type(exe_data, "app.exe")

    def test_reject_windows_pe_executable_renamed_to_pdf(self):
        """CRITICAL: Disguised executable named .pdf must be rejected."""
        disguised_data = b"MZ\x90\x00\x03\x00\x00\x00Fake PDF header"
        with pytest.raises(SecurityValidationError, match="Windows executable"):
            inspect_file_type(disguised_data, "financial_statement.pdf")

    def test_reject_linux_elf_executable(self):
        elf_data = b"\x7fELF\x02\x01\x01\x00"
        with pytest.raises(SecurityValidationError, match="Linux ELF"):
            inspect_file_type(elf_data, "binary_file.txt")

    def test_reject_shebang_script(self):
        script_data = b"#!/bin/bash\nrm -rf /"
        with pytest.raises(SecurityValidationError, match="executable script shebang"):
            inspect_file_type(script_data, "script.txt")

    def test_reject_text_with_null_bytes(self):
        corrupted_text = b"Normal text\x00Hidden binary injection\x00"
        with pytest.raises(SecurityValidationError, match="binary null bytes"):
            inspect_file_type(corrupted_text, "report.txt")

    def test_reject_misleading_extension(self):
        # File named .pdf but containing random non-PDF junk
        garbage = b"Not a real PDF at all"
        with pytest.raises(UnsupportedFormatError, match="content signature does not match"):
            inspect_file_type(garbage, "document.pdf")


# =====================================================================
# 3. Document Import, Listing, Export & Preservation Tests
# =====================================================================

class TestImportExportOperations:
    def test_import_and_list_documents(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "tax_return.pdf"
        src_content = b"%PDF-1.7\nSample Tax Document Content for Testing..."
        src_file.write_bytes(src_content)

        # Import document
        meta = manager.import_document(src_file, vmk)

        assert meta.doc_id is not None
        assert len(meta.doc_id) == 32  # 16-byte hex
        assert meta.original_filename == "tax_return.pdf"
        assert meta.file_size == len(src_content)
        assert meta.mime_type == "application/pdf"

        # Check ciphertext exists on disk with UUID filename
        cipher_path = manager.documents_dir / f"{meta.doc_id}.enc"
        assert cipher_path.exists()
        # Verify NO plaintext filename exists in storage directory
        assert not (manager.documents_dir / "tax_return.pdf").exists()

        # List documents (verifies manifest read without payload decryption)
        docs = manager.list_documents(vmk)
        assert len(docs) == 1
        assert docs[0].doc_id == meta.doc_id
        assert docs[0].original_filename == "tax_return.pdf"

    def test_source_file_preserved_untouched(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        """Verifies that original file is NOT modified, deleted, or moved."""
        src_file = tmp_path / "original_passport.png"
        src_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"Passport image data"
        src_file.write_bytes(src_content)
        orig_mtime = src_file.stat().st_mtime

        manager.import_document(src_file, vmk)

        # Original source must remain exactly intact
        assert src_file.exists()
        assert src_file.read_bytes() == src_content
        assert src_file.stat().st_mtime == orig_mtime

    def test_read_document_bytes_in_memory(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "notes.txt"
        src_content = b"Confidential meeting notes from board."
        src_file.write_bytes(src_content)

        meta = manager.import_document(src_file, vmk)

        # Read in-memory
        plaintext, ret_meta = manager.read_document_bytes(meta.doc_id, vmk)
        assert plaintext == src_content
        assert ret_meta.doc_id == meta.doc_id

    def test_export_document_roundtrip(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "contract.pdf"
        src_content = b"%PDF-1.7\nLegally binding contract clauses..."
        src_file.write_bytes(src_content)

        meta = manager.import_document(src_file, vmk)

        export_dir = tmp_path / "exported_docs"
        exported_path = manager.export_document(meta.doc_id, export_dir, vmk)

        assert exported_path.exists()
        assert exported_path.name == "contract.pdf"
        assert exported_path.read_bytes() == src_content

    def test_export_prevents_unauthorized_overwrite(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "statement.pdf"
        src_file.write_bytes(b"%PDF-1.7\nStatement")

        meta = manager.import_document(src_file, vmk)
        export_dir = tmp_path / "exports"
        manager.export_document(meta.doc_id, export_dir, vmk)

        # Attempt to export again without overwrite=True
        with pytest.raises(FileExistsError, match="already exists"):
            manager.export_document(meta.doc_id, export_dir, vmk, overwrite=False)

        # Export with overwrite=True succeeds
        re_exported = manager.export_document(meta.doc_id, export_dir, vmk, overwrite=True)
        assert re_exported.exists()

    def test_multiple_document_types_import(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        samples = [
            ("doc.pdf", b"%PDF-1.4\nTest PDF"),
            ("img.png", b"\x89PNG\r\n\x1a\nHeader"),
            ("photo.jpg", b"\xff\xd8\xff\xe0JFIF"),
            ("data.csv", b"a,b,c\n1,2,3"),
            ("config.json", b'{"status": "ok"}'),
            ("doc.docx", b"PK\x03\x04\x00\x00[Content_Types].xml docx"),
        ]
        doc_ids = []
        for name, data in samples:
            f = tmp_path / name
            f.write_bytes(data)
            meta = manager.import_document(f, vmk)
            doc_ids.append(meta.doc_id)

        docs = manager.list_documents(vmk)
        assert len(docs) == len(samples)
        for doc_id, (name, data) in zip(doc_ids, samples):
            read_bytes, _ = manager.read_document_bytes(doc_id, vmk)
            assert read_bytes == data


# =====================================================================
# 4. Tamper Resistance, Key Enforcement & AAD Mismatches
# =====================================================================

class TestTamperResistanceAndSecurity:
    def test_tampered_ciphertext_fails_closed(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "classified.pdf"
        src_file.write_bytes(b"%PDF-1.7\nClassified records")

        meta = manager.import_document(src_file, vmk)
        cipher_path = manager.documents_dir / f"{meta.doc_id}.enc"

        # Tamper with ciphertext
        data = bytearray(cipher_path.read_bytes())
        data[25] ^= 0xFF
        cipher_path.write_bytes(bytes(data))

        with pytest.raises(AuthenticationError):
            manager.read_document_bytes(meta.doc_id, vmk)

    def test_wrong_key_fails_closed(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "file.txt"
        src_file.write_bytes(b"Sensitive text")
        meta = manager.import_document(src_file, vmk)

        wrong_key = generate_master_key()
        # Listing with wrong key fails
        with pytest.raises(ManifestCorruptedError):
            manager.list_documents(wrong_key)

        # Reading with wrong key fails
        with pytest.raises(ManifestCorruptedError):
            manager.read_document_bytes(meta.doc_id, wrong_key)

    def test_ciphertext_swap_attack_rejected_by_aad(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        """
        Attacker swaps doc1.enc with doc2.enc on disk.
        Since AAD binds the doc_id and filename to the ciphertext,
        reading doc1 will fail authentication.
        """
        f1 = tmp_path / "file1.txt"
        f1.write_bytes(b"Document 1 text")
        f2 = tmp_path / "file2.txt"
        f2.write_bytes(b"Document 2 text")

        meta1 = manager.import_document(f1, vmk)
        meta2 = manager.import_document(f2, vmk)

        c1 = manager.documents_dir / f"{meta1.doc_id}.enc"
        c2 = manager.documents_dir / f"{meta2.doc_id}.enc"

        # Swap the files on disk
        c2_bytes = c2.read_bytes()
        c1.write_bytes(c2_bytes)

        # Reading doc1 must fail because doc2's ciphertext was encrypted with doc2's AAD
        with pytest.raises(AuthenticationError):
            manager.read_document_bytes(meta1.doc_id, vmk)


# =====================================================================
# 5. Manifest Integrity & Atomic Backup Recovery
# =====================================================================

class TestManifestRecovery:
    def test_manifest_backup_recovery_on_corruption(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src_file = tmp_path / "important.pdf"
        src_file.write_bytes(b"%PDF-1.7\nImportant data")

        meta = manager.import_document(src_file, vmk)
        assert manager.manifest_backup.exists()

        # Corrupt the primary manifest
        manager.manifest_file.write_bytes(b"GARBAGE_CORRUPTED_MANIFEST_DATA")

        # VaultManager should automatically recover from manifest.enc.bak
        docs = manager.list_documents(vmk)
        assert len(docs) == 1
        assert docs[0].doc_id == meta.doc_id

        # Verify primary manifest was restored
        assert manager.manifest_file.read_bytes() != b"GARBAGE_CORRUPTED_MANIFEST_DATA"

    def test_both_manifests_corrupted_raises_manifest_corrupted_error(self, manager: VaultManager, vmk: bytes):
        manager.manifest_file.write_bytes(b"GARBAGE_PRIMARY")
        manager.manifest_backup.write_bytes(b"GARBAGE_BACKUP")

        with pytest.raises(ManifestCorruptedError, match="Failed to decrypt vault manifest"):
            manager.list_documents(vmk)


# =====================================================================
# 6. Rename, Delete & Document Lifecycle Tests
# =====================================================================

class TestDocumentLifecycle:
    def test_rename_document_updates_manifest_and_aad(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src = tmp_path / "initial.txt"
        src.write_bytes(b"File content to be renamed")
        meta = manager.import_document(src, vmk)

        # Rename
        renamed_meta = manager.rename_document(meta.doc_id, "updated_title.txt", vmk)
        assert renamed_meta.original_filename == "updated_title.txt"

        # Verify manifest reflects new name
        docs = manager.list_documents(vmk)
        assert docs[0].original_filename == "updated_title.txt"

        # Verify decryption works with new AAD
        plaintext, _ = manager.read_document_bytes(meta.doc_id, vmk)
        assert plaintext == b"File content to be renamed"

    def test_delete_document(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        src = tmp_path / "to_delete.pdf"
        src.write_bytes(b"%PDF-1.7\nDelete me")
        meta = manager.import_document(src, vmk)

        cipher_file = manager.documents_dir / f"{meta.doc_id}.enc"
        assert cipher_file.exists()

        # Delete
        success = manager.delete_document(meta.doc_id, vmk)
        assert success is True

        # Manifest must be empty
        assert len(manager.list_documents(vmk)) == 0
        # Ciphertext file removed
        assert not cipher_file.exists()

        # Attempting to read deleted document raises DocumentNotFoundError
        with pytest.raises(DocumentNotFoundError):
            manager.read_document_bytes(meta.doc_id, vmk)

    def test_delete_nonexistent_document_raises_not_found(self, manager: VaultManager, vmk: bytes):
        with pytest.raises(DocumentNotFoundError, match="not found in vault"):
            manager.delete_document("nonexistent_id", vmk)


# =====================================================================
# 7. Size Limits & Key Authorization Interfaces
# =====================================================================

class TestSizeLimitsAndKeyHolder:
    def test_file_size_limit_exceeded(self, vault_dir: Path, vmk: bytes, tmp_path: Path):
        # Create manager with a strict 1 KiB limit
        strict_manager = VaultManager(vault_root=vault_dir, max_file_size=1024)

        big_file = tmp_path / "large_file.txt"
        big_file.write_bytes(b"A" * 2048)  # 2 KiB

        with pytest.raises(FileSizeLimitExceededError, match="exceeds maximum vault limit"):
            strict_manager.import_document(big_file, vmk)

    def test_secure_key_holder_authorization_interface(self, manager: VaultManager, vmk: bytes, tmp_path: Path):
        holder = SecureKeyHolder()
        holder.unlock(vmk)

        src = tmp_path / "authorized.txt"
        src.write_bytes(b"Session authorized access")

        # Import using SecureKeyHolder
        meta = manager.import_document(src, holder)
        assert meta.doc_id is not None

        # List using SecureKeyHolder
        docs = manager.list_documents(holder)
        assert len(docs) == 1

        # Lock the holder
        holder.lock()
        assert holder.is_locked is True

        # Attempting operations with locked holder must raise VaultLockedError
        with pytest.raises(VaultLockedError):
            manager.list_documents(holder)

        with pytest.raises(VaultLockedError):
            manager.read_document_bytes(meta.doc_id, holder)
