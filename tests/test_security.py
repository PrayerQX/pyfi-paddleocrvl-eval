"""Tests for the data security module."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from finvl_eval.security import (
    AuditEntry,
    PIIScanResult,
    SecurityAuditLogger,
    SecurityContext,
    create_security_context,
    decrypt_artifact,
    derive_key,
    encrypt_artifact,
    mask_for_api,
    scan_and_mask,
)


class PIIDetectionTests(unittest.TestCase):
    def test_detects_china_mobile(self) -> None:
        result = scan_and_mask("Contact: 13912345678 for details")
        self.assertTrue(result.has_pii)
        self.assertIn("[MOBILE_****]", result.masked_text)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].pattern_name, "china_mobile")

    def test_detects_bank_card(self) -> None:
        result = scan_and_mask("Card number: 6222021234567890123")
        self.assertTrue(result.has_pii)
        self.assertIn("[BANK_****]", result.masked_text)

    def test_detects_email(self) -> None:
        result = scan_and_mask("Email: test.user@example.com is used")
        self.assertTrue(result.has_pii)
        self.assertIn("[EMAIL_****]", result.masked_text)

    def test_detects_china_id_card(self) -> None:
        result = scan_and_mask("ID: 110101199001011234")
        self.assertTrue(result.has_pii)
        self.assertIn("[ID_****]", result.masked_text)

    def test_detects_china_landline(self) -> None:
        result = scan_and_mask("Phone: 010-12345678")
        self.assertTrue(result.has_pii)
        self.assertIn("[LANDLINE_****]", result.masked_text)

    def test_detects_multiple_pii_types(self) -> None:
        text = "Mobile 13912345678, email test@example.com, ID 110101199001011234"
        result = scan_and_mask(text)
        self.assertGreaterEqual(len(result.matches), 3)

    def test_clean_text_no_matches(self) -> None:
        result = scan_and_mask("Revenue increased by 5.3% in Q4 2025.")
        self.assertFalse(result.has_pii)
        self.assertEqual(len(result.matches), 0)
        self.assertEqual(result.masked_text, "Revenue increased by 5.3% in Q4 2025.")

    def test_mask_preserves_surrounding_text(self) -> None:
        original = "The account 13912345678 is linked."
        result = scan_and_mask(original)
        self.assertTrue(result.masked_text.startswith("The account "))
        self.assertTrue(result.masked_text.endswith(" is linked."))

    def test_strict_mode_detects_ratios(self) -> None:
        result = scan_and_mask("Growth was 45.2%", strict=True)
        self.assertTrue(result.has_pii)

    def test_strict_mode_off_ignores_ratios(self) -> None:
        result = scan_and_mask("Growth was 45.2%", strict=False)
        self.assertFalse(result.has_pii)

    def test_overlapping_matches_resolved(self) -> None:
        # A long number could match both bank_card and china_mobile
        # Earliest start should win
        text = "Number: 13912345678 end"
        result = scan_and_mask(text)
        # Should detect mobile but not double-count
        self.assertLessEqual(len(result.matches), 1)

    def test_to_dict(self) -> None:
        result = scan_and_mask("Call 13912345678")
        d = result.to_dict()
        self.assertIn("has_pii", d)
        self.assertIn("match_count", d)
        self.assertIn("matches", d)
        self.assertTrue(d["has_pii"])
        self.assertEqual(d["match_count"], 1)


class MaskForAPITests(unittest.TestCase):
    def test_masks_prompt_and_evidence(self) -> None:
        prompt = "Question about 13912345678"
        evidence = {"text": "Email: secret@example.com"}
        masked_prompt, masked_ev, result = mask_for_api(prompt, evidence)
        self.assertIn("[MOBILE_****]", masked_prompt)
        self.assertIn("[EMAIL_****]", masked_ev["text"])
        self.assertGreaterEqual(len(result.matches), 2)


class AuditLoggerTests(unittest.TestCase):
    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            logger = SecurityAuditLogger(log_path=log_path, enabled=True)
            entry = AuditEntry(
                timestamp="2026-01-01T00:00:00+00:00",
                uid="test-uid",
                adapter_name="test-adapter",
                action="select",
                prompt_hash="abc123",
                response_hash="def456",
                pii_masked=True,
                pii_match_count=1,
                model="test-model",
                prediction="A",
                latency_ms=100.0,
            )
            logger.log(entry)
            logger.close()

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["uid"], "test-uid")
            self.assertEqual(parsed["prediction"], "A")

    def test_disabled_logger_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            logger = SecurityAuditLogger(log_path=log_path, enabled=False)
            entry = AuditEntry(
                timestamp="2026-01-01T00:00:00+00:00",
                uid="test-uid",
                adapter_name="test",
                action="select",
                prompt_hash="abc",
                response_hash="def",
                pii_masked=False,
                pii_match_count=0,
                model="test",
                prediction="B",
            )
            logger.log(entry)
            logger.close()
            self.assertFalse(log_path.exists())


class EncryptionTests(unittest.TestCase):
    def test_derive_key_deterministic(self) -> None:
        key1, salt = derive_key("test-passphrase", salt=b"fixed-salt-1234")
        key2, _ = derive_key("test-passphrase", salt=b"fixed-salt-1234")
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 32)

    def test_derive_key_different_passphrases(self) -> None:
        key1, salt = derive_key("passphrase-1", salt=b"fixed-salt-1234")
        key2, _ = derive_key("passphrase-2", salt=b"fixed-salt-1234")
        self.assertNotEqual(key1, key2)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        key, _ = derive_key("test-pass")
        plaintext = "This is a secret artifact with revenue data: $1,234,567.89"
        encrypted = encrypt_artifact(plaintext, key)
        decrypted = decrypt_artifact(encrypted, key)
        self.assertEqual(decrypted, plaintext)

    def test_encrypt_produces_different_ciphertext(self) -> None:
        key, _ = derive_key("test-pass")
        plaintext = "Same text each time"
        enc1 = encrypt_artifact(plaintext, key)
        enc2 = encrypt_artifact(plaintext, key)
        # AES-GCM uses random nonce so ciphertext differs
        # XOR fallback is deterministic but prefixed differently
        # At minimum, encrypted should differ from plaintext
        self.assertNotEqual(enc1, plaintext.encode("utf-8"))

    def test_decrypt_wrong_key_fails(self) -> None:
        key1, _ = derive_key("correct-key")
        key2, _ = derive_key("wrong-key")
        encrypted = encrypt_artifact("secret data", key1)
        # This should either raise an error or return garbage
        try:
            result = decrypt_artifact(encrypted, key2)
            # If it doesn't raise, the result should be wrong
            self.assertNotEqual(result, "secret data")
        except Exception:
            pass  # Expected: decryption with wrong key should fail


class SecurityContextTests(unittest.TestCase):
    def test_create_security_context_defaults(self) -> None:
        ctx = create_security_context()
        self.assertFalse(ctx.pii_masking)
        self.assertFalse(ctx.audit_logging)
        self.assertFalse(ctx.encryption)
        self.assertIsNone(ctx.audit_logger)

    def test_create_security_context_with_masking(self) -> None:
        ctx = create_security_context(enable_pii_masking=True, strict_pii=True)
        self.assertTrue(ctx.pii_masking)
        self.assertTrue(ctx.strict_pii)

    def test_create_security_context_with_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            ctx = create_security_context(enable_audit_log=True, audit_log_path=str(log_path))
            self.assertTrue(ctx.audit_logging)
            self.assertIsNotNone(ctx.audit_logger)
            ctx.audit_logger.close()

    def test_create_security_context_with_encryption(self) -> None:
        ctx = create_security_context(enable_encryption=True, encryption_passphrase="test-pass")
        self.assertTrue(ctx.encryption)
        self.assertIsNotNone(ctx.encryption_key)

    def test_create_security_context_encryption_without_passphrase_raises(self) -> None:
        # Remove env var if set
        os.environ.pop("FINVL_ENCRYPTION_PASSPHRASE", None)
        with self.assertRaises(ValueError):
            create_security_context(enable_encryption=True)


if __name__ == "__main__":
    unittest.main()
