"""Data security utilities: PII detection, audit logging, and artifact encryption."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = []


def _register(name: str, pattern: str, mask_template: str) -> None:
    _PII_PATTERNS.append((name, re.compile(pattern), mask_template))


# Chinese mobile: 1[3-9]XXXXXXXXX
_register("china_mobile", r"(?<!\d)1[3-9]\d{9}(?!\d)", "[MOBILE_****]")

# Chinese ID card (18 digits, last char can be X/x)
_register(
    "china_id_card",
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
    "[ID_****]",
)

# Bank card number: 16-19 digits with optional spaces/dashes between groups
_register(
    "bank_card",
    r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2,7}(?!\d)",
    "[BANK_****]",
)

# Email address
_register(
    "email",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "[EMAIL_****]",
)

# Chinese landline with area code: 0XXX-XXXXXXX
_register("china_landline", r"(?<!\d)0\d{2,3}-\d{7,8}(?!\d)", "[LANDLINE_****]")


# ---------------------------------------------------------------------------
# PII data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PIIMatch:
    """A single PII detection hit."""

    pattern_name: str
    start: int
    end: int
    matched_text: str
    masked_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "start": self.start,
            "end": self.end,
            "masked_text": self.masked_text,
        }


@dataclass(slots=True)
class PIIScanResult:
    """Result of scanning one text block for sensitive content."""

    original_length: int
    matches: list[PIIMatch]
    masked_text: str

    @property
    def has_pii(self) -> bool:
        return len(self.matches) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_length": self.original_length,
            "has_pii": self.has_pii,
            "match_count": len(self.matches),
            "matches": [m.to_dict() for m in self.matches],
        }


# ---------------------------------------------------------------------------
# PII scanning
# ---------------------------------------------------------------------------


def scan_and_mask(
    text: str,
    *,
    strict: bool = False,
    extra_patterns: Iterable[tuple[str, str, str]] | None = None,
) -> PIIScanResult:
    """Scan *text* for PII patterns and return masked version with match details.

    Args:
        text: The text to scan.
        strict: If True, apply additional strict patterns.
        extra_patterns: Optional ``(name, regex, mask_template)`` tuples.

    Returns:
        A :class:`PIIScanResult` with the masked text and all match details.
    """
    patterns = list(_PII_PATTERNS)
    if strict:
        patterns.append(
            ("financial_ratio", re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%"), "[RATIO_****]")
        )
    if extra_patterns:
        for name, regex, mask in extra_patterns:
            patterns.append((name, re.compile(regex), mask))

    # Collect all matches across all patterns
    all_matches: list[tuple[int, int, str, str]] = []  # (start, end, mask, name)
    for name, pattern, mask in patterns:
        for m in pattern.finditer(text):
            all_matches.append((m.start(), m.end(), mask, name))

    # Sort by start position, resolve overlaps (earliest start wins)
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered: list[tuple[int, int, str, str]] = []
    last_end = 0
    for start, end, mask, name in all_matches:
        if start >= last_end:
            filtered.append((start, end, mask, name))
            last_end = end

    # Build masked text and PIIMatch list
    pii_matches: list[PIIMatch] = []
    if not filtered:
        return PIIScanResult(original_length=len(text), matches=[], masked_text=text)

    parts: list[str] = []
    prev_end = 0
    for start, end, mask, name in filtered:
        parts.append(text[prev_end:start])
        parts.append(mask)
        pii_matches.append(
            PIIMatch(
                pattern_name=name,
                start=start,
                end=end,
                matched_text=text[start:end],
                masked_text=mask,
            )
        )
        prev_end = end
    parts.append(text[prev_end:])

    return PIIScanResult(
        original_length=len(text),
        matches=pii_matches,
        masked_text="".join(parts),
    )


def mask_for_api(
    prompt: str,
    evidence: dict[str, str],
    *,
    strict: bool = False,
) -> tuple[str, dict[str, str], PIIScanResult]:
    """Mask PII across a prompt and its evidence sections.

    Returns:
        ``(masked_prompt, masked_evidence_dict, combined_scan_result)``
    """
    total_matches: list[PIIMatch] = []
    total_original = 0

    prompt_result = scan_and_mask(prompt, strict=strict)
    total_matches.extend(prompt_result.matches)
    total_original += prompt_result.original_length

    masked_evidence: dict[str, str] = {}
    for key, text in evidence.items():
        ev_result = scan_and_mask(text, strict=strict)
        total_matches.extend(ev_result.matches)
        total_original += ev_result.original_length
        masked_evidence[key] = ev_result.masked_text

    combined = PIIScanResult(
        original_length=total_original,
        matches=total_matches,
        masked_text=prompt_result.masked_text,
    )
    return prompt_result.masked_text, masked_evidence, combined


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuditEntry:
    """One API interaction record for the audit log."""

    timestamp: str
    uid: str
    adapter_name: str
    action: str  # "predict" | "select" | "vote_pass"
    prompt_hash: str
    response_hash: str
    pii_masked: bool
    pii_match_count: int
    model: str
    prediction: str | None
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SecurityAuditLogger:
    """File-based append-only JSONL audit logger for API interactions."""

    log_path: Path
    enabled: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _file: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")

    def log(self, entry: AuditEntry) -> None:
        """Append an :class:`AuditEntry` as a single JSONL line."""
        if not self.enabled or self._file is None:
            return
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            with self._lock:
                self._file.close()
                self._file = None


# ---------------------------------------------------------------------------
# Artifact encryption
# ---------------------------------------------------------------------------


def derive_key(passphrase: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a 32-byte key from a passphrase using PBKDF2-HMAC-SHA256.

    Args:
        passphrase: User-supplied passphrase string.
        salt: Optional salt bytes. Generated randomly if not provided.

    Returns:
        ``(key_bytes, salt_bytes)``
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000)
    return key, salt


def encrypt_artifact(plaintext: str, key: bytes) -> bytes:
    """Encrypt *plaintext* using AES-256-GCM (if ``cryptography`` is installed)
    or a stdlib fallback.

    Args:
        plaintext: The artifact content.
        key: A 32-byte encryption key (from :func:`derive_key`).

    Returns:
        The encrypted bytes.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return b"AES:" + nonce + ciphertext
    except ImportError:
        pass

    # Stdlib fallback: XOR stream cipher using SHA-256 counter mode
    stream = _xor_stream(key, len(plaintext.encode("utf-8")))
    raw = plaintext.encode("utf-8")
    encrypted = bytes(a ^ b for a, b in zip(raw, stream))
    return b"XOR:" + raw[:0] + encrypted  # prefix marks the mode


def decrypt_artifact(ciphertext: bytes, key: bytes) -> str:
    """Decrypt an artifact encrypted by :func:`encrypt_artifact`.

    Args:
        ciphertext: The encrypted bytes.
        key: The same 32-byte key used for encryption.

    Returns:
        The original plaintext string.
    """
    if ciphertext.startswith(b"AES:"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        data = ciphertext[4:]
        nonce = data[:12]
        ct = data[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None).decode("utf-8")

    if ciphertext.startswith(b"XOR:"):
        data = ciphertext[4:]
        stream = _xor_stream(key, len(data))
        raw = bytes(a ^ b for a, b in zip(data, stream))
        return raw.decode("utf-8")

    raise ValueError("Unknown encryption format in ciphertext")


def _xor_stream(key: bytes, length: int) -> bytes:
    """Generate a deterministic XOR keystream using SHA-256 counter mode."""
    parts: list[bytes] = []
    counter = 0
    while len(b"".join(parts)) < length:
        block = hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
        parts.append(block)
        counter += 1
    return b"".join(parts)[:length]


# ---------------------------------------------------------------------------
# Security context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SecurityContext:
    """Carries security configuration through the evaluation loop."""

    pii_masking: bool = False
    audit_logging: bool = False
    encryption: bool = False
    encryption_key: bytes | None = None
    strict_pii: bool = False
    audit_logger: SecurityAuditLogger | None = None


def create_security_context(
    *,
    enable_pii_masking: bool = False,
    enable_audit_log: bool = False,
    enable_encryption: bool = False,
    encryption_passphrase: str | None = None,
    audit_log_path: str | Path | None = None,
    strict_pii: bool = False,
) -> SecurityContext:
    """Build a :class:`SecurityContext` from CLI flag combinations."""
    audit_logger: SecurityAuditLogger | None = None
    if enable_audit_log and audit_log_path:
        audit_logger = SecurityAuditLogger(log_path=Path(audit_log_path), enabled=True)

    encryption_key: bytes | None = None
    if enable_encryption:
        passphrase = encryption_passphrase or os.getenv("FINVL_ENCRYPTION_PASSPHRASE")
        if not passphrase:
            raise ValueError(
                "Encryption passphrase required: use --encryption-passphrase "
                "or set FINVL_ENCRYPTION_PASSPHRASE"
            )
        encryption_key, _ = derive_key(passphrase)

    return SecurityContext(
        pii_masking=enable_pii_masking,
        audit_logging=enable_audit_log,
        encryption=enable_encryption,
        encryption_key=encryption_key,
        strict_pii=strict_pii,
        audit_logger=audit_logger,
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
