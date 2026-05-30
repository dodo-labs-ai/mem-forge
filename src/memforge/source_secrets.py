"""Encrypted source-configuration secrets.

Source configs live in SQLite and are returned to the admin UI. Secret fields
therefore use authenticated encryption before persistence and write-only
redaction before display.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

SECRET_KEY_ENV = "MEMFORGE_SECRET_KEY"
SECRET_KEY_FILE_ENV = "MEMFORGE_SECRET_KEY_FILE"
ENCRYPTED_PREFIX = "enc:v1:"
SOURCE_SECRET_FIELDS = ("pat",)
DEFAULT_KEY_PATH = Path.home() / ".memforge" / "secrets" / "source-secrets.key"


class SecretConfigurationError(RuntimeError):
    """Raised when source secrets cannot be encrypted or decrypted safely."""


def source_secret_fields(source_type: str, gene_registry: dict[str, Any]) -> tuple[str, ...]:
    """Return the secret fields declared by a source's gene schema."""
    gene_cls = gene_registry.get(source_type)
    if gene_cls is None:
        return SOURCE_SECRET_FIELDS

    from memforge.models import ConfigFieldType

    return tuple(sorted({
        field.key
        for field in gene_cls.config_schema().fields
        if field.field_type == ConfigFieldType.SECRET
    }))


def prepare_source_config_for_storage(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
    secret_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Encrypt write-only source secrets before storing a source config."""
    prepared = dict(config)
    existing_config = existing_config or {}

    for field in _secret_fields(secret_fields):
        incoming_provided = field in prepared
        incoming = prepared.pop(field, None)
        encrypted_key = f"{field}_encrypted"
        configured_key = f"{field}_configured"
        hint_key = f"{field}_hint"

        if isinstance(incoming, str) and incoming.strip():
            secret = incoming.strip()
            existing_encrypted = existing_config.get(encrypted_key)
            if existing_encrypted and _encrypted_secret_matches(str(existing_encrypted), secret):
                prepared[encrypted_key] = existing_encrypted
                prepared[configured_key] = True
                prepared.pop(hint_key, None)
                continue
            prepared[encrypted_key] = encrypt_secret(secret)
            prepared[configured_key] = True
            prepared.pop(hint_key, None)
            continue

        if existing_config.get(encrypted_key):
            prepared[encrypted_key] = existing_config[encrypted_key]
            prepared[configured_key] = True
            prepared.pop(hint_key, None)
            continue

        legacy_plaintext = existing_config.get(field)
        if isinstance(legacy_plaintext, str) and legacy_plaintext.strip():
            secret = legacy_plaintext.strip()
            prepared[encrypted_key] = encrypt_secret(secret)
            prepared[configured_key] = True
            prepared.pop(hint_key, None)
            continue

        prepared.pop(encrypted_key, None)
        prepared.pop(hint_key, None)
        if incoming_provided or configured_key in existing_config:
            prepared[configured_key] = False
        else:
            prepared.pop(configured_key, None)

    return prepared


def decrypt_source_config_for_runtime(
    config: dict[str, Any],
    secret_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a runtime-only config with encrypted secrets decrypted."""
    runtime_config = dict(config)

    for field in _secret_fields(secret_fields):
        encrypted_key = f"{field}_encrypted"
        configured_key = f"{field}_configured"
        hint_key = f"{field}_hint"
        encrypted = runtime_config.pop(encrypted_key, None)
        runtime_config.pop(configured_key, None)
        runtime_config.pop(hint_key, None)
        if encrypted:
            runtime_config[field] = decrypt_secret(str(encrypted))

    return runtime_config


def redact_source_config(
    config: dict[str, Any],
    secret_fields: Iterable[str] | None = None,
    validate_encryption: bool = False,
) -> dict[str, Any]:
    """Return a source config safe for API responses."""
    redacted = dict(config)

    for field in _secret_fields(secret_fields):
        encrypted_key = f"{field}_encrypted"
        configured_key = f"{field}_configured"
        hint_key = f"{field}_hint"
        decrypt_failed_key = f"{field}_decrypt_failed"
        encrypted = redacted.get(encrypted_key)
        has_secret = bool(encrypted or redacted.get(field))
        redacted.pop(field, None)
        redacted.pop(encrypted_key, None)
        redacted.pop(hint_key, None)
        redacted.pop(decrypt_failed_key, None)
        if encrypted and validate_encryption:
            try:
                decrypt_secret(str(encrypted))
            except SecretConfigurationError:
                redacted[configured_key] = False
                redacted[decrypt_failed_key] = True
                continue
        configured = bool(redacted.get(configured_key) or has_secret)
        if configured or configured_key in redacted:
            redacted[configured_key] = configured

    return redacted


def encrypt_secret(secret: str) -> str:
    token = _fernet(allow_create=True).encrypt(secret.encode("utf-8")).decode("ascii")
    return f"{ENCRYPTED_PREFIX}{token}"


def decrypt_secret(encrypted: str) -> str:
    token = encrypted
    if token.startswith(ENCRYPTED_PREFIX):
        token = token[len(ENCRYPTED_PREFIX):]
    try:
        return _fernet(allow_create=False).decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise SecretConfigurationError("Source secret could not be decrypted with the configured key") from exc


def _encrypted_secret_matches(encrypted: str, secret: str) -> bool:
    try:
        return decrypt_secret(encrypted) == secret
    except SecretConfigurationError:
        return False


def _secret_fields(secret_fields: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(secret_fields) if secret_fields is not None else SOURCE_SECRET_FIELDS


def _fernet(*, allow_create: bool) -> Fernet:
    return Fernet(_key_material(allow_create=allow_create).encode("ascii"))


def _key_material(*, allow_create: bool) -> str:
    env_value = os.environ.get(SECRET_KEY_ENV, "").strip()
    if env_value:
        try:
            Fernet(env_value.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise SecretConfigurationError(
                f"{SECRET_KEY_ENV} must be a 32-byte url-safe base64 Fernet key"
            ) from exc
        return env_value

    key_path = _key_path()
    if key_path.exists():
        return _read_key_file(key_path)
    if allow_create:
        return _create_key_file(key_path)

    raise SecretConfigurationError(
        "Source secret key is missing; restore the key file or re-enter stored source PATs: "
        f"{key_path}"
    )


def _key_path() -> Path:
    configured_path = os.environ.get(SECRET_KEY_FILE_ENV, "").strip()
    if configured_path:
        return Path(configured_path).expanduser()

    base_dir = os.environ.get("MEMFORGE_BASE_DIR", "").strip()
    if base_dir:
        return Path(base_dir).expanduser() / "secrets" / "source-secrets.key"

    return DEFAULT_KEY_PATH


def _read_key_file(key_path: Path) -> str:
    key = key_path.read_text(encoding="ascii").strip()
    try:
        Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise SecretConfigurationError(
            f"Source secret key file must contain a 32-byte url-safe base64 Fernet key: {key_path}"
        ) from exc
    return key


def _create_key_file(key_path: Path) -> str:
    key_dir = key_path.parent
    key_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(key_dir, 0o700)

    key = Fernet.generate_key().decode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(key_path, flags, 0o600)
    except FileExistsError:
        return _read_key_file(key_path)
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(f"{key}\n")
    os.chmod(key_path, 0o600)
    return key
