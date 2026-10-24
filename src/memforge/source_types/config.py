"""How a source type completes, checks and stores its configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

LocalInbox = Callable[[], str]
"""Create this Source's local-agent inbox and return its path."""


@dataclass(frozen=True)
class SourceConfigPolicy:
    """The per-type steps the admin API runs on a Source's configuration.

    On create and update, ``complete`` fills values derived from the Source or
    its stored configuration, and ``store`` adjusts the configuration after its
    secrets are encrypted. Two checks raise ``ValueError`` for an unusable
    configuration: ``validate_input`` sees it as the user gave it, before the
    Gene normalizes it and the shared checks run, and ``validate`` sees the
    normalized configuration after them. ``public`` shapes what a reader of
    the Source sees. ``credentials_changed`` says an update replaced the
    credential a running sync uses, so that sync is cancelled.
    ``extra_secret_fields`` are write-only fields the Gene's schema does not
    declare.
    """

    extra_secret_fields: tuple[str, ...] = ()
    complete: Callable[[dict[str, Any], str, Mapping[str, Any] | None], dict[str, Any]] | None = None
    validate_input: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    validate: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    store: Callable[[dict[str, Any], LocalInbox], dict[str, Any]] | None = None
    public: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    credentials_changed: Callable[[dict[str, Any], Mapping[str, Any]], bool] | None = None


def secret_has_value(field: str, config: Mapping[str, Any], existing_config: Mapping[str, Any]) -> bool:
    """Whether a write-only secret is supplied now or already stored."""

    value = config.get(field)
    if isinstance(value, str) and value.strip():
        return True
    return bool(
        config.get(f"{field}_encrypted")
        or existing_config.get(f"{field}_encrypted")
        or (isinstance(existing_config.get(field), str) and existing_config[field].strip())
    )


def secret_replaced(field: str, config: Mapping[str, Any], existing_config: Mapping[str, Any]) -> bool:
    """Whether an update supplies a write-only secret other than the one stored.

    A blank or absent incoming value keeps the stored secret, so it replaces
    nothing; a stored secret that cannot be read counts as replaced::

        secret_replaced("token", {"token": "new"}, {"token_encrypted": "<old, encrypted>"})  # True
        secret_replaced("token", {"token": ""}, {"token_encrypted": "<old, encrypted>"})     # False
    """

    incoming = config.get(field)
    if not isinstance(incoming, str) or not incoming.strip():
        return False
    return stored_secret_value(field, existing_config) != incoming.strip()


def stored_secret_value(field: str, config: Mapping[str, Any]) -> str | None:
    """The stored value of a write-only secret, decrypted; ``None`` when none is stored or it cannot be read."""

    from memforge.source_secrets import SecretConfigurationError, decrypt_secret

    encrypted = config.get(f"{field}_encrypted")
    if encrypted:
        try:
            return decrypt_secret(str(encrypted))
        except SecretConfigurationError:
            return None
    plaintext = config.get(field)
    if isinstance(plaintext, str) and plaintext.strip():
        return plaintext.strip()
    return None


def list_has_value(value: Any) -> bool:
    """Whether a list field, given as a list or a comma-separated string, names anything."""

    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    if isinstance(value, str):
        return any(part.strip() for part in value.split(","))
    return False
