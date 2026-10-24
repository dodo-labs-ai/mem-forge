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

    On create, update, scope-change preview and config-check, ``complete``
    first fills values derived from the Source or its stored configuration.
    The admin API then runs, in order:

    1. ``validate_input``, on the configuration as given. It refuses only
       what the input itself shows, such as an unsupported key; it never
       refuses a missing value and never derives one.
    2. The TLS CA bundle check.
    3. The Gene's ``normalize_config``, which shapes and derives values.
    4. The shared form check (:func:`memforge.source_types.form.check_config`):
       required while visible, choices, list shapes and booleans.
    5. The secret-origin guard: a type with a secret needs an HTTPS
       ``base_url``.
    6. ``validate``, on the normalized configuration, for the rules a form
       description cannot state, such as a URL that must stay on a site.

    A hook refuses with ``SourceConfigInvalid`` naming the field
    (:class:`memforge.genes.toolkit.config.SourceConfigInvalid`), or with a
    plain ``ValueError`` when no single field is at fault. ``store`` adjusts
    the configuration after its secrets are encrypted. ``public`` shapes what
    a reader of the Source sees. ``credentials_changed`` says an update
    replaced the credential a running sync uses, so that sync is cancelled.
    ``facts`` states what a person should see about the normalized
    configuration without it being stored, as ``(label, value)`` pairs that
    config-check returns, for example ``[("Repository", "acme/handbook")]``;
    it reads a configuration that may still be incomplete, and states only
    what it can. ``extra_secret_fields`` are write-only fields the Gene's
    schema does not declare.
    """

    extra_secret_fields: tuple[str, ...] = ()
    complete: Callable[[dict[str, Any], str, Mapping[str, Any] | None], dict[str, Any]] | None = None
    validate_input: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    validate: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    store: Callable[[dict[str, Any], LocalInbox], dict[str, Any]] | None = None
    public: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    credentials_changed: Callable[[dict[str, Any], Mapping[str, Any]], bool] | None = None
    facts: Callable[[Mapping[str, Any]], list[tuple[str, str]]] | None = None


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
