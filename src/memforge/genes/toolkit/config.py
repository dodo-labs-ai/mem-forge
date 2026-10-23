"""Read a connector's configuration values the same way in every connector.

A Source's configuration arrives as a plain ``dict`` of what the user saved,
so a value may be a real boolean, a number or a string. These helpers turn
those values into what a connector needs, and refuse a value that cannot mean
anything::

    from memforge.genes.toolkit.config import bool_config, int_config, list_config

    include_comments = bool_config(self.config, "include_comments", default=True)
    max_pages = int_config(self.config, "max_pages", default=200)
    spaces = list_config(self.config.get("spaces"))
"""

from __future__ import annotations

import os
import ssl
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from memforge.genes.base import SourceConfigurationError
from memforge.models import ConfigField, ConfigFieldType


@dataclass(frozen=True)
class FieldIssue:
    """Why one configuration field is refused; ``key`` is ``None`` when no single field is at fault."""

    key: str | None
    message: str


class SourceConfigInvalid(ValueError):
    """A Source configuration refused field by field.

    Any configuration hook of a source type (``complete``, ``validate_input``,
    ``normalize_config``, ``validate``) may raise it to name the field it
    refuses, so the admin UI shows the reason on that field::

        raise SourceConfigInvalid([FieldIssue("page_url", "GitHub Pages Page URL must stay on the site origin")])

    The admin API answers it as 400 with ``detail`` ``{"code": "config_invalid",
    "message", "fields": [{"key", "message"}]}``, where ``message`` is the first
    issue's. A plain ``ValueError`` from a hook is answered the same way, as one
    issue with ``key`` ``None``. ``str()`` of the exception is the first issue's
    message, so a caller that catches ``ValueError`` still reads a reason.
    """

    def __init__(self, issues: Iterable[FieldIssue]) -> None:
        self.issues: tuple[FieldIssue, ...] = tuple(issues)
        if not self.issues:
            raise ValueError("SourceConfigInvalid needs at least one issue")
        super().__init__(self.issues[0].message)


_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})


def bool_config(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    """Read one boolean field, or raise ``SourceConfigurationError`` for a value that is not a boolean.

    Accepted values:

    - ``True`` and ``False``;
    - the integers ``1`` and ``0``;
    - the strings ``true``/``false``, ``yes``/``no``, ``on``/``off`` and
      ``1``/``0``, in any case and with surrounding whitespace ignored.

    A missing field, ``None`` and an empty string read as ``default``::

        bool_config({"include_comments": "No"}, "include_comments", default=True)   # False
        bool_config({"include_comments": ""}, "include_comments", default=True)     # True
        bool_config({}, "sitemap_authoritative", default=False)                     # False
        bool_config({"include_comments": "maybe"}, "include_comments", default=True)
        # raises SourceConfigurationError: include_comments must be true or false ...

    A field declared as ``ConfigFieldType.BOOLEAN`` in a document connector's
    ``config_schema()`` is already checked when the Gene is created, so a run
    never starts with a value this function refuses.
    """
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return value == 1
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise SourceConfigurationError(
        f"{key} must be true or false (also accepted: yes/no, on/off, 1/0); got {value!r}"
    )


def validate_bool_fields(config: Mapping[str, Any], fields: Iterable[ConfigField]) -> None:
    """Check every ``BOOLEAN`` field of a config schema with :func:`bool_config`.

    Pass the fields a Gene declares, so a new boolean field is checked as soon
    as it is declared::

        validate_bool_fields(config, JiraGene.config_schema().fields)

    Raises ``SourceConfigurationError`` naming the first field whose value is
    not a boolean.
    """
    for field in fields:
        if field.field_type == ConfigFieldType.BOOLEAN:
            bool_config(config, field.key, default=False)


def int_config(config: Mapping[str, Any], key: str, *, default: int) -> int:
    """Read a positive integer field; a missing or unreadable value reads as ``default``.

    A value below 1 reads as 1::

        int_config({"max_files": "50"}, "max_files", default=500)   # 50
        int_config({"max_files": 0}, "max_files", default=500)      # 1
        int_config({"max_files": "lots"}, "max_files", default=500) # 500
    """
    try:
        return max(int(config.get(key, default)), 1)
    except (TypeError, ValueError):
        return default


def list_config(value: object) -> list[str]:
    """Read a list field given as a list or as a comma-separated string.

    Items are stripped and empty items dropped; any other value is an empty
    list::

        list_config(["PAY", " OPS ", ""])   # ["PAY", "OPS"]
        list_config("PAY, OPS,")            # ["PAY", "OPS"]
        list_config(None)                   # []
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def resolve_pat(config: Mapping[str, Any], product_name: str) -> str:
    """Return the personal access token from ``pat``, or from the environment variable ``pat_env_var`` names.

    ``product_name`` only words the error::

        resolve_pat({"pat": "abc"}, "Jira")                       # "abc"
        resolve_pat({"pat_env_var": "WIKI_TOKEN"}, "Confluence")  # value of $WIKI_TOKEN
        resolve_pat({}, "Jira")  # raises ValueError: Jira PAT is required
    """
    pat = str(config.get("pat") or "").strip()
    if pat:
        return pat

    env_var = str(config.get("pat_env_var") or "").strip()
    if env_var:
        env_pat = os.environ.get(env_var, "").strip()
        if env_pat:
            return env_pat

    raise ValueError(f"{product_name} PAT is required")


def bearer_headers(config: Mapping[str, Any], product_name: str) -> dict[str, str]:
    """Headers for a JSON API that takes the personal access token as a bearer token::

        bearer_headers({"pat": "abc"}, "Confluence")
        # {"Authorization": "Bearer abc", "Accept": "application/json"}
    """
    return {
        "Authorization": f"Bearer {resolve_pat(config, product_name)}",
        "Accept": "application/json",
    }


def require_https_base_url(base_url: str, product_name: str) -> None:
    """Refuse a base URL that would send stored credentials over plain HTTP::

        require_https_base_url("https://jira.example.com", "Jira")  # passes
        require_https_base_url("http://jira.example.com", "Jira")   # raises ValueError
    """
    if urlsplit(base_url).scheme != "https":
        raise ValueError(f"{product_name} base_url must use HTTPS when storing authenticated source credentials")


def validate_tls_ca_bundle(config: Mapping[str, Any]) -> None:
    """Refuse a ``tls_ca_bundle`` that is not a readable file; an empty field passes.

    ``~`` expands to the home directory of the user running MemForge.
    """
    ca_bundle = str(config.get("tls_ca_bundle") or "").strip()
    if not ca_bundle:
        return
    path = Path(ca_bundle).expanduser()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"TLS CA bundle does not exist or is not readable: {ca_bundle}")


def tls_ca_bundle(config: Mapping[str, Any], *, honor_requests_ca_bundle: bool = False) -> str | None:
    """The CA bundle a connector trusts, or ``None`` for the default trust store.

    An explicit ``tls_ca_bundle`` comes first, and is checked with
    :func:`validate_tls_ca_bundle`. With ``honor_requests_ca_bundle=True``
    the ``REQUESTS_CA_BUNDLE`` environment variable comes next, for a
    connector whose users already point it at a private CA that way::

        tls_ca_bundle({"tls_ca_bundle": "/etc/ssl/corp-ca.pem"})  # "/etc/ssl/corp-ca.pem"
        tls_ca_bundle({})                                         # None
    """
    validate_tls_ca_bundle(config)
    ca_bundle = str(config.get("tls_ca_bundle") or "").strip()
    if ca_bundle:
        return str(Path(ca_bundle).expanduser())
    if honor_requests_ca_bundle:
        env_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
        if env_bundle:
            return env_bundle
    return None


def tls_verify(config: Mapping[str, Any], *, honor_requests_ca_bundle: bool = False) -> ssl.SSLContext | bool:
    """The ``verify`` argument for an ``httpx`` client.

    A configured CA bundle (see :func:`tls_ca_bundle`) becomes an
    ``ssl.SSLContext`` that trusts it. Without one this returns ``True``, and
    httpx uses its default trust store, honouring ``SSL_CERT_FILE`` and
    ``SSL_CERT_DIR``::

        httpx.AsyncClient(base_url=base_url, verify=tls_verify(self.config))
    """
    ca_bundle = tls_ca_bundle(config, honor_requests_ca_bundle=honor_requests_ca_bundle)
    if ca_bundle is None:
        return True
    if os.path.isdir(ca_bundle):
        return ssl.create_default_context(capath=ca_bundle)
    return ssl.create_default_context(cafile=ca_bundle)
