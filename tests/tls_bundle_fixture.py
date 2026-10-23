"""A small CA bundle for TLS tests, and a way to read which CAs an ``ssl.SSLContext`` trusts."""

from __future__ import annotations

import re
import ssl
from pathlib import Path

import certifi

BUNDLE_CA_COUNT = 2
"""The number of certificates :func:`write_ca_bundle` writes, far fewer than any default trust store."""


def write_ca_bundle(path: Path) -> Path:
    """Write the first :data:`BUNDLE_CA_COUNT` certificates of certifi's bundle to ``path``."""
    pems = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----\n",
        Path(certifi.where()).read_text(encoding="ascii"),
        re.S,
    )
    path.write_text("".join(pems[:BUNDLE_CA_COUNT]), encoding="ascii")
    return path


def trusted_ca_count(context: ssl.SSLContext) -> int:
    """The number of CA certificates ``context`` has loaded.

    ``pip-system-certs`` replaces ``ssl.SSLContext`` with truststore's wrapper,
    which keeps the loaded certificates in its inner ``_ctx``.
    """
    return getattr(context, "_ctx", context).cert_store_stats()["x509_ca"]
