"""Provider-neutral repository identity normalization."""

from __future__ import annotations

import re


def normalize_repo_identifier(repo: str | None) -> str | None:
    """Return a stable repository identity without transport-specific syntax.

    Remote URLs are normalized to ``host/org/repo`` without protocol, user, or
    ``.git`` suffix. Plain repository slugs are lower-cased and returned
    unchanged. Provider adapters remain responsible for validating that the
    input identifies a repository they can authoritatively read.
    """

    if repo is None:
        return None
    value = repo.strip()
    if not value:
        return None

    ssh_match = re.match(r"^[^/@]+@([^:/]+):(.+)$", value)
    if ssh_match:
        host, path = ssh_match.groups()
        value = f"{host}/{path}"
    else:
        value = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^[^@/]+@", "", value)

    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    value = re.sub(r"/+", "/", value)
    return value.lower() or None
