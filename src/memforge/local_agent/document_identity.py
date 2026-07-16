"""Stable document identities for local-agent package adapters."""

from __future__ import annotations

import hashlib

from memforge.models import slugify


def build_local_markdown_doc_id(
    *,
    source_id: str,
    vault_id: str,
    relative_path: str,
) -> str:
    identity = "|".join([source_id.strip(), vault_id.strip(), relative_path.strip()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join(
        [
            "local-md",
            slugify(source_id)[:30],
            slugify(relative_path)[:50] or "doc",
            digest,
        ]
    )


def build_jira_doc_id(*, source_id: str, issue_key: str) -> str:
    identity = "|".join([source_id.strip(), issue_key.strip().upper()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join(
        [
            "jira",
            slugify(source_id)[:30],
            slugify(issue_key)[:30] or "issue",
            digest,
        ]
    )


def build_teams_doc_id(*, source_id: str, window_id: str) -> str:
    identity = "|".join([source_id.strip(), window_id.strip()])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "-".join(
        [
            "teams",
            slugify(source_id)[:30],
            slugify(window_id)[:50] or "window",
            digest,
        ]
    )
