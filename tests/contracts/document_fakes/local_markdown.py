"""A local folder collected only on its owner's device; the server never reads it.

Its listing is the snapshot manifest the device submits: here two notes.
"""

from __future__ import annotations

from collections.abc import Mapping

from tests.contracts.document_fakes import LONG_AGO, RECENTLY, DocumentConnectorFakes


class LocalMarkdownFakes(DocumentConnectorFakes):
    device_configs = {"folder": {"root": "/Users/me/notes"}}

    def manifest_entries(self, config: Mapping[str, object]) -> list[dict]:
        return [
            {
                "doc_id": f"local-markdown-{name}",
                "relative_path": f"{name}.md",
                "title": name,
                "version": f"sha-{name}",
                "last_modified": changed_at.isoformat(),
                "change_kind": "upsert",
            }
            for name, changed_at in (("ideas", LONG_AGO), ("plans", RECENTLY))
        ]


FAKES = LocalMarkdownFakes()
