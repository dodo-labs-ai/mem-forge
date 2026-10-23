"""A local folder collected only on its owner's device; the server never reads it.

Its listing is the snapshot manifest the device submits: here three notes,
one edited in the cursor's minute.
"""

from __future__ import annotations

from collections.abc import Mapping

from tests.contracts.document_fakes import IN_THE_CURSOR_MINUTE, LONG_AGO, RECENTLY, DocumentConnectorFakes


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
            for name, changed_at in (("ideas", LONG_AGO), ("plans", RECENTLY), ("standup", IN_THE_CURSOR_MINUTE))
        ]


FAKES = LocalMarkdownFakes()
