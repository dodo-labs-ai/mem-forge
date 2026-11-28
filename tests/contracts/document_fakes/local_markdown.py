"""A fake local folder: the files on the Source's execution device.

The folder holds three notes, ``ideas`` changed long ago, ``team/plans``
recently and ``team/standup`` in the cursor's minute, and a ``drafts`` folder,
an Obsidian settings file and an image the configuration leaves out. Each
directory the walk enumerates is a request to this provider. A truncated
listing is a folder with a subdirectory the walk may not enumerate; a failed
one is a folder that is gone.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock

from memforge.genes import local_markdown_gene
from memforge.genes.local_markdown_gene import LocalMarkdownGene
from memforge.local_agent.document_identity import build_local_markdown_doc_id
from tests.contracts.document_fakes import (
    IN_THE_CURSOR_MINUTE,
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)

SOURCE_ID = "src-contract"
VAULT_ID = "notes"
# relative path -> changed at
NOTES: dict[str, datetime] = {"ideas.md": LONG_AGO, "team/plans.md": RECENTLY, "team/standup.md": IN_THE_CURSOR_MINUTE}
OUTSIDE = {"drafts/wip.md": RECENTLY, ".obsidian/workspace.json": RECENTLY, "team/diagram.png": RECENTLY}


def _write(root: Path, files: Mapping[str, datetime]) -> None:
    for relative_path, changed_at in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {Path(relative_path).stem}\n\nNotes.\n")
        os.utime(path, (changed_at.timestamp(), changed_at.timestamp()))


class LocalMarkdownFakes(DocumentConnectorFakes):
    opens_clients = False
    listing_configs = {
        "folder": ListingConfig(
            config={"root": "/Users/me/notes", "vault_id": VAULT_ID, "exclude": ["drafts/**", ".obsidian/**"]},
            walks_scope=True,
            edited_in_the_cursor_minute=build_local_markdown_doc_id(
                source_id=SOURCE_ID, vault_id=VAULT_ID, relative_path="team/standup.md"
            ),
        )
    }

    @asynccontextmanager
    async def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AsyncIterator[FakeConnection]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "notes"
            _write(root, {**NOTES, **OUTSIDE})
            if listing is ProviderListing.TRUNCATED:
                (root / "team" / "locked").mkdir()
            gene = LocalMarkdownGene(
                config={**config, "root": str(root / "gone" if listing is ProviderListing.FAILED else root)},
                source_id=SOURCE_ID,
            )
            connection = FakeConnection(gene=gene)
            enumerate_directory = os.scandir

            def scandir(path):
                connection.requests.append(f"LIST {Path(path).relative_to(root).as_posix()}")
                if Path(path).name == "locked":
                    raise PermissionError(f"Permission denied: {path}")
                return enumerate_directory(path)

            with mock.patch.object(local_markdown_gene.os, "scandir", scandir):
                yield connection
            await gene.close()


FAKES = LocalMarkdownFakes()
