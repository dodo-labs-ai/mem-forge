from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.genes.jira_gene import JiraGene
from memforge.models import ChangelogEntry
from memforge.storage.database import Database
from tests.gene_fixture import FakeGene
from tests.test_sync_bookkeeping import _hold_document

SINCE = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "store.db"))
    await database.connect()
    yield database
    await database.close()


async def _source_with_unit(db: Database, *, source_id: str, source_type: str, doc_id: str, pdf: str | None) -> None:
    await db.upsert_source(source_id, source_type, source_id, "{}", access_policy="workspace", owner_user_id="dev")
    await _hold_document(
        db,
        source_id=source_id,
        source_type=source_type,
        doc_id=doc_id,
        title=doc_id,
        markdown=f"# {doc_id}",
        version="1",
        source_url=f"https://provider.example/{doc_id}",
        space_or_project="PAY",
        normalized_content_uri=f"/tmp/{doc_id}.md",
        pdf_content_uri=pdf,
    )


async def test_missing_pdfs_are_counted_by_source_and_listed_by_type(db: Database) -> None:
    await _source_with_unit(db, source_id="src-wiki", source_type="confluence", doc_id="page-1", pdf=None)
    await _source_with_unit(db, source_id="src-repo", source_type="github_repo", doc_id="file-1", pdf=None)
    await _source_with_unit(db, source_id="src-wiki-2", source_type="confluence", doc_id="page-2", pdf="/tmp/p2.pdf")

    # The store answers for any Source it is asked about; the caller asks only
    # for a type that exports PDFs.
    assert await db.count_missing_pdf_uris("src-wiki") == 1
    assert await db.count_missing_pdf_uris("src-repo") == 1
    assert await db.count_missing_pdf_uris("src-wiki-2") == 0
    assert await db.list_documents_missing_pdf(["confluence"]) == {"page-1"}
    assert await db.list_documents_missing_pdf([]) == set()


async def test_a_changelog_entry_is_written_through_the_store(db: Database) -> None:
    detected_at = datetime(2026, 10, 22, 9, 30, tzinfo=timezone.utc)
    await db.upsert_source("src-wiki", "confluence", "Wiki", "{}", access_policy="workspace", owner_user_id="dev")
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("page-1", "src-wiki", "https://wiki.example/page-1", "Payroll", "PAY",
         detected_at.isoformat(), "2", "hash-2", detected_at.isoformat()),
    )
    await db.db.commit()
    await db.insert_changelog(
        ChangelogEntry(
            id=None,
            doc_id="page-1",
            change_type="updated",
            previous_version="1",
            current_version="2",
            content_diff="- old\n+ new",
            ai_change_summary=None,
            detected_at=detected_at,
            title="Payroll",
            source="src-wiki",
        )
    )

    async with db.db.execute(
        "SELECT doc_id, change_type, previous_version, current_version, detected_at, source FROM changelog"
    ) as cursor:
        rows = [tuple(row) async for row in cursor]
    assert rows == [("page-1", "updated", "1", "2", detected_at.isoformat(), "src-wiki")]


def test_the_filter_summary_names_the_discovery_window() -> None:
    gene = FakeGene()

    assert gene.filter_summary(None) is None
    assert gene.filter_summary(SINCE) == f"updated >= {SINCE.isoformat()}"


def test_jira_adds_its_jql_filter_to_the_summary() -> None:
    gene = JiraGene({"base_url": "https://jira.example", "jql_filter": " status != Done "}, "src-jira")

    assert gene.filter_summary(SINCE) == f"status != Done AND updated >= {SINCE.isoformat()}"
    assert gene.filter_summary(None) == "status != Done"
    assert JiraGene({"base_url": "https://jira.example"}, "src-jira").filter_summary(None) is None
