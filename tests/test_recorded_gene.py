"""The recorded Gene: a type's own Gene whose acquisition is answered from an acquisition record.

Acquisition ran elsewhere and left a record; the recorded Gene answers
``discover``, ``list_scope``, ``confirm_absent``, ``rediscover``, ``fetch`` and
the artifacts from it, and interprets with the type's own code. A question
the record cannot answer fails the run, not one Document.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.genes.recorded import (
    AcquiredInput,
    AcquisitionRecordMismatch,
    recorded_gene_class,
)
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, ScopeListingKind
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload
from memforge.storage.database import Database
from tests.test_document_absence import JIRA_CONFIG, SOURCE_ID, Workspace
from tests.test_sync_bookkeeping import BlockingFetchGene, NoopMemoryEngine, _jira_raw_content

UNREACHABLE = "a recorded Gene never reaches the provider"


class JiraIssues(BlockingFetchGene):
    """The type's Gene: interpretation of its own, and acquisition that must never run."""

    listing_kinds = frozenset({ScopeListingKind.QUERY})

    def __init__(self, config: dict, source_id: str) -> None:
        release = asyncio.Event()
        release.set()
        super().__init__(item_count=0, release=release)
        self.config, self.source_id = config, source_id

    async def authenticate(self) -> None:
        raise AssertionError(UNREACHABLE)

    async def discover(self, since=None):
        raise AssertionError(UNREACHABLE)
        yield

    async def list_scope(self):
        raise AssertionError(UNREACHABLE)

    async def confirm_absent(self, items):
        raise AssertionError(UNREACHABLE)

    async def rediscover(self, item):
        raise AssertionError(UNREACHABLE)

    async def fetch(self, item):
        raise AssertionError(UNREACHABLE)


class MemoryRecord:
    """An acquisition record held in memory; each question it holds no answer to raises."""

    def __init__(
        self,
        *,
        discovered: Sequence[ContentItem],
        listing: ScopeListing | None = None,
        answers: dict[str, AbsenceAnswer] | None = None,
        rediscovered: dict[str, ContentItem | None] | None = None,
        acquired: Sequence[AcquiredInput] = (),
        artifact_bodies: dict[str, bytes] | None = None,
    ) -> None:
        self._discovered = list(discovered)
        self._listing = listing
        self._answers = dict(answers or {})
        self._rediscovered = dict(rediscovered or {})
        self._acquired = {acquired_input.item.item_id: acquired_input for acquired_input in acquired}
        self._artifact_bodies = dict(artifact_bodies or {})
        self.closed = False

    def discovered(self) -> Sequence[ContentItem]:
        return self._discovered

    def listing(self) -> ScopeListing:
        if self._listing is None:
            raise AcquisitionRecordMismatch("the record holds no listing")
        return self._listing

    def absence_answer(self, item_id: str) -> AbsenceAnswer:
        if item_id not in self._answers:
            raise AcquisitionRecordMismatch(f"the record holds no absence answer for {item_id}")
        return self._answers[item_id]

    def rediscovered(self, item_id: str) -> ContentItem | None:
        if item_id not in self._rediscovered:
            raise AcquisitionRecordMismatch(f"the record holds no rediscovery of {item_id}")
        return self._rediscovered[item_id]

    async def acquired(self, item_id: str) -> AcquiredInput:
        if item_id not in self._acquired:
            raise AcquisitionRecordMismatch(f"item {item_id} was not acquired")
        return self._acquired[item_id]

    @asynccontextmanager
    async def open_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
        body = self._artifact_bodies[artifact.provider_key]

        async def chunks() -> AsyncIterator[bytes]:
            yield body

        yield SourceArtifactDownload(chunks=chunks(), media_type=artifact.media_type, content_length=len(body))

    async def close(self) -> None:
        self.closed = True


class MemoryRecordStore:
    def __init__(self, record: MemoryRecord) -> None:
        self.record = record

    async def open(self) -> MemoryRecord:
        return self.record


def _issue(number: int) -> ContentItem:
    return ContentItem(
        item_id=f"jira-{number}",
        title=f"Jira {number}",
        source_url=f"https://jira.example/browse/{number}",
        last_modified=datetime(2026, 11, 20, 9, number, tzinfo=timezone.utc),
        content_type="application/json",
        space_or_project="PAY",
        version=str(number),
        extra={"issue_id": str(100000 + number), "issue_key": f"PAY-{number}"},
    )


def _fetched(item: ContentItem, *, artifacts: tuple[RawSourceArtifact, ...] = ()) -> AcquiredInput:
    """An item as the device's ``fetch`` left it: with the comments it read noted on the item."""

    fetched = replace(item, extra={**item.extra, "comments_read": 2})
    raw = _jira_raw_content(fetched)
    return AcquiredInput(item=fetched, body=raw.body, content_type=raw.content_type, artifacts=artifacts)


def _recorded(record: MemoryRecord, config: dict | None = None):
    return recorded_gene_class(JiraIssues)(dict(config or JIRA_CONFIG), SOURCE_ID, record_store=MemoryRecordStore(record))


ARTIFACT = RawSourceArtifact(
    provider_key="attachment-7",
    parent_observation_type="jira_issue",
    parent_provider_key="100001",
    provider_revision="1",
    filename="diagram.png",
    media_type="image/png",
    declared_size_bytes=4,
    locator={"attachment_id": "7"},
)


async def test_acquisition_is_answered_from_the_record_and_interpretation_is_the_types_own() -> None:
    first, second = _issue(1), _issue(2)
    record = MemoryRecord(
        discovered=[first, second],
        listing=ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"jira-1", "jira-2"})),
        answers={"jira-3": AbsenceAnswer.ABSENT},
        rediscovered={"jira-1": first, "jira-4": None},
        acquired=[_fetched(first, artifacts=(ARTIFACT,)), _fetched(second)],
        artifact_bodies={"attachment-7": b"\x89PNG"},
    )
    gene = _recorded(record)

    await gene.authenticate()
    items = [item async for item in gene.discover(datetime(2026, 11, 1, tzinfo=timezone.utc))]
    items[0].extra["changed_by_the_run"] = True
    items[0].stored_extra = {"recorded_by": "this Source"}
    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)
    async with gene.open_source_artifact(raw.artifacts[0]) as download:
        artifact_body = b"".join([chunk async for chunk in download.chunks])

    assert isinstance(gene, JiraIssues) and type(gene).__name__ == "RecordedJiraIssues"
    assert [item.item_id for item in items] == ["jira-1", "jira-2"]
    # Discovery hands out copies: what the run changes never reaches the record.
    assert "changed_by_the_run" not in record.discovered()[0].extra
    assert await gene.list_scope() == record.listing()
    assert await gene.confirm_absent([_issue(3)]) == {"jira-3": AbsenceAnswer.ABSENT}
    assert (await gene.rediscover(_issue(1))).item_id == "jira-1"
    assert await gene.rediscover(_issue(4)) is None
    # The item takes the state the device's fetch left it in, and keeps what this Source stored.
    assert items[0].extra["comments_read"] == 2
    assert items[0].stored_extra == {"recorded_by": "this Source"}
    assert raw.body == record._acquired["jira-1"].body and raw.artifacts == (ARTIFACT,)
    assert artifact_body == b"\x89PNG"
    assert normalized.markdown_body == "# Jira 1\n\nBody"

    await gene.close()
    assert record.closed


async def test_a_question_the_record_cannot_answer_is_a_mismatch() -> None:
    record = MemoryRecord(discovered=[_issue(1)], acquired=[_fetched(_issue(1))])
    gene = _recorded(record)

    with pytest.raises(RuntimeError, match="authenticate"):
        await gene.fetch(_issue(1))
    await gene.authenticate()
    with pytest.raises(AcquisitionRecordMismatch, match="jira-2"):
        await gene.fetch(_issue(2))
    with pytest.raises(AcquisitionRecordMismatch):
        await gene.list_scope()
    with pytest.raises(AcquisitionRecordMismatch):
        await gene.confirm_absent([_issue(2)])
    with pytest.raises(AcquisitionRecordMismatch):
        await gene.rediscover(_issue(2))


def test_the_recorded_gene_replaces_acquisition_only() -> None:
    recorded = recorded_gene_class(JiraIssues)

    assert recorded is recorded_gene_class(JiraIssues)
    assert {name for name in vars(recorded) if not name.startswith("_")} == {
        "authenticate",
        "close",
        "discover",
        "list_scope",
        "confirm_absent",
        "rediscover",
        "fetch",
        "fetch_pdf",
        "open_source_artifact",
    }


@pytest.fixture
async def workspace(tmp_path):
    database = Database(str(tmp_path / "recorded.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    await database.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(JIRA_CONFIG),
        access_policy="workspace",
        owner_user_id="dev",
    )
    try:
        yield Workspace(database)
    finally:
        NoopMemoryEngine.db = None
        await database.close()


def _listing(*numbers: int) -> ScopeListing:
    return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset(f"jira-{number}" for number in numbers))


async def test_a_recorded_run_commits_each_item_as_its_fetch_left_it(workspace) -> None:
    record = MemoryRecord(
        discovered=[_issue(1), _issue(2)],
        listing=_listing(1, 2),
        acquired=[_fetched(_issue(1)), _fetched(_issue(2))],
    )

    state = await workspace.sync(_recorded(record))

    assert state.last_sync_status == "success", state.error_message
    assert await workspace.held() == {"jira-1", "jira-2"}
    stored = await workspace.db.get_source_unit_input(await workspace.unit_id("jira-1"))
    assert stored is not None and stored.item.extra["comments_read"] == 2
    assert record.closed is False


async def test_a_run_fails_when_it_fetches_an_item_its_record_did_not_acquire(workspace) -> None:
    record = MemoryRecord(discovered=[_issue(1), _issue(2)], listing=_listing(1, 2), acquired=[_fetched(_issue(1))])

    state = await workspace.sync(_recorded(record))

    assert state.last_sync_status == "failed"
    assert "jira-2 was not acquired" in (state.error_message or "")
    # The disagreement is the run's, so no Document is left in the failure ledger for it.
    assert "jira-2" not in await workspace.ledger()


async def test_a_run_fails_when_its_record_holds_no_listing(workspace) -> None:
    record = MemoryRecord(discovered=[_issue(1)], acquired=[_fetched(_issue(1))])

    state = await workspace.sync(_recorded(record))

    assert state.last_sync_status == "failed"
    assert state.absence_check_skipped_reason is None
    assert "no listing" in (state.error_message or "")


async def test_a_run_fails_when_its_record_holds_no_answer_for_an_unlisted_document(workspace) -> None:
    both = [_fetched(_issue(1)), _fetched(_issue(2))]
    first = MemoryRecord(discovered=[_issue(1), _issue(2)], listing=_listing(1, 2), acquired=both)
    assert (await workspace.sync(_recorded(first))).last_sync_status == "success"

    # jira-2 left the query, and the record has no confirmation answer for it.
    state = await workspace.sync(_recorded(MemoryRecord(discovered=[], listing=_listing(1), acquired=both)))

    assert state.last_sync_status == "failed"
    assert state.absence_check_skipped_reason is None
    assert "no absence answer for jira-2" in (state.error_message or "")
    assert await workspace.held() == {"jira-1", "jira-2"}
