"""A Teams Source collected through the conversation layer, end to end over SQLite.

A fake provider stands in for the Teams Chat Service on the device. Each
attempt answers the request the Gene issues by collecting through the shared
``collect`` every executor runs, hands its manifest, artifacts and bodies to the
server's intake, and syncs through the orchestrator, so every decision is the
layer's. The device's upload protocol is covered in
``test_conversation_executor``.
"""

from __future__ import annotations

import logging
import re

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from memforge.conversations.collect import collect
from memforge.conversations.collections import ArtifactLink, CollectedBody, StoredCollections
from memforge.conversations.intake import CollectionIntake
from memforge.conversations.port import (
    ArtifactBody,
    ConversationAccess,
    ConversationListing,
    ConversationPoll,
    ConversationStatus,
    ConversationUnavailable,
    Interval,
    PollOutcome,
)
from memforge.conversations.reader import SourceProjectionUnitReader
from memforge.conversations.report import plan_report
from memforge.conversations.units import UnitKind
from memforge.genes.conversation_gene import CollectionAttempt
from memforge.genes.teams_gene import TeamsGene
from memforge.genes.teams_provider import TeamsConversationProvider, teams_conversation_ref, teams_provider_message
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.source_adapters.teams import MESSAGE_SCHEMA
from memforge.source_artifact_inputs import retain_artifact_input
from memforge.source_projection import ProjectionScopeTransition, ProjectionScopeTransitionStatus
from memforge.source_projection_config import canonical_projection_scope
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore
from tests.test_sync_bookkeeping import (
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    _OrderRecordingMemoryEngine,
    _skip_retry_delay,
)

SOURCE_ID = "src-teams-conversations"
WORKSPACE_ID = "default"
CHAT = "19:chat@thread.v2"
NOW = datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class FakeTeams:
    """Teams as the provider shows it: the messages of each chat, and the chats the user can no longer read."""

    messages: dict[str, dict[str, dict]] = field(default_factory=dict)
    refused: set[str] = field(default_factory=set)
    # Inline images Microsoft Graph will not serve right now.
    unavailable_images: set[str] = field(default_factory=set)
    # Chats the recent listing leaves out; each is still read by id, as Teams reads it.
    unlisted: set[str] = field(default_factory=set)
    # Chat names; a chat without one is named by its id.
    topics: dict[str, str] = field(default_factory=dict)
    # The newest messages one poll reads before its budget runs out; ``None`` reads every page.
    poll_limit: int | None = None
    record_schema = MESSAGE_SCHEMA

    def post(
        self,
        conversation_id: str,
        message_id: str,
        at: datetime,
        text: str,
        *,
        root: str | None = None,
        images: tuple[str, ...] = (),
    ) -> None:
        self.messages.setdefault(conversation_id, {})[message_id] = {
            "id": message_id,
            "from": "Ada",
            "content": text,
            "time": at,
            "edited_time": None,
            "attachments": [],
            "hosted_content_ids": list(images),
            **({"rootMessageId": root} if root else {}),
        }

    def delete(self, conversation_id: str, message_id: str) -> None:
        del self.messages[conversation_id][message_id]

    async def list_conversations(self) -> ConversationListing:
        refs = tuple(
            teams_conversation_ref({"id": cid, "topic": self.topics.get(cid, cid), "type": "group_chat"})
            for cid in self.messages
            if cid not in self.unlisted
        )
        return ConversationListing(
            conversations=refs,
            statuses={ref.conversation_id: ConversationStatus(ConversationAccess.READABLE, "listed") for ref in refs},
        )

    async def get_conversation(self, conversation_id: str):
        return await TeamsConversationProvider(client=None).get_conversation(conversation_id)

    async def poll(self, conversation, interval: Interval) -> ConversationPoll:
        if conversation.conversation_id in self.refused:
            raise ConversationUnavailable(ConversationStatus(ConversationAccess.NO_ACCESS, "HTTP 403"))
        everything = sorted(self.messages[conversation.conversation_id].values(), key=lambda m: m["time"], reverse=True)
        inside = [m for m in everything if m["time"] >= interval.start and (interval.end is None or m["time"] < interval.end)]
        if self.poll_limit is not None and len(inside) > self.poll_limit:
            read = inside[: self.poll_limit]
            return ConversationPoll(
                messages=tuple(teams_provider_message(message, conversation) for message in read),
                outcome=PollOutcome.STOPPED_EARLY,
                earliest_complete_page_time=read[-1]["time"],
                stop_reason="budget",
            )
        before = [m["time"] for m in everything if m["time"] < interval.start]
        return ConversationPoll(
            messages=tuple(teams_provider_message(message, conversation) for message in inside),
            outcome=PollOutcome.REACHED_START if before else PollOutcome.HISTORY_EXHAUSTED,
            earliest_complete_page_time=max(before) if before else None,
        )

    async def read_artifact(self, conversation, message, artifact):
        hosted_content_id = artifact.provider_key.rpartition("/hostedContents/")[2]
        if hosted_content_id in self.unavailable_images:
            raise RuntimeError("Graph refused the hosted content")
        return ArtifactBody(content=PNG + hosted_content_id.encode(), media_type="image/png", filename=f"{hosted_content_id}.png")


PNG = b"\x89PNG\r\n\x1a\n"


class TeamsWorkspace:
    def __init__(self, db: Database, config: dict, documents: LocalDocumentStore) -> None:
        self.db = db
        self.config = config
        self.engine = _OrderRecordingMemoryEngine(defer=False)
        self.documents = documents
        self.attempts = 0

    async def collect_and_sync(self, teams: FakeTeams, *, dry_run: bool = False):
        """One device job attempt and the sync run it starts, or with ``dry_run`` the plan it would apply."""

        self.attempts += 1
        attempt_id = f"job-{self.attempts}:attempt:1"
        gene = TeamsGene(dict(self.config), SOURCE_ID)
        reader = SourceProjectionUnitReader(self.db)
        collections = StoredCollections(self.db, workspace_id=WORKSPACE_ID)
        request = await gene.issue_collection_request(collections, attempt_id, now=datetime.now(timezone.utc))
        collection = await collect(
            teams, request, source_id=SOURCE_ID, collection_attempt_id=attempt_id, config_revision="cfg"
        )
        intake = CollectionIntake(reader=reader, collections=collections)
        payload = _device_payload(collection)
        missing = await intake.receive_manifest(
            payload,
            source_id=SOURCE_ID,
            collection_attempt_id=attempt_id,
            config_revision="cfg",
            record_schema=gene.record_schema,
            received_at=datetime.now(timezone.utc),
        )
        uploads = {}
        for item in missing:
            links = []
            for artifact in item.artifacts:
                try:
                    body = await teams.read_artifact(
                        collection.refs[item.identity.conversation_id], collection.provider_messages[item.identity], artifact
                    )
                except RuntimeError:
                    continue
                input_sha256 = await retain_artifact_input(
                    self.db,
                    self.documents,
                    source_id=SOURCE_ID,
                    workspace_id=WORKSPACE_ID,
                    identity={
                        "source_unit_key": item.identity.conversation_id,
                        "provider_key": artifact.provider_key,
                        "provider_revision": artifact.provider_revision,
                        "parent_observation_type": "message",
                        "parent_provider_key": item.identity.message_id,
                        "filename": body.filename,
                    },
                    media_type=body.media_type,
                    content=io.BytesIO(body.content),
                    size_bytes=len(body.content),
                    content_sha256=hashlib.sha256(body.content).hexdigest(),
                )
                links.append(ArtifactLink(artifact.provider_key, artifact.provider_revision, input_sha256))
            uploads[item.identity] = CollectedBody(record=collection.bodies[item.identity], artifacts=tuple(links))
        await intake.receive_bodies(uploads, source_id=SOURCE_ID, collection_attempt_id=attempt_id)
        assert await intake.missing_bodies_of(SOURCE_ID, attempt_id) == ()
        if dry_run:
            gene.bind_conversation_run(projection_store=self.db, collections=collections, collection_attempt_id=attempt_id)
            return plan_report(await gene.preview(), source_id=SOURCE_ID), missing
        self.engine.events.clear()
        state = await self._orchestrator().sync_gene(
            gene=gene,
            source_name="Teams",
            source_id=SOURCE_ID,
            collection_attempt=CollectionAttempt(WORKSPACE_ID, attempt_id),
        )
        return state, missing

    async def reprocess(self, doc_ids: frozenset[str]):
        """An operator reprocess of committed Documents; it collects nothing."""

        gene = TeamsGene(dict(self.config), SOURCE_ID)
        self.engine.events.clear()
        return await self._orchestrator().sync_gene(
            gene=gene,
            source_name="Teams",
            source_id=SOURCE_ID,
            reprocess_doc_ids=doc_ids,
            execution_mode=SourceSyncMode.REPROCESS,
            collection_attempt=CollectionAttempt(WORKSPACE_ID, "reprocess-run:attempt:1"),
        )

    def _orchestrator(self) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=self.documents,
            memory_extractor=ProjectionFragmentRecordingExtractor(),
            memory_engine=self.engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.db),
            max_concurrent=1,
            retry_sleep=_skip_retry_delay,
        )

    async def units(self) -> dict[str, list[str]]:
        """Each current Unit's kind and its members' message ids."""

        reader = SourceProjectionUnitReader(self.db)
        held = await reader.units_spanning(SOURCE_ID, CHAT, NOW - timedelta(days=3650), NOW + timedelta(days=1))
        return {
            f"{unit.kind.value}:{unit.members[0].identity.message_id}": [m.identity.message_id for m in unit.members]
            for unit in held
        }


def _device_payload(collection) -> dict:
    payload = json.loads(json.dumps(_manifest_payload(collection)))
    del payload["received_at"]
    return payload


def _manifest_payload(collection):
    from memforge.conversations.manifest import manifest_to_payload

    return manifest_to_payload(collection.manifest(received_at=datetime.now(timezone.utc)))


@pytest.fixture
async def workspace(tmp_path):
    database = Database(str(tmp_path / "teams.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    config = {"conversation_ids": [CHAT], "initial_history_days": 30}
    await database.upsert_source(
        id=SOURCE_ID,
        type="teams",
        name="Teams",
        config_json=json.dumps(config),
        access_policy="workspace",
        owner_user_id="dev",
    )
    try:
        yield TeamsWorkspace(database, config, LocalDocumentStore(str(tmp_path / "documents")))
    finally:
        NoopMemoryEngine.db = None
        await database.close()


def _ago(hours: float) -> datetime:
    return NOW - timedelta(hours=hours)


async def test_messages_become_windows_and_unchanged_messages_upload_nothing_again(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(30), "first")
    teams.post(CHAT, "m2", _ago(29.5), "second")
    teams.post(CHAT, "m3", _ago(5), "later")

    state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert len(missing) == 3
    assert await workspace.units() == {
        f"{UnitKind.WINDOW.value}:m1": ["m1", "m2"],
        f"{UnitKind.WINDOW.value}:m3": ["m3"],
    }

    state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert missing == ()
    assert workspace.engine.events == []


async def test_a_deleted_window_is_removed_and_an_edit_revises_its_window(workspace, caplog):
    teams = FakeTeams()
    # History older than the requested range shows the poll read its whole range.
    teams.post(CHAT, "ancient", NOW - timedelta(days=60), "long ago")
    teams.post(CHAT, "m1", _ago(30), "first")
    teams.post(CHAT, "m3", _ago(5), "later")
    await workspace.collect_and_sync(teams)

    teams.delete(CHAT, "m1")
    teams.post(CHAT, "m3", _ago(5), "later, edited")
    with caplog.at_level(logging.INFO, logger="memforge.genes.conversation_gene"):
        state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert [item.identity.message_id for item in missing] == ["m3"]
    assert await workspace.units() == {f"{UnitKind.WINDOW.value}:m3": ["m3"]}
    assert sorted(event for event, _doc in workspace.engine.events) == ["commit", "tombstone"]
    assert [
        re.sub(r" (source|attempt)=\S+", "", record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("conversation_units_removed")
    ] == [f"conversation_units_removed conversation={CHAT} reason=absent_from_source units=1"]


async def test_a_message_older_than_the_poll_is_kept_when_history_runs_out(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(30), "first")
    teams.post(CHAT, "m2", _ago(2), "recent")
    await workspace.collect_and_sync(teams)

    # The account can no longer see anything before m2; the provider simply has nothing older to show.
    teams.delete(CHAT, "m1")
    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1", f"{UnitKind.WINDOW.value}:m2"}


async def test_a_root_gaining_a_reply_moves_to_its_thread_over_two_runs(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "r", _ago(10), "question")
    teams.post(CHAT, "b", _ago(9.9), "aside")
    await workspace.collect_and_sync(teams)

    teams.post(CHAT, "x", _ago(1), "answer", root="r")
    await workspace.collect_and_sync(teams)
    assert await workspace.units() == {
        f"{UnitKind.WINDOW.value}:r": ["r", "b"],
        f"{UnitKind.THREAD.value}:r": ["r", "x"],
    }

    await workspace.collect_and_sync(teams)
    assert await workspace.units() == {
        f"{UnitKind.WINDOW.value}:b": ["b"],
        f"{UnitKind.THREAD.value}:r": ["r", "x"],
    }


async def test_lost_access_keeps_every_unit_and_names_the_chat_in_a_partial_run(workspace):
    teams = FakeTeams(topics={CHAT: "Release crew"})
    teams.post(CHAT, "m1", _ago(3), "first")
    await workspace.collect_and_sync(teams)

    teams.refused.add(CHAT)
    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "partial"
    assert state.docs_failed == 0
    assert state.error_message == (
        "1 conversation could not be read: Release crew. Their memories are kept. "
        "Restore access, or deselect a conversation that is no longer used."
    )
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1"}
    assert workspace.engine.events == []
    assert await workspace.db.list_source_item_failures(source_id=SOURCE_ID) == ()

    state, _missing = await workspace.collect_and_sync(teams)

    assert "could not be read: Release crew." in state.error_message

    teams.refused.clear()
    teams.unlisted.add(CHAT)
    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert state.error_message is None
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1"}
    assert workspace.engine.events == []


async def _images_of(workspace: TeamsWorkspace) -> dict[str, list[str]]:
    """The inline images each current Unit holds, by its first member."""

    images = {}
    for unit in await SourceProjectionUnitReader(workspace.db).units_spanning(
        SOURCE_ID, CHAT, NOW - timedelta(days=3650), NOW + timedelta(days=1)
    ):
        images[unit.members[0].identity.message_id] = sorted(
            artifact.provider_key for member in unit.members for artifact in member.artifacts
        )
    return images


async def test_an_inline_image_is_projected_with_its_message_and_kept_while_the_message_is_carried(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "ancient", NOW - timedelta(days=60), "long ago")
    teams.post(CHAT, "m1", _ago(30), "diagram", images=("hosted-1",))
    await workspace.collect_and_sync(teams)

    assert await _images_of(workspace) == {"m1": ["m1/hostedContents/hosted-1"]}

    # A new message joins the window; the image of the carried message stays and is not uploaded again.
    teams.post(CHAT, "m2", _ago(29.5), "follow-up")
    _state, missing = await workspace.collect_and_sync(teams)

    assert [(item.identity.message_id, item.artifacts) for item in missing] == [("m2", ())]
    assert await _images_of(workspace) == {"m1": ["m1/hostedContents/hosted-1"]}


async def test_an_image_the_provider_will_not_serve_is_fetched_by_a_later_attempt(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "diagram", images=("hosted-1",))
    teams.unavailable_images.add("hosted-1")

    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert await _images_of(workspace) == {"m1": []}

    teams.unavailable_images.clear()
    _state, missing = await workspace.collect_and_sync(teams)

    assert [(item.identity.message_id, item.body, len(item.artifacts)) for item in missing] == [("m1", False, 1)]
    assert await _images_of(workspace) == {"m1": ["m1/hostedContents/hosted-1"]}


async def test_a_deselected_conversation_is_removed_from_scope_and_the_transition_applies(workspace):
    other = "19:other@thread.v2"
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "kept")
    teams.post(other, "o1", _ago(3), "dropped")
    previous = {"conversation_ids": [CHAT, other], "initial_history_days": 30}
    workspace.config = previous
    await workspace.collect_and_sync(teams)

    target = {"conversation_ids": [CHAT], "initial_history_days": 30}
    workspace.config = target
    await workspace.db.upsert_source(
        id=SOURCE_ID, type="teams", name="Teams", config_json=json.dumps(target),
        access_policy="workspace", owner_user_id="dev",
    )
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="transition-drop-other",
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("teams", previous),
            target_scope=canonical_projection_scope("teams", target),
        )
    )
    state, _missing = await workspace.collect_and_sync(teams)
    transition = (await workspace.db.list_projection_scope_transitions(SOURCE_ID))[0]
    removed_units = await SourceProjectionUnitReader(workspace.db).units_of(SOURCE_ID, other)

    assert state.last_sync_status == "success"
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert removed_units == ()
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1"}


async def test_a_dry_run_reports_the_plan_and_changes_nothing(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "ancient", NOW - timedelta(days=60), "long ago")
    teams.post(CHAT, "m1", _ago(30), "first")
    teams.post(CHAT, "m3", _ago(5), "later")
    await workspace.collect_and_sync(teams)

    teams.delete(CHAT, "m1")
    teams.post(CHAT, "m3", _ago(5), "later, edited")
    teams.post(CHAT, "m4", _ago(4.5), "new")
    report, _missing = await workspace.collect_and_sync(teams, dry_run=True)

    [unit] = report["units"]
    assert (unit["change"], unit["added"], unit["changed"], unit["extracts"]) == ("changed", ["m4"], ["m3"], True)
    assert [removal["reason"] for removal in report["removals"]] == ["absent_from_source"]
    assert report["summary"] == {"new_units": 0, "changed_units": 1, "units_extracted_again": 1, "removed_units": 1}
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1", f"{UnitKind.WINDOW.value}:m3"}


async def test_rolling_retention_removes_a_window_once_its_latest_message_ages_out(workspace, monkeypatch):
    teams = FakeTeams()
    teams.post(CHAT, "old", NOW - timedelta(days=400), "old")
    teams.post(CHAT, "new", _ago(2), "new")
    # The old window stands for one collected while it was still within Initial History.
    monkeypatch.setattr("memforge.genes.teams_gene.TEAMS_MAX_INITIAL_HISTORY_DAYS", 500)
    workspace.config = {"conversation_ids": [CHAT], "initial_history_days": 500}
    await workspace.collect_and_sync(teams)
    monkeypatch.undo()

    workspace.config = {"conversation_ids": [CHAT], "initial_history_days": 365, "rolling_retention_days": 365}
    report, _missing = await workspace.collect_and_sync(teams, dry_run=True)
    state, _missing = await workspace.collect_and_sync(teams)

    assert [removal["reason"] for removal in report["removals"]] == ["outside_rolling_retention"]
    assert state.last_sync_status == "success"
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:new"}


async def test_a_poll_that_stops_early_is_continued_by_the_next_attempt_without_removing_anything_unread(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "ancient", NOW - timedelta(days=60), "long ago")
    for index, hours in enumerate((50, 40, 30, 20)):
        teams.post(CHAT, f"m{index}", _ago(hours), f"message {index}")
    await workspace.collect_and_sync(teams)

    teams.delete(CHAT, "m0")
    teams.poll_limit = 2
    state, _missing = await workspace.collect_and_sync(teams)

    # The poll read only m3 and m2; nothing older is proven absent yet.
    assert state.last_sync_status == "success"
    assert "m0" in {member for members in (await workspace.units()).values() for member in members}

    teams.poll_limit = None
    await workspace.collect_and_sync(teams)

    assert "m0" not in {member for members in (await workspace.units()).values() for member in members}


async def test_a_reprocess_reads_each_unit_from_its_committed_input(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "first")
    await workspace.collect_and_sync(teams)
    doc_ids = frozenset(await workspace.db.list_indexed_doc_ids(SOURCE_ID))

    state = await workspace.reprocess(doc_ids)

    assert state.last_sync_status == "success"
    assert state.docs_failed == 0
    assert set(await workspace.units()) == {f"{UnitKind.WINDOW.value}:m1"}


async def test_a_processed_attempt_keeps_its_manifest_and_drops_its_bodies(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "first")
    await workspace.collect_and_sync(teams)
    collections = StoredCollections(workspace.db, workspace_id=WORKSPACE_ID)

    await collections.prune(SOURCE_ID, "job-1:attempt:1")

    assert await collections.bodies(SOURCE_ID, "job-1:attempt:1") == {}
    assert (await collections.latest_manifest(SOURCE_ID)).collection_attempt_id == "job-1:attempt:1"
    _state, missing = await workspace.collect_and_sync(teams)
    assert missing == ()


async def test_a_chat_the_recent_listing_leaves_out_keeps_its_name(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "first")
    await workspace.collect_and_sync(teams)

    teams.unlisted.add(CHAT)
    teams.post(CHAT, "m2", _ago(2), "second")
    state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert [item.identity.message_id for item in missing] == ["m2"]
    [unit] = await SourceProjectionUnitReader(workspace.db).units_spanning(
        SOURCE_ID, CHAT, NOW - timedelta(days=1), NOW + timedelta(days=1)
    )
    assert {member.record["conversation_name"] for member in unit.members} == {CHAT}


async def _store_as_undated(workspace: TeamsWorkspace, message_id: str, text: str) -> None:
    """Rewrite a stored message as a record of an earlier schema, which carries no creation time."""

    record = json.dumps({"attachments": [], "content": text, "deleted": False}, sort_keys=True, separators=(",", ":"))
    await workspace.db.db.execute(
        "UPDATE source_observation_revisions SET content = ?, representation_schema_version = 1 "
        "WHERE id IN (SELECT current_revision_id FROM source_observations WHERE provider_key = ?)",
        (record, message_id),
    )
    await workspace.db.db.commit()


async def _stored_records(workspace: TeamsWorkspace) -> dict[str, dict]:
    reader = SourceProjectionUnitReader(workspace.db)
    held = await reader.units_spanning(SOURCE_ID, CHAT, NOW - timedelta(days=3650), NOW + timedelta(days=1))
    return {member.identity.message_id: dict(member.record) for unit in held for member in unit.members}


async def test_a_message_stored_without_its_creation_time_is_carried_until_a_poll_returns_it(workspace):
    teams = FakeTeams()
    # The record stored before joining is not the one the provider later returns.
    teams.post(CHAT, "m1", _ago(3), "as first stored")
    await workspace.collect_and_sync(teams)
    await _store_as_undated(workspace, "m1", "first")

    teams.delete(CHAT, "m1")
    teams.post(CHAT, "m2", _ago(2.5), "second")
    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert await workspace.units() == {f"{UnitKind.WINDOW.value}:m1": ["m1", "m2"]}
    assert "time" not in (await _stored_records(workspace))["m1"]

    teams.post(CHAT, "m1", _ago(3), "first")
    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert await workspace.units() == {f"{UnitKind.WINDOW.value}:m1": ["m1", "m2"]}
    assert (await _stored_records(workspace))["m1"]["time"] == _ago(3).isoformat()
