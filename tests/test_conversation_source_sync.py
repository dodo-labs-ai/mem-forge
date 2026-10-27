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
from fastapi.testclient import TestClient

from memforge.conversations.collect import collect
from memforge.conversations.collections import ArtifactLink, CollectedBody, StoredCollections
from memforge.conversations.intake import CollectionIntake
from memforge.conversations.layer import ConversationLayer
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
from memforge.pipeline.collection_family import collection_family
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.server.admin_api import create_admin_app
from memforge.source_adapters import teams as teams_adapter
from memforge.source_adapters.teams import MESSAGE_SCHEMA
from memforge.source_artifact_inputs import retain_artifact_input
from memforge.source_projection import ProjectionScopeTransition, ProjectionScopeTransitionStatus
from memforge.source_projection_config import canonical_projection_scope
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore
from tests.conversation_attempts_fixture import poll_read, without
from tests.test_startup_runtime import TEST_SOURCE_KEY, _config
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
        # Without an older message, the oldest page is complete from the chat's oldest message.
        complete_from = max(before) if before else (everything[-1]["time"] if everything else None)
        return ConversationPoll(
            messages=tuple(teams_provider_message(message, conversation) for message in inside),
            outcome=PollOutcome.REACHED_START if before else PollOutcome.HISTORY_EXHAUSTED,
            earliest_complete_page_time=complete_from,
        )

    async def read_artifact(self, conversation, message, artifact):
        hosted_content_id = artifact.provider_key.rpartition("/hostedContents/")[2]
        if hosted_content_id in self.unavailable_images:
            raise RuntimeError("Graph refused the hosted content")
        return ArtifactBody(content=PNG + hosted_content_id.encode(), media_type="image/png", filename=f"{hosted_content_id}.png")


PNG = b"\x89PNG\r\n\x1a\n"


UNPROJECTABLE = "cannot be projected"


class UnprojectableEngine(_OrderRecordingMemoryEngine):
    """While ``failing``, fails to commit a Unit whose content holds ``UNPROJECTABLE``, as a failed document."""

    failing = False

    async def prepare_and_commit_projected_lifecycle(self, **kwargs):
        if self.failing and UNPROJECTABLE in kwargs["document_content"]:
            raise ValueError("lifecycle commit failed")
        return await super().prepare_and_commit_projected_lifecycle(**kwargs)


class TeamsWorkspace:
    def __init__(self, db: Database, config: dict, documents: LocalDocumentStore, app_config) -> None:
        self.db = db
        self.config = config
        self.engine = UnprojectableEngine(defer=False)
        self.extractor = ProjectionFragmentRecordingExtractor()
        self.documents = documents
        self.app_config = app_config
        self.attempts = 0

    async def save(self, config: dict) -> None:
        """Save the Source's configuration as the user does, through the Source route."""

        with TestClient(create_admin_app(db=self.db, config=self.app_config)) as client:
            response = client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": config})
        assert response.status_code == 200, response.text
        self.config = (await self.db.get_source(SOURCE_ID))["config"]

    async def collect_and_sync(self, teams: FakeTeams, *, dry_run: bool = False, requested_at: datetime | None = None):
        """One device job attempt and the sync run it starts, or with ``dry_run`` the plan it would apply.

        A dry run plans as the plan route does: the run's collection family
        binds the attempt and starts the open transition. ``requested_at`` is
        when the server issued the attempt's request, now unless given.
        """

        self.attempts += 1
        attempt_id = f"job-{self.attempts}:attempt:1"
        gene = TeamsGene(dict(self.config), SOURCE_ID)
        reader = SourceProjectionUnitReader(self.db)
        collections = StoredCollections(self.db, workspace_id=WORKSPACE_ID)
        request = await gene.issue_collection_request(
            collections, attempt_id, now=requested_at or datetime.now(timezone.utc)
        )
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
            family = collection_family(gene, db=self.db, doc_store=self.documents, source_id=SOURCE_ID)
            family.prepare(collection_attempt=CollectionAttempt(WORKSPACE_ID, attempt_id), input_snapshot=None)
            transition = await self.db.get_open_projection_scope_transition(SOURCE_ID)
            if transition is not None:
                family.begin_scope_transition(transition)
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
            memory_extractor=self.extractor,
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
async def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
        execution_owner_user_id="dev",
    )
    try:
        yield TeamsWorkspace(
            database, config, LocalDocumentStore(str(tmp_path / "documents")), _config(tmp_path)
        )
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


Y_CHAT = "19:y@thread.v2"
Z_CHAT = "19:z@thread.v2"
W_CHAT = "19:w@thread.v2"


def _selection(*conversation_ids: str) -> dict:
    return {"conversation_ids": sorted(conversation_ids), "initial_history_days": 30}


def _teams_with_each_chat() -> FakeTeams:
    """X (``CHAT``), Y, Z and W; Y, Z and W each hold two windows, one of which fails while the engine fails."""

    teams = FakeTeams()
    teams.post(CHAT, "x1", _ago(3), "kept")
    for chat in (Y_CHAT, Z_CHAT, W_CHAT):
        name = chat.split(":")[1].split("@")[0]
        teams.post(chat, f"{name}1", _ago(40), f"{name} projects")
        teams.post(chat, f"{name}2", _ago(3), f"{name} {UNPROJECTABLE}")
    return teams


async def _held(workspace: TeamsWorkspace) -> dict[str, int]:
    """How many current Units each conversation of the Source holds."""

    reader = SourceProjectionUnitReader(workspace.db)
    held = await reader.conversations_held(SOURCE_ID)
    return {conversation_id: len(await reader.units_of(SOURCE_ID, conversation_id)) for conversation_id in sorted(held)}


async def _hold_open(workspace: TeamsWorkspace, teams: FakeTeams, target: dict) -> ProjectionScopeTransition:
    """Save ``target`` and sync it, held open as production holds it: a Unit commits and one document fails."""

    await workspace.save(target)
    workspace.engine.failing = True
    state, _missing = await workspace.collect_and_sync(teams)
    workspace.engine.failing = False
    assert state.docs_failed > 0
    transition = await workspace.db.get_open_projection_scope_transition(SOURCE_ID)
    assert transition is not None and transition.target_scope == canonical_projection_scope("teams", target)
    return transition


def _removed(caplog) -> set[tuple[str, str]]:
    """Each conversation the run removed Units of, with the reason, as its removal log names them."""

    removed = set()
    for record in caplog.records:
        logged = re.match(r"conversation_units_removed .* conversation=(\S+) reason=(\S+) ", record.getMessage())
        if logged:
            removed.add((logged[1], logged[2]))
    return removed


async def test_a_to_b_superseded_by_a_to_c_removes_what_only_b_added_and_completes(workspace, caplog):
    teams = _teams_with_each_chat()
    await workspace.collect_and_sync(teams)
    to_b = await _hold_open(workspace, teams, _selection(CHAT, Y_CHAT))
    assert await _held(workspace) == {CHAT: 1, Y_CHAT: 1}

    await workspace.save(_selection(CHAT, Z_CHAT))
    to_c = await workspace.db.get_open_projection_scope_transition(SOURCE_ID)
    with caplog.at_level(logging.INFO, logger="memforge.genes.conversation_gene"):
        state, _missing = await workspace.collect_and_sync(teams)

    assert to_c.predecessor_transition_id == to_b.id
    assert to_c.previous_scope == canonical_projection_scope("teams", _selection(CHAT))
    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert _removed(caplog) == {(Y_CHAT, "removed_from_scope")}
    assert await _held(workspace) == {CHAT: 1, Z_CHAT: 2}
    assert {transition.id: transition.status for transition in await workspace.db.list_projection_scope_transitions(SOURCE_ID)} == {
        to_b.id: ProjectionScopeTransitionStatus.SUPERSEDED,
        to_c.id: ProjectionScopeTransitionStatus.APPLIED,
    }


async def test_a_revert_to_a_removes_what_the_superseded_targets_added(workspace, caplog):
    teams = _teams_with_each_chat()
    await workspace.collect_and_sync(teams)
    to_b = await _hold_open(workspace, teams, _selection(CHAT, Y_CHAT))

    await workspace.save(_selection(CHAT))
    back_to_a = await workspace.db.get_open_projection_scope_transition(SOURCE_ID)
    with caplog.at_level(logging.INFO, logger="memforge.genes.conversation_gene"):
        state, _missing = await workspace.collect_and_sync(teams)

    assert back_to_a.previous_scope == back_to_a.target_scope == canonical_projection_scope("teams", _selection(CHAT))
    assert back_to_a.predecessor_transition_id == to_b.id
    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert _removed(caplog) == {(Y_CHAT, "removed_from_scope")}
    assert await _held(workspace) == {CHAT: 1}
    assert await workspace.db.get_open_projection_scope_transition(SOURCE_ID) is None


async def test_a_chain_of_three_removes_what_every_superseded_target_added(workspace, caplog):
    teams = _teams_with_each_chat()
    await workspace.collect_and_sync(teams)
    to_b = await _hold_open(workspace, teams, _selection(CHAT, Y_CHAT))
    with caplog.at_level(logging.INFO, logger="memforge.genes.conversation_gene"):
        to_c = await _hold_open(workspace, teams, _selection(CHAT, Z_CHAT))
    removed_by_c = _removed(caplog)
    held_under_c = await _held(workspace)
    caplog.clear()

    await workspace.save(_selection(CHAT, W_CHAT))
    to_d = await workspace.db.get_open_projection_scope_transition(SOURCE_ID)
    with caplog.at_level(logging.INFO, logger="memforge.genes.conversation_gene"):
        state, _missing = await workspace.collect_and_sync(teams)

    assert removed_by_c == {(Y_CHAT, "removed_from_scope")}
    assert held_under_c == {CHAT: 1, Z_CHAT: 1}
    assert (to_c.predecessor_transition_id, to_d.predecessor_transition_id) == (to_b.id, to_c.id)
    assert to_d.previous_scope == canonical_projection_scope("teams", _selection(CHAT))
    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert _removed(caplog) == {(Z_CHAT, "removed_from_scope")}
    assert await _held(workspace) == {CHAT: 1, W_CHAT: 2}
    assert [transition.status for transition in await workspace.db.list_projection_scope_transitions(SOURCE_ID)] == [
        ProjectionScopeTransitionStatus.APPLIED,
        ProjectionScopeTransitionStatus.SUPERSEDED,
        ProjectionScopeTransitionStatus.SUPERSEDED,
    ]


async def test_a_dry_run_of_a_superseding_transition_plans_the_same_removals(workspace):
    teams = _teams_with_each_chat()
    await workspace.collect_and_sync(teams)
    await _hold_open(workspace, teams, _selection(CHAT, Y_CHAT))
    y_units = {unit.document_id for unit in await SourceProjectionUnitReader(workspace.db).units_of(SOURCE_ID, Y_CHAT)}
    await workspace.save(_selection(CHAT, Z_CHAT))

    report, _missing = await workspace.collect_and_sync(teams, dry_run=True)
    held_after_the_dry_run = await _held(workspace)
    await workspace.collect_and_sync(teams)

    assert {(removal["document_id"], removal["reason"]) for removal in report["removals"]} == {
        (document_id, "removed_from_scope") for document_id in y_units
    }
    assert held_after_the_dry_run == {CHAT: 1, Y_CHAT: 1}
    assert await SourceProjectionUnitReader(workspace.db).units_of(SOURCE_ID, Y_CHAT) == ()


async def test_a_transition_deselects_every_held_conversation_its_target_does_not_select(workspace):
    teams = _teams_with_each_chat()
    await workspace.save(_selection(CHAT, Y_CHAT, Z_CHAT))
    await workspace.collect_and_sync(teams)
    assert await _held(workspace) == {CHAT: 1, Y_CHAT: 2, Z_CHAT: 2}
    # The last applied scope says {X, Y, Z}; a transition recorded from {X, Y}
    # still deselects Z, which the Source holds and the target leaves out.
    target = _selection(CHAT)
    await workspace.db.upsert_source(
        id=SOURCE_ID, type="teams", name="Teams", config_json=json.dumps(target),
        access_policy="workspace", owner_user_id="dev", execution_owner_user_id="dev",
    )
    workspace.config = target
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="transition-from-a-narrower-scope",
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("teams", _selection(CHAT, Y_CHAT)),
            target_scope=canonical_projection_scope("teams", target),
        )
    )

    state, _missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success"
    assert await _held(workspace) == {CHAT: 1}
    assert await workspace.db.get_open_projection_scope_transition(SOURCE_ID) is None


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


async def test_rolling_retention_removes_a_window_once_its_latest_message_ages_out(workspace):
    teams = FakeTeams()
    teams.post(CHAT, "old", NOW - timedelta(days=400), "old")
    teams.post(CHAT, "new", _ago(2), "new")
    # The old window was collected by an attempt requested 40 days ago, while
    # it was still within Initial History.
    workspace.config = {"conversation_ids": [CHAT], "initial_history_days": 365}
    await workspace.collect_and_sync(teams, requested_at=NOW - timedelta(days=40))
    assert f"{UnitKind.WINDOW.value}:old" in await workspace.units()

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


async def test_a_backfill_poll_that_stops_early_is_continued_without_removing_anything_unread(workspace):
    teams = FakeTeams()
    for index, hours in enumerate((200, 150, 100, 50, 40, 30, 20)):
        teams.post(CHAT, f"m{index}", _ago(hours), f"message {index}")
    await workspace.collect_and_sync(teams)
    teams.delete(CHAT, "m0")
    # Every poll stops after one page of two messages.
    teams.poll_limit = 2
    collections = StoredCollections(workspace.db, workspace_id=WORKSPACE_ID)
    read: list[tuple[datetime, datetime]] = []

    for _attempt in range(3):
        state, _missing = await workspace.collect_and_sync(teams)

        assert state.last_sync_status == "success"
        manifest = await collections.latest_manifest(SOURCE_ID)
        request = await collections.request(SOURCE_ID, manifest.collection_attempt_id)
        assert len(request.conversations[0].intervals) <= 3
        read.extend(_read_by(manifest))
        following = ConversationLayer.collection_request(
            selection=TeamsGene(dict(workspace.config), SOURCE_ID).conversation_selection(),
            now=datetime.now(timezone.utc),
            previous=manifest,
        )
        floor, *backfills = following.conversations[0].intervals
        regular_end = _read_by(manifest)[0][1]
        assert without([(floor.start, regular_end)], [*read, *((b.start, b.end) for b in backfills)]) == []
        # m0 never lay inside a proven interval of a poll that stopped early.
        assert "m0" in {member for members in (await workspace.units()).values() for member in members}
    assert any(len(conversation.intervals) == 3 for conversation in request.conversations)


def _read_by(manifest) -> list[tuple[datetime, datetime]]:
    """What each poll of the manifest's one conversation read."""

    (evidence,) = manifest.conversations
    return [poll_read(poll, received_at=manifest.received_at, conversation=evidence.ref) for poll in evidence.polls]


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


async def _message_schema_versions(workspace: TeamsWorkspace) -> dict[str, int]:
    """The record schema version each current message revision is read under, by message id."""

    async with workspace.db.db.execute(
        "SELECT so.provider_key, sor.representation_schema_version FROM source_observations so "
        "JOIN source_observation_revisions sor ON sor.id = so.current_revision_id WHERE so.observation_type = 'message'"
    ) as cursor:
        return {str(row[0]): int(row[1]) async for row in cursor}


async def test_a_unit_collected_under_an_earlier_record_schema_is_reprocessed_under_it_and_moves_unread(
    workspace, monkeypatch,
):
    earlier = teams_adapter.CANONICAL_RECORD_SCHEMAS[(MESSAGE_SCHEMA.name, 2)]
    teams = FakeTeams()
    teams.post(CHAT, "m1", _ago(3), "first")
    with monkeypatch.context() as collected_under_earlier:
        collected_under_earlier.setattr(teams_adapter, "MESSAGE_SCHEMA", earlier)
        collected_under_earlier.setattr(TeamsGene, "record_schema", earlier.qualified_name)
        collected_under_earlier.setattr(teams, "record_schema", earlier)
        state, _missing = await workspace.collect_and_sync(teams)
    assert state.last_sync_status == "success" and workspace.extractor.fragment_calls
    assert await _message_schema_versions(workspace) == {"m1": 2}

    state = await workspace.reprocess(frozenset(await workspace.db.list_indexed_doc_ids(SOURCE_ID)))

    assert state.last_sync_status == "success" and state.docs_failed == 0
    assert await _message_schema_versions(workspace) == {"m1": 2}

    workspace.extractor.fragment_calls.clear()
    state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success" and state.docs_failed == 0
    assert [item.identity.message_id for item in missing] == ["m1"]
    assert await _message_schema_versions(workspace) == {"m1": MESSAGE_SCHEMA.version}
    # The message reads the same under the new schema, so nothing is extracted again.
    assert workspace.extractor.fragment_calls == []


async def test_a_member_carried_from_a_collection_under_an_earlier_record_schema_keeps_it(workspace, monkeypatch):
    earlier = teams_adapter.CANONICAL_RECORD_SCHEMAS[(MESSAGE_SCHEMA.name, 2)]
    teams = FakeTeams()
    teams.post(CHAT, "ancient", NOW - timedelta(days=60), "long ago")
    teams.post(CHAT, "m1", _ago(30), "first")
    with monkeypatch.context() as collected_under_earlier:
        collected_under_earlier.setattr(teams_adapter, "MESSAGE_SCHEMA", earlier)
        collected_under_earlier.setattr(TeamsGene, "record_schema", earlier.qualified_name)
        collected_under_earlier.setattr(teams, "record_schema", earlier)
        state, _missing = await workspace.collect_and_sync(teams)
    assert state.last_sync_status == "success"
    assert await _message_schema_versions(workspace) == {"m1": 2}

    # The next poll reads only the new message before its budget runs out, so the window carries m1.
    teams.post(CHAT, "m2", _ago(29.5), "follow-up")
    teams.poll_limit = 1
    state, missing = await workspace.collect_and_sync(teams)

    assert state.last_sync_status == "success" and state.docs_failed == 0
    assert [item.identity.message_id for item in missing] == ["m2"]
    assert await workspace.units() == {f"{UnitKind.WINDOW.value}:m1": ["m1", "m2"]}
    assert await _message_schema_versions(workspace) == {"m1": 2, "m2": MESSAGE_SCHEMA.version}
    assert (await _stored_records(workspace))["m1"]["representation"] == earlier.qualified_name
