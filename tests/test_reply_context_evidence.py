"""Context Evidence is the Observation the Primary answers, shown as a reader sees it."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import EvidenceReference, EvidenceRole
from memforge.models import RawMemory
from memforge.pipeline.evidence_fragments import observation_presentation
from memforge.pipeline.projection_context import preceding_observation_id, replied_to_observation_id
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database
from tests.test_confluence_native_evidence import projection as confluence_projection
from tests.test_evidence_unit_support import _seed_complete_unit_support
from tests.test_reading_order_and_record_framing import CONVERSATION, _messages, _reading, _window

REPLY, REPLIED_TO, FOLLOWER, FOLLOWED = "msg-2", "msg-1", "msg-3", "msg-2"


def _observation_id(projection, provider_key: str) -> str:
    return next(o.id for o in projection.observations if o.provider_key == provider_key)


def _context(projection, primary_key: str, required_keys: tuple[str, ...] = ()) -> tuple[EvidenceReference, ...]:
    """The Context references of one claim whose Primary is the given message."""
    catalog = _reading(projection).catalog
    ref_of = {
        next(o.provider_key for o in projection.observations if o.id == fragment.anchor.observation_id): fragment.reference
        for fragment in catalog.fragments
    }
    selection = catalog.resolve_selection(
        primary_ref=ref_of[primary_key], required_refs=tuple(ref_of[key] for key in required_keys),
    )
    evidence = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(RawMemory(
            content="The nightly export moves to the new cluster on Friday.", memory_type="decision",
            evidence_quote="moves to the new cluster", resolved_evidence_selection=selection,
        ),),
        doc_id="window-1", source_type="teams", project_key=None, visibility="workspace", owner_user_id=None,
        repo_identifier=None, access_context_hash="scope", extractor_run_id="run",
    )
    return tuple(reference for reference in evidence.references if reference.role is EvidenceRole.CONTEXT)


def test_a_reply_is_read_with_its_referent_and_a_follower_with_its_predecessor() -> None:
    projection = _window(_messages())
    reply, follower = _observation_id(projection, REPLY), _observation_id(projection, FOLLOWER)

    assert replied_to_observation_id(projection, reply) == _observation_id(projection, REPLIED_TO)
    assert replied_to_observation_id(projection, follower) is None
    assert preceding_observation_id(projection, reply) == _observation_id(projection, REPLIED_TO)
    assert preceding_observation_id(projection, follower) == _observation_id(projection, FOLLOWED)


def test_a_claim_stated_in_a_reply_carries_the_message_it_answers_as_readable_context() -> None:
    projection = _window(_messages())

    [context] = _context(projection, REPLY)

    assert context.anchor == SourceAnchor(
        kind=AnchorKind.WHOLE_OBSERVATION,
        observation_id=_observation_id(projection, REPLIED_TO),
        observation_revision_id=next(
            r.id for r in projection.observation_revisions if r.observation_id == _observation_id(projection, REPLIED_TO)
        ),
    )
    assert context.excerpt == "\n".join((
        "Reported sender: Ben",
        "Message created: 2026-07-14T10:01:00Z",
        "Conversation ID: conv-1",
        CONVERSATION[1][1],
    ))


def test_a_message_that_merely_precedes_the_primary_is_not_context() -> None:
    assert _context(_window(_messages()), FOLLOWER) == ()


def test_a_referent_selected_as_required_is_not_repeated_as_context() -> None:
    assert _context(_window(_messages()), REPLY, required_keys=(REPLIED_TO,)) == ()


def test_a_whole_record_reads_as_its_framing_once_then_its_entries() -> None:
    rows = (("Ana", "<p>We tested two calls.</p><p>The export API is read-only for clients.</p>", None),)
    revision = _window(_messages(rows)).observation_revisions[0]

    assert observation_presentation(revision) == "\n".join((
        "Reported sender: Ana",
        "Message created: 2026-07-14T10:00:00Z",
        "Conversation ID: conv-1",
        "We tested two calls.",
        "The export API is read-only for clients.",
    ))


def test_a_record_without_declared_framing_reads_as_its_entries() -> None:
    page = confluence_projection("<p>Only after cutoff and approval.</p>").observation_revisions[0]

    assert observation_presentation(page) == "Only after cutoff and approval."


@pytest.mark.asyncio
async def test_stored_context_without_a_presentation_reads_as_its_record_presents_it(tmp_path) -> None:
    message = _window(_messages(CONVERSATION[:1])).observation_revisions[0]
    profile = message.evidence_profile
    now = datetime(2026, 8, 27, 8, 0, tzinfo=timezone.utc).isoformat()
    db = Database(str(tmp_path / "context.db"))
    await db.connect()
    try:
        memory_id, evidence_unit_id, _source_id, _access_hash = await _seed_complete_unit_support(db)
        await db.db.execute(
            """INSERT INTO source_observations (
                   id, source_id, source_unit_id, observation_type, provider_key, locator_json,
                   current_revision_id, updated_at
               ) VALUES ('obs-message', 'source-1', 'unit-1', 'message', 'msg-0', '{}', 'obsrev-message', ?)""",
            (now,),
        )
        await db.db.execute(
            """INSERT INTO source_observation_revisions (
                   id, observation_id, semantic_hash, content, metadata_json, observed_at, profile_name,
                   profile_version, coordinate_space, representation_schema_name,
                   representation_schema_version, created_at
               ) VALUES ('obsrev-message', 'obs-message', ?, ?, '{}', ?, ?, ?, ?, ?, ?, ?)""",
            (
                hashlib.sha256(message.content.encode()).hexdigest(), message.content, now, profile.name,
                profile.version, profile.coordinate_space.value, profile.schema_name, profile.schema_version, now,
            ),
        )
        await db.db.commit()
        await db.replace_evidence_context_associations(evidence_unit_id, (
            EvidenceReference(role=EvidenceRole.CONTEXT, anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION, observation_id="obs-message",
                observation_revision_id="obsrev-message",
            )),
        ))

        [group] = await db.get_memory_evidence_units(memory_id)
        [context] = [item for item in group.items if item.role is EvidenceRole.CONTEXT]
    finally:
        await db.close()

    assert json.loads(message.content)["content"] == CONVERSATION[0][1]
    assert context.excerpt == "\n".join((
        "Reported sender: Ana",
        "Message created: 2026-07-14T10:00:00Z",
        "Conversation ID: conv-1",
        CONVERSATION[0][1],
    ))


@pytest.mark.asyncio
async def test_the_migration_keeps_stored_context_only_for_a_declared_reply(tmp_path) -> None:
    now = datetime(2026, 8, 27, 8, 0, tzinfo=timezone.utc).isoformat()
    path = tmp_path / "stored-context.db"
    db = Database(str(path))
    await db.connect()
    try:
        memory_id, evidence_unit_id, source_id, _access_hash = await _seed_complete_unit_support(db)
        for observation_id, text in (("obs-answered", "Should release wait?"), ("obs-preceding", "Lunch is at noon.")):
            await db.db.execute(
                """INSERT INTO source_observations (
                       id, source_id, source_unit_id, observation_type, provider_key, locator_json,
                       current_revision_id, updated_at
                   ) VALUES (?, ?, 'unit-1', 'document_body', ?, '{}', ?, ?)""",
                (observation_id, source_id, observation_id, f"rev-{observation_id}", now),
            )
            await db.db.execute(
                """INSERT INTO source_observation_revisions (
                       id, observation_id, semantic_hash, content, metadata_json, observed_at, profile_name,
                       profile_version, coordinate_space, created_at
                   ) VALUES (?, ?, ?, ?, '{}', ?, 'markdown-structural', 1, 'unicode-scalar', ?)""",
                (f"rev-{observation_id}", observation_id, hashlib.sha256(text.encode()).hexdigest(), text, now, now),
            )
        await db.db.execute(
            """INSERT INTO source_projection_runs (
                   id, source_id, source_type, coverage, scope_json, checkpoint_json, payload_json,
                   payload_hash, created_at
               ) VALUES ('run-1', ?, 'github_repo', 'complete_snapshot', '{}', '{}', '{}', 'hash', ?)""",
            (source_id, now),
        )
        for index, (relation_type, from_id, to_id) in enumerate((
            ("replies_to", "obs-primary", "obs-answered"),
            ("precedes", "obs-preceding", "obs-primary"),
        )):
            await db.db.execute(
                """INSERT INTO source_projection_relations (
                       projection_run_id, relation_index, relation_type, from_id, to_id
                   ) VALUES ('run-1', ?, ?, ?, ?)""",
                (index, relation_type, from_id, to_id),
            )
        await db.db.commit()
        await db.replace_evidence_context_associations(evidence_unit_id, tuple(
            EvidenceReference(role=EvidenceRole.CONTEXT, anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION, observation_id=observation_id,
                observation_revision_id=f"rev-{observation_id}",
            ))
            for observation_id in ("obs-answered", "obs-preceding")
        ))
        [stored] = await db.get_memory_evidence_units(memory_id)
        assert {item.anchor.observation_id for item in stored.items if item.role is EvidenceRole.CONTEXT} == {
            "obs-answered", "obs-preceding",
        }
        supporting = [item for item in stored.items if item.grants_support]
        await db.db.execute("DELETE FROM schema_migrations WHERE version = ?", (111,))
        await db.db.commit()
    finally:
        await db.close()

    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        [group] = await upgraded.get_memory_evidence_units(memory_id)
        references = await upgraded.db.execute_fetchall(
            "SELECT COUNT(*) AS count FROM evidence_references WHERE role = 'context'"
        )
    finally:
        await upgraded.close()

    assert [item.anchor.observation_id for item in group.items if item.role is EvidenceRole.CONTEXT] == ["obs-answered"]
    # Supporting Evidence and the immutable Context references are untouched.
    assert [item for item in group.items if item.grants_support] == supporting
    assert references[0]["count"] == 2
