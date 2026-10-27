"""An Observation Revision is its content read under one Evidence Representation Profile.

The shared projector names a revision by its Observation, its content and its
profile. A representation upgrade therefore revises every Observation it
covers once; the reading relation then authorizes nothing that reads the same
and rebinds its Support exactly, with no model call.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge import source_representation
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline import evidence_fragments
from memforge.pipeline.evidence_fragments import EvidenceFragmentKind
from memforge.pipeline.projection_context import ExtractionAuthority
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.pipeline.source_projection_adapters import _stable_id, project_source_item
from memforge.pipeline.support_reading import EvidenceCorrespondence, SupportRoute, plan_support_revision
from memforge.source_adapters import jira as jira_adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE, canonical_record_profile
from memforge.source_artifacts import StoredSourceArtifact
from memforge.storage.database import Database
from tests.test_displayed_framing import _comment, _issue
from tests.test_reading_equivalence import _plan, _support_item
from tests.test_reading_order_and_record_framing import CONVERSATION, _messages, _window, prior_of
from tests.test_source_revision_time import _item, _project
from tests.test_unit_title import NoModelClient

_COMMENT = _comment(["Retain A7 for the payroll run.", "Cutoff is Friday."], "2026-07-02T09:00:00.000+0000")


def _comment_stored_under_v5(monkeypatch, *, record_names_its_schema):
    """The comment as a build declaring ``jira-comment`` 5 stored it.

    A record names the schema it was written under, so that build wrote a
    different record; ``record_names_its_schema=False`` keeps the record
    unchanged, as an upgrade of a representation that does not name itself.
    """
    earlier = jira_adapter.CANONICAL_RECORD_SCHEMAS[("jira-comment", 5)]
    with monkeypatch.context() as stored:
        stored.setitem(
            source_representation._REPRESENTATION_CONTRACTS, ("jira", "comment"),
            canonical_record_profile(earlier.name, earlier.version),
        )
        if record_names_its_schema:
            stored.setattr(jira_adapter, "COMMENT_SCHEMA", earlier)
        return _issue(_COMMENT)


def _named_without_profiles(projection):
    """The projection as a build that named each revision by its Observation and content alone stored it."""
    renamed = {
        revision.id: _stable_id("obsrev", revision.observation_id, revision.semantic_hash)
        for revision in projection.observation_revisions
    }
    return replace(
        projection,
        observation_revisions=tuple(replace(revision, id=renamed[revision.id]) for revision in projection.observation_revisions),
        source_unit_revisions=tuple(
            replace(unit, observation_revision_ids=tuple(sorted(renamed[i] for i in unit.observation_revision_ids)))
            for unit in projection.source_unit_revisions
        ),
        deltas=tuple(
            replace(delta, changed_anchors=tuple(
                replace(anchor, observation_revision_id=renamed[anchor.observation_revision_id])
                for anchor in delta.changed_anchors
            ))
            for delta in projection.deltas
        ),
    )


def _revision(projection, schema_name):
    [revision] = [r for r in projection.observation_revisions if r.evidence_profile.schema_name == schema_name]
    return revision


async def _assert_reads_the_same(base, target, text, revised, *, kind=None):
    """Nothing is authorized, and the Support on ``text`` is rebound exactly to ``revised`` without a model call."""
    authority = _plan(base, target)
    assert isinstance(authority, ExtractionAuthority)
    assert authority.ranges_by_observation_id == {}
    item = _support_item(base, target, text, kind=kind)
    [support] = plan_support_revision(item.context, [item]).supports
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.REBIND_SUPPORT
    [result] = (await RevisionWorkExecutor(client=NoModelClient(), model="fixture").assess_many([item])).values()
    assert result.supported and result.rebound
    assert [part.anchor.observation_revision_id for part in result.memory.resolved_evidence_selection.parts] == [
        revised.id,
    ]


@pytest.mark.parametrize("record_names_its_schema, stored", [
    (True, lambda projection: projection),
    (False, lambda projection: projection),
    (False, _named_without_profiles),
], ids=["record-rewritten", "record-unchanged", "record-unchanged-named-before-profiles"])
async def test_a_comment_under_a_new_schema_is_a_new_revision_that_reads_the_same(
    monkeypatch, tmp_path, record_names_its_schema, stored,
) -> None:
    base = stored(_comment_stored_under_v5(monkeypatch, record_names_its_schema=record_names_its_schema))
    target = _issue(_COMMENT, prior=base)
    old, new = _revision(base, "jira-comment"), _revision(target, "jira-comment")

    assert (old.evidence_profile.schema_version, new.evidence_profile.schema_version) == (5, 6)
    assert new.id != old.id
    assert (new.content == old.content) is not record_names_its_schema
    assert new.observation_id in {anchor.observation_id for anchor in target.deltas[0].changed_anchors}
    await _assert_reads_the_same(base, target, "Cutoff is Friday.", new)

    db = Database(str(tmp_path / "jira.db"))
    await db.connect()
    try:
        await db.upsert_source(
            id=base.source_id, type="jira", name="Jira", config_json="{}",
            access_policy="workspace", owner_user_id="owner-1",
        )
        await db.record_source_projection(base)
        await db.record_source_projection(target)
        current = await db.get_current_source_unit_revision(target.source_units[0].id)
        assert current is not None and new.id in current.observation_revision_ids
        assert (await db.get_source_observation_revisions((new.id,)))[new.id].evidence_profile == new.evidence_profile
    finally:
        await db.close()


async def test_an_edited_message_collected_again_under_a_new_schema_reads_the_same() -> None:
    messages = _messages(CONVERSATION[:1])
    messages[0]["edited_time"] = "2026-07-14T10:05:00Z"
    base = _window(messages, record_schema="teams-message:2")
    target = _window(messages, base, record_schema="teams-message:3")
    old, new = _revision(base, "teams-message"), _revision(target, "teams-message")

    assert (old.evidence_profile.schema_version, new.evidence_profile.schema_version) == (2, 3)
    assert new.id != old.id
    await _assert_reads_the_same(base, target, CONVERSATION[0][1], new)


def test_a_stored_conversation_payload_is_read_under_the_schema_it_was_collected_under() -> None:
    messages = _messages(CONVERSATION[:1])
    base = _window(messages, record_schema="teams-message:2")

    again = _window(messages, base, record_schema="teams-message:2")

    assert _revision(base, "teams-message").evidence_profile == canonical_record_profile("teams-message", 2)
    assert again.observation_revisions == base.observation_revisions
    assert again.deltas[0].changed_anchors == ()


@pytest.mark.parametrize("record_schema", ["teams-message:99", "jira-comment:6", "teams-message", None])
def test_a_conversation_payload_under_a_schema_its_contract_does_not_register_is_not_read(record_schema) -> None:
    with pytest.raises(ValueError, match="record schema"):
        _window(_messages(CONVERSATION[:1]), record_schema=record_schema)


_DIAGRAM = StoredSourceArtifact(
    id="artifact-1", provider_key="docs/diagram.png", parent_observation_type="file_content",
    parent_provider_key="content", provider_revision="blob-1", filename="diagram.png",
    media_type="image/png", size_bytes=17, sha256="a" * 64, uri="/store/artifact-1/diagram.png",
    inference_eligible=True,
)


async def test_an_unchanged_artifact_named_before_profiles_reads_the_same() -> None:
    def guide(base=None):
        return _project(
            "github_repo", _item(relative_path="docs/guide.md"), b"# Guide\n\nRelease after approval.\n", {},
            artifacts=(_DIAGRAM,), **prior_of(base),
        )

    base = _named_without_profiles(guide())
    target = guide(base)
    [old], [new] = (
        [r for r in projection.observation_revisions if r.evidence_profile.name == "binary-artifact"]
        for projection in (base, target)
    )

    assert new.semantic_hash == old.semantic_hash and new.id != old.id
    assert {anchor.observation_id for anchor in target.deltas[0].changed_anchors} == {
        revision.observation_id for revision in target.observation_revisions
    }
    await _assert_reads_the_same(base, target, "", new, kind=EvidenceFragmentKind.ARTIFACT)


def _markdown(text, base=None):
    item = ContentItem(
        item_id="local-guide", title="Guide", source_url="file:///vault/docs/guide.md",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc), version="1",
        extra={"relative_path": "docs/guide.md", "file_lineage_id": "file-77"},
    )
    raw = {"vault_id": "vault-a", "relative_path": "docs/guide.md", "file_lineage_id": "file-77", "markdown": text}
    return project_source_item(
        source_id="src-md", source_type="local_markdown", run_id="run-md", item=item,
        raw=RawContent(item=item, body=json.dumps(raw).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body=text), **prior_of(base),
    )


def test_an_unchanged_document_under_a_new_markdown_profile_is_read_again_whole(monkeypatch) -> None:
    text = "# Release\n\nUse approval before release.\n"
    base = _markdown(text)
    bumped = replace(MARKDOWN_STRUCTURAL_PROFILE, version=2)
    monkeypatch.setitem(source_representation._REPRESENTATION_CONTRACTS, ("local_markdown", "file_content"), bumped)
    monkeypatch.setitem(
        source_representation._SUPPORTED_REPRESENTATION_CONTRACTS, bumped,
        replace(source_representation.representation_contract_for_profile(MARKDOWN_STRUCTURAL_PROFILE), profile=bumped),
    )
    monkeypatch.setitem(
        evidence_fragments._PROFILE_COMPILERS, ("markdown-structural", 2),
        evidence_fragments._PROFILE_COMPILERS[("markdown-structural", 1)],
    )

    target = _markdown(text, base)
    [old], [new] = base.observation_revisions, target.observation_revisions

    assert new.content == old.content and new.id != old.id
    assert [anchor.observation_id for anchor in target.deltas[0].changed_anchors] == [new.observation_id]
    authority = _plan(base, target)
    assert isinstance(authority, ExtractionAuthority)
    assert authority.ranges_by_observation_id == {new.observation_id: None}
