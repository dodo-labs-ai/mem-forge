"""Every registered source type keeps a quality snapshot that its own code still reads.

The public quality cohort (``scripts/compose_smoke.sh --quality``) syncs one
Source per source type from ``evals/quality/quick/snapshots/<source_type>``,
revision 1 and then revision 2, so a change to how a type is read is gated by
a model run like Markdown's. These tests keep that cohort complete: a
registered type without a snapshot fails here, and so does a snapshot its
type no longer normalizes and projects, read through the same snapshot Gene or
snapshot provider the cohort replays it with.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memforge.conversations.membership import PlannedUnit
from memforge.conversations.port import Interval
from memforge.conversations.projection import unit_content_item, unit_payload, unit_raw_content
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.units import Member, UnitKind
from memforge.evals.source_snapshots import (
    SNAPSHOT_CONFIG_FIELD,
    SNAPSHOT_ROOT_VARIABLE,
    item_dirs,
    revision_dirs,
)
from memforge.genes import create_gene, gene_exports_pdf
from memforge.genes.family import CollectionFamilyKind
from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import SourceProjection
from memforge.source_types import SourceTypeDescriptor, registered_source_types
from tests.test_compose_smoke import smoke

QUICK = Path(__file__).resolve().parents[1] / "evals" / "quality" / "quick"
SNAPSHOTS = QUICK / "snapshots"
SOURCE_FILE = "source.json"
REPLAYED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
DESCRIPTORS = registered_source_types()


def _missing_snapshot(descriptor: SourceTypeDescriptor, problem: str) -> str:
    """Tell a contributor what to add, and which snapshot of the same family to copy."""

    examples = [
        other.source_type
        for other in DESCRIPTORS
        if other.family is descriptor.family and other.source_type != descriptor.source_type
        and (SNAPSHOTS / other.source_type).is_dir()
    ]
    example = f"copy evals/quality/quick/snapshots/{examples[0]}/ and replace its content" if examples else (
        "see CONTRIBUTING.md, 'Quality snapshots'"
    )
    return (
        f"source type {descriptor.source_type!r} {problem}: add evals/quality/quick/snapshots/"
        f"{descriptor.source_type}/ with {SOURCE_FILE} and at least one item with revisions 1 and 2 "
        f"({example}), then add its cases to evals/quality/quick/cohort.json"
    )


def _set_problem(source_type: str) -> str | None:
    set_dir = SNAPSHOTS / source_type
    if not set_dir.is_dir():
        return "has no quality snapshot"
    if not (set_dir / SOURCE_FILE).is_file():
        return f"has a quality snapshot without {SOURCE_FILE}"
    revisions = [[number for number, _ in revision_dirs(item_dir)] for item_dir in item_dirs(set_dir)]
    if not any(numbers[:2] == [1, 2] for numbers in revisions):
        return "has no snapshot item with revisions 1 and 2"
    return None


def test_every_registered_source_type_has_a_quality_snapshot() -> None:
    problems = [
        _missing_snapshot(descriptor, problem)
        for descriptor in DESCRIPTORS
        if (problem := _set_problem(descriptor.source_type)) is not None
    ]
    assert not problems, "\n".join(problems)


def test_every_snapshot_set_names_a_registered_source_type() -> None:
    registered = {descriptor.source_type for descriptor in DESCRIPTORS}
    assert {path.name for path in SNAPSHOTS.iterdir() if path.is_dir()} <= registered


def _config(source_type: str, revision: int) -> dict:
    config = json.loads((SNAPSHOTS / source_type / SOURCE_FILE).read_text(encoding="utf-8"))["config"]
    return {
        **config,
        SNAPSHOT_CONFIG_FIELD: {"set": source_type, "revision": revision, "replayed_at": REPLAYED_AT.isoformat()},
    }


def _revisions(source_type: str) -> list[int]:
    return sorted(
        {number for item_dir in item_dirs(SNAPSHOTS / source_type) for number, _ in revision_dirs(item_dir)}
    )


def _assert_reads(projection: SourceProjection, label: str) -> None:
    assert projection.source_units and projection.observation_revisions, f"{label} projects no Observation"
    for revision in projection.observation_revisions:
        index = build_revision_fragment_index(revision)
        assert index.complete and index.fragments, (
            f"{label}: the {revision.evidence_profile} reading of an Observation fails: {index.errors}"
        )


async def _document_projections(source_type: str, revision: int) -> dict[str, tuple]:
    """Each item of one revision, acquired by the snapshot Gene and read by the type's own code."""

    descriptor = next(d for d in DESCRIPTORS if d.source_type == source_type)
    gene = create_gene(source_type, _config(source_type, revision), f"src-snapshot-{source_type}")
    await gene.authenticate()
    read = {}
    try:
        async for item in gene.discover(None):
            raw = await gene.fetch(item)
            normalized = await gene.normalize(raw)
            assert normalized.markdown_body.strip(), f"{source_type} item {item.item_id} normalizes to nothing"
            for artifact in raw.artifacts:
                async with gene.open_source_artifact(artifact) as download:
                    body = b"".join([chunk async for chunk in download.chunks])
                assert len(body) == artifact.declared_size_bytes
            if gene_exports_pdf(gene):
                assert await gene.fetch_pdf(item), f"{source_type} exports a PDF; add export.pdf to {item.item_id}"
            read[item.item_id] = (item, raw, normalized)
    finally:
        await gene.close()
    assert read, f"revision {revision} of the {source_type} snapshot holds no item"
    assert descriptor.family is CollectionFamilyKind.DOCUMENT
    return read


async def _conversation_reads(source_type: str, revision: int) -> dict[str, tuple]:
    """Each conversation of one revision as one Unit, from the snapshot provider the device executor runs."""

    descriptor = next(d for d in DESCRIPTORS if d.source_type == source_type)
    config = _config(source_type, revision)
    gene = create_gene(source_type, config, f"src-snapshot-{source_type}")
    assert descriptor.conversation_provider is not None
    read = {}
    async with descriptor.conversation_provider.open(config) as provider:
        listing = await provider.list_conversations()
        schema = provider.record_schema.qualified_name
        for ref in listing.conversations:
            poll = await provider.poll(ref, Interval(start=REPLAYED_AT - timedelta(days=30)))
            assert poll.messages, f"{source_type} conversation {ref.conversation_id} returns no message"
            assert all(message.created_at < REPLAYED_AT for message in poll.messages)
            members = tuple(
                Member(
                    identity=MessageIdentity(ref.conversation_id, message.message_id),
                    created_at=message.created_at,
                    record=message.record,
                    revision=record_revision(message.record),
                    record_schema=schema,
                )
                for message in sorted(poll.messages, key=lambda message: message.created_at)
            )
            for message in poll.messages:
                assert message.record.get("representation") == schema, (
                    f"{source_type} message {message.message_id} is not a {schema} record"
                )
                for artifact in message.artifacts:
                    assert (await provider.read_artifact(ref, message, artifact)).content
            unit = PlannedUnit(
                kind=UnitKind.WINDOW, provider_key=f"snapshot-window:{ref.conversation_id}",
                conversation=ref.label, members=members,
            )
            item = unit_content_item(unit, source_id=f"src-snapshot-{source_type}")
            raw = unit_raw_content(item, unit_payload(unit, record_schema=schema), with_artifacts=False)
            read[ref.conversation_id] = (item, raw, await gene.normalize(raw))
    return read


CONVERSATION_TYPES = [
    descriptor for descriptor in DESCRIPTORS
    if descriptor.family is not CollectionFamilyKind.DOCUMENT and (SNAPSHOTS / descriptor.source_type).is_dir()
]


@pytest.mark.parametrize("descriptor", CONVERSATION_TYPES, ids=lambda descriptor: descriptor.source_type)
def test_conversation_text_states_no_iso_date(descriptor) -> None:
    """Replay moves message times but not text, so a date written in the text would fall behind its message."""

    def text(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from text(item)
        elif isinstance(value, list):
            for item in value:
                yield from text(item)
        elif isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                yield value

    dated = [
        f"{path.relative_to(SNAPSHOTS)}: {message['message_id']}"
        for item_dir in item_dirs(SNAPSHOTS / descriptor.source_type)
        for _, revision_dir in revision_dirs(item_dir)
        for path in [revision_dir / "messages.json"]
        for message in json.loads(path.read_text(encoding="utf-8"))
        if any(re.search(r"\b\d{4}-\d{2}-\d{2}\b", value) for value in text(message["record"]))
    ]
    assert not dated, f"message text states a calendar date: {dated}"


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda descriptor: descriptor.source_type)
async def test_a_source_type_snapshot_normalizes_and_projects_in_every_revision(descriptor, monkeypatch) -> None:
    source_type = descriptor.source_type
    problem = _set_problem(source_type)
    if problem is not None:
        pytest.fail(_missing_snapshot(descriptor, problem))
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(SNAPSHOTS))
    read = _document_projections if descriptor.family is CollectionFamilyKind.DOCUMENT else _conversation_reads
    previous: dict[str, SourceProjection] = {}
    revised = set()
    for revision in _revisions(source_type):
        current = {}
        for key, (item, raw, normalized) in (await read(source_type, revision)).items():
            prior = previous.get(key)
            projection = project_source_item(
                source_id=f"src-snapshot-{source_type}",
                source_type=source_type,
                run_id=f"run-{revision}",
                item=item,
                raw=raw,
                normalized=normalized,
                prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
                prior_observation_revisions=(
                    {revision.observation_id: revision for revision in prior.observation_revisions} if prior else None
                ),
            )
            _assert_reads(projection, f"{source_type} revision {revision} of {key}")
            if prior is not None and projection.source_unit_revisions[0].id != prior.source_unit_revisions[0].id:
                revised.add(key)
            current[key] = projection
        previous = {**previous, **current}
    assert revised, f"no item of the {source_type} snapshot changes in revision 2, so no Support case can pin it"


def _item_keys(descriptor: SourceTypeDescriptor) -> dict[str, Path]:
    """Each item directory of a set, by the item id or conversation id it holds."""

    keys = {}
    for item_dir in item_dirs(SNAPSHOTS / descriptor.source_type):
        spec = json.loads((revision_dirs(item_dir)[0][1] / "item.json").read_text(encoding="utf-8"))
        key = spec["item"]["item_id"] if "item" in spec else spec["conversation"]["conversation_id"]
        keys[key] = item_dir
    return keys


async def _case_titles(descriptor: SourceTypeDescriptor) -> dict[str, set[str]]:
    """The title of each item's Document at the set's last revision, as the cohort's cases name it.

    A conversation item names each thread its messages form, with the time
    replay gives its first message.
    """

    source_type = descriptor.source_type
    revision = _revisions(source_type)[-1]
    if descriptor.family is CollectionFamilyKind.DOCUMENT:
        return {key: {item.title} for key, (item, _, _) in (await _document_projections(source_type, revision)).items()}
    assert descriptor.conversation_provider is not None
    titles: dict[str, set[str]] = {}
    async with descriptor.conversation_provider.open(_config(source_type, revision)) as provider:
        schema = provider.record_schema.qualified_name
        for ref in (await provider.list_conversations()).conversations:
            poll = await provider.poll(ref, Interval(start=REPLAYED_AT - timedelta(days=30)))
            messages = sorted(poll.messages, key=lambda message: message.created_at)
            for root in {message.thread_root_id for message in messages if message.thread_root_id}:
                thread = [message for message in messages if root in (message.message_id, message.thread_root_id)]
                unit = PlannedUnit(
                    kind=UnitKind.THREAD, provider_key=f"snapshot-thread:{root}", conversation=ref.label,
                    thread_root_id=root,
                    members=tuple(
                        Member(
                            identity=MessageIdentity(ref.conversation_id, message.message_id),
                            created_at=message.created_at, record=message.record,
                            revision=record_revision(message.record), record_schema=schema,
                        )
                        for message in thread
                    ),
                )
                titles.setdefault(ref.conversation_id, set()).add(unit_content_item(unit, source_id="src").title)
    return titles


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=lambda descriptor: descriptor.source_type)
async def test_every_snapshot_item_pins_its_cases_in_the_cohort(descriptor, monkeypatch) -> None:
    """Each item pins a derivation case, each revised item a Support case, each named as replay titles it."""

    source_type = descriptor.source_type
    if _set_problem(source_type) is not None:
        pytest.skip("the snapshot is missing; test_every_registered_source_type_has_a_quality_snapshot says what to add")
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(SNAPSHOTS))
    cohort = json.loads((QUICK / "cohort.json").read_text(encoding="utf-8"))
    titles = await _case_titles(descriptor)
    named = {title for item_titles in titles.values() for title in item_titles}
    replayed = {
        list_name: [
            smoke._replayed_title(case["title"], SNAPSHOTS / source_type, REPLAYED_AT.isoformat())
            for case in cohort[list_name]
            if case.get("snapshot") == source_type
        ]
        for list_name in ("cases", "support_cases")
    }
    assert set(replayed["cases"]) | set(replayed["support_cases"]) <= named, (
        f"a {source_type} snapshot case names no Document of the set; its Documents are {sorted(named)}"
    )
    keys = _item_keys(descriptor)
    assert set(titles) == set(keys), (
        f"a {source_type} conversation item forms no thread for a case to name; give a message a reply"
    )
    revised = {key for key, item_dir in keys.items() if len(revision_dirs(item_dir)) > 1}
    for key, item_titles in titles.items():
        assert item_titles & set(replayed["cases"]), f"{source_type} item {key} has no derivation case in cohort.json"
        if key in revised:
            assert item_titles & set(replayed["support_cases"]), (
                f"{source_type} item {key} has a second revision and no Support case in cohort.json"
            )
