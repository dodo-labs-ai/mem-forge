from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import memforge
import memforge.main as main
import memforge.runtime as runtime_module
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.genes.conversation_gene import CollectionAttempt
from memforge.genes.teams_gene import TeamsGene
from memforge.main import cli
from memforge.pipeline.collection_family import (
    ConversationFamily,
    DocumentFamily,
    collection_family,
)
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore
from tests.gene_fixture import FakeGene

FAMILY_CLASSES = {"ConversationGene", "DocumentGene"}
TEAMS_CONFIG = {"conversation_ids": ["19:chat@thread.v2"], "initial_history_days": 30}
# The only modules that ask a Gene class which family it belongs to.
FAMILY_CLASSIFIERS = {Path("genes/family.py"), Path("genes/contract.py")}


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "store.db"))
    await database.connect()
    yield database
    await database.close()


def _family(gene, db, tmp_path):
    return collection_family(gene, db=db, doc_store=LocalDocumentStore(str(tmp_path / "docs")), source_id="src-1")


async def test_a_run_takes_the_family_of_its_gene(db, tmp_path) -> None:
    assert isinstance(_family(FakeGene(), db, tmp_path), DocumentFamily)
    assert isinstance(_family(TeamsGene(dict(TEAMS_CONFIG), "src-1"), db, tmp_path), ConversationFamily)


async def test_a_gene_of_neither_family_has_no_sync_run(db, tmp_path) -> None:
    with pytest.raises(ValueError, match="no collection family"):
        _family(AgentSessionGene({}, "src-1"), db, tmp_path)


async def test_a_conversation_run_needs_its_collection_attempt(db, tmp_path) -> None:
    family = _family(TeamsGene(dict(TEAMS_CONFIG), "src-1"), db, tmp_path)

    with pytest.raises(ValueError, match="collection attempt"):
        family.prepare(None)
    family.prepare(CollectionAttempt("default", "src-1:attempt:1"))
    assert family.gene.collection_attempt_id == "src-1:attempt:1"


async def test_a_document_run_needs_no_collection_attempt(db, tmp_path) -> None:
    _family(FakeGene(), db, tmp_path).prepare(None)


def test_only_the_family_module_asks_a_gene_class_for_its_family() -> None:
    package = Path(memforge.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path.relative_to(package) in FAMILY_CLASSIFIERS:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"isinstance", "issubclass"}
                and len(node.args) == 2
            ):
                continue
            names = {n.id for n in ast.walk(node.args[1]) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(node.args[1]) if isinstance(n, ast.Attribute)}
            if names & FAMILY_CLASSES:
                offenders.append(f"{path.relative_to(package)}:{node.lineno}")
    assert offenders == []


def test_the_cli_syncs_only_sources_with_a_sync_run(monkeypatch, tmp_path) -> None:
    sources = [
        {"id": "src-capture", "name": "Captured", "type": "agent_session", "status": "active"},
        {"id": "src-chat", "name": "Chat", "type": "teams", "status": "active"},
        {"id": "src-wiki", "name": "Wiki", "type": "confluence", "status": "active"},
    ]
    synced = []

    class FakeDb:
        async def list_sources(self):
            return sources

        async def close(self):
            return None

    async def fake_get_db(config):
        return FakeDb()

    async def fake_build_sync_runtime(db, config):
        return SimpleNamespace()

    async def fake_run_source_sync(*, source, **kwargs):
        synced.append(source["id"])
        return SimpleNamespace(
            last_sync_status="success",
            docs_processed=0,
            docs_updated=0,
            docs_failed=0,
            memories_extracted=0,
            skipped_reading_group_count=0,
            error_message=None,
        )

    monkeypatch.setattr(main, "_get_db", fake_get_db)
    monkeypatch.setattr(runtime_module, "build_sync_runtime", fake_build_sync_runtime)
    monkeypatch.setattr(runtime_module, "run_source_sync", fake_run_source_sync)

    result = CliRunner().invoke(cli, ["sync"], env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")})

    assert result.exit_code == 0, result.output
    assert synced == ["src-wiki"]
    assert "Skipping Captured" in result.output
    assert "Skipping Chat" in result.output


class _PdfGene(FakeGene):
    @classmethod
    def metadata(cls):
        return type(FakeGene.metadata())(**{**FakeGene.metadata().__dict__, "name": "pdf_test"})


class _PdfStore:
    def __init__(self, missing: int) -> None:
        self.missing = missing
        self.counted = 0

    async def count_missing_pdf_uris(self, source_id: str) -> int:
        self.counted += 1
        return self.missing


async def test_document_discovery_reads_from_the_cursor_unless_a_full_discovery_is_due(monkeypatch, tmp_path) -> None:
    from datetime import datetime, timezone

    from memforge.pipeline.collection_family import DEFAULT_INCREMENTAL_SYNC_OVERLAP
    from tests.source_type_fixture import register_test_source_type

    register_test_source_type(monkeypatch, "pdf_test", _PdfGene, exports_pdf=True)
    cursor = datetime(2026, 10, 22, 8, 0, tzinfo=timezone.utc)
    state = SimpleNamespace(last_sync_at=cursor)

    async def since(store, *, indexed=frozenset({"page-1"}), full=False, existing=state):
        family = collection_family(
            _PdfGene(), db=store, doc_store=LocalDocumentStore(str(tmp_path / "docs")), source_id="src-1"
        )
        return await family.since(existing_state=existing, indexed_doc_ids=set(indexed), full_discovery=full)

    assert await since(_PdfStore(0)) == cursor - DEFAULT_INCREMENTAL_SYNC_OVERLAP
    assert await since(_PdfStore(2)) is None
    assert await since(_PdfStore(0), indexed=frozenset()) is None
    assert await since(_PdfStore(0), existing=None) is None
    skipped = _PdfStore(2)
    assert await since(skipped, full=True) is None
    assert skipped.counted == 0
