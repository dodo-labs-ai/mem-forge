"""The Memory vector index belongs to one Embedding Profile, and only its model's vectors reach it."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import pytest

from memforge import runtime
from memforge.config import AppConfig
from memforge.llm.bindings import (
    ModelBinding,
    ValidationResult,
    build_binding,
    embedding_identity,
    embedding_profile,
)
from memforge.llm.embedding import (
    EmbeddingIndexMismatch,
    EmbeddingRuntime,
    EmbeddingSettings,
    embed_documents_for_index,
    embed_query_for_index,
)
from memforge.llm.embedding_profile import EmbeddingIdentity, EmbeddingProfile
from memforge.llm.providers import ModelConfigError
from memforge.memory.repair import MemoryIndexRepairer
from memforge.model_settings import ModelSettings
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.vector_index import (
    STORED_VECTORS_COLLECTION,
    collection_name_for,
    create_index_collection,
    delete_index_collection,
    open_memory_vector_index,
    settle_memory_vector_index,
)
from memforge.runtime import DefaultRuntimeProvider
from memforge.storage.adapters.sqlite import SqliteVectorStore
from memforge.storage.database import Database
from memforge.vector_index_errors import NoVectorIndex
from tests.embedding_fixture import embedding_invocation, texts_transport
from tests.vector_index_fixture import EmbeddingModels

ENDPOINT = "https://embed.example.test/v1"
IDENTITY_A = EmbeddingIdentity(provider="openai_compatible", route="openai/embed-a", endpoint=ENDPOINT, model_id="embed-a")
IDENTITY_B = EmbeddingIdentity(provider="openai_compatible", route="openai/embed-b", endpoint=ENDPOINT, model_id="embed-b")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "index.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def models(monkeypatch) -> EmbeddingModels:
    return EmbeddingModels(monkeypatch, **{"embed-a": 3, "embed-b": 4, "embed-c": 3})


def _config(tmp_path: Path, model: str = "embed-a") -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = ""
    config.llm.embedding_model = model
    config.llm.embedding_base_url = ENDPOINT
    config.llm.embedding_api_key = "embedding-key"
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


def _memory(memory_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        created_at=now,
        updated_at=now,
        status="active",
    )


async def _write_memory(db: Database, config: AppConfig, memory_id: str, content: str) -> None:
    """Write one Memory and its vector the way a sync does."""
    now = datetime.now(timezone.utc)
    await db.upsert_source("src-index", "confluence", "Index", "{}", access_policy="workspace", owner_user_id="dev")
    await db.upsert_document(
        DocumentRecord(
            doc_id=f"doc-{memory_id}",
            source="src-index",
            source_url=f"https://docs.example.test/{memory_id}",
            title=memory_id,
            space_or_project="PAY",
            author="author",
            last_modified=now,
            labels=[],
            version="1",
            content_hash="hash",
            token_count=1,
            last_synced=now,
        )
    )
    sync_runtime = await runtime.build_sync_runtime(db, config)
    outcome = await sync_runtime.memory_store.insert_memory(
        _memory(memory_id, content),
        f"doc-{memory_id}",
        "manual",
        source_updated_at=None,
    )
    assert outcome is None


def _stored_vectors(chroma_path: str, name: str) -> dict[str, list[float]]:
    collection = chromadb.PersistentClient(path=chroma_path).get_collection(name)
    stored = collection.get(include=["embeddings"])
    return {
        record_id: [round(float(value), 4) for value in vector]
        for record_id, vector in zip(stored["ids"], stored["embeddings"])
    }


def _collection_names(chroma_path: str) -> list[str]:
    return sorted(collection.name for collection in chromadb.PersistentClient(path=chroma_path).list_collections())


async def _vector_hits(db: Database, config: AppConfig, query: str = "unrelated words only") -> list[str]:
    """Search with a query no keyword matches, so that every hit comes from the vector channel."""
    engine = await runtime.build_search_engine(db, config)
    result = await engine.search(query, top_k=5)
    return [hit.memory_id for hit in result["results"]]


# -- The profile and its fingerprint ------------------------------------------------


def test_a_profile_is_identified_by_its_six_fields() -> None:
    profile = IDENTITY_A.profile(3)

    assert profile.fields() == {
        "provider": "openai_compatible",
        "route": "openai/embed-a",
        "endpoint": ENDPOINT,
        "model_id": "embed-a",
        "requested_dimensions": None,
        "dimension": 3,
    }
    assert profile.fingerprint == IDENTITY_A.profile(3).fingerprint
    assert EmbeddingProfile.from_fields(profile.fields()) == profile


@pytest.mark.parametrize(
    "other",
    [
        EmbeddingIdentity("openai", "openai/embed-a", ENDPOINT, "embed-a").profile(3),
        EmbeddingIdentity("openai_compatible", "ollama/embed-a", ENDPOINT, "embed-a").profile(3),
        EmbeddingIdentity("openai_compatible", "openai/embed-a", "https://other.example.test/v1", "embed-a").profile(3),
        EmbeddingIdentity("openai_compatible", "openai/embed-a", ENDPOINT, "embed-A").profile(3),
        EmbeddingIdentity("openai_compatible", "openai/embed-a", ENDPOINT, "embed-a", 3).profile(3),
        IDENTITY_A.profile(4),
    ],
    ids=["provider", "route", "endpoint", "model", "requested_dimensions", "observed_dimension"],
)
def test_every_field_of_a_profile_changes_its_fingerprint(other: EmbeddingProfile) -> None:
    assert other.fingerprint != IDENTITY_A.profile(3).fingerprint


def _binding_at(endpoint: str) -> ModelBinding:
    return build_binding(
        kind="embedding",
        provider="openai_compatible",
        endpoint=endpoint,
        model_id="embed-a",
        secret_ref="secret:one",
        config_source="settings",
    )


@pytest.mark.parametrize(
    "written",
    ["https://Embed.Example.Test/v1", "HTTPS://embed.example.test:443/v1", "https://embed.example.test./v1/"],
    ids=["host_case", "scheme_case_and_default_port", "trailing_dot_and_slash"],
)
def test_an_endpoint_written_another_way_is_the_same_vector_space(written: str) -> None:
    assert embedding_identity(_binding_at(written)) == IDENTITY_A


@pytest.mark.parametrize(
    "written",
    ["https://embed.example.test/V1", "https://embed.example.test:8443/v1", "http://embed.example.test/v1"],
    ids=["path_case", "port", "scheme"],
)
def test_another_address_is_another_vector_space(written: str) -> None:
    identity = embedding_identity(_binding_at(written))

    assert identity.endpoint == written
    assert identity.profile(3).fingerprint != IDENTITY_A.profile(3).fingerprint


def test_a_binding_has_a_profile_only_while_a_passed_check_vouches_for_it() -> None:
    binding = build_binding(
        kind="embedding",
        provider="openai_compatible",
        endpoint=ENDPOINT,
        model_id="embed-a",
        secret_ref="secret:one",
        config_source="settings",
    )
    passed = ValidationResult(
        checked_at="2026-10-14T00:00:00+00:00",
        digest=binding.digest,
        facts={"dimension": 3, "provider_model": "reported-by-a-gateway"},
    )
    failed = ValidationResult(checked_at=passed.checked_at, digest=binding.digest, error_code="timeout")
    stale = ValidationResult(checked_at=passed.checked_at, digest="another-digest", facts={"dimension": 3})

    assert embedding_profile(binding) is None
    assert embedding_profile(replace(binding, validation=failed)) is None
    assert embedding_profile(replace(binding, validation=stale)) is None
    profile = embedding_profile(replace(binding, validation=passed))
    assert profile == embedding_identity(binding).profile(3)
    assert profile.fingerprint == IDENTITY_A.profile(3).fingerprint
    assert "reported-by-a-gateway" not in str(profile.fields())


@pytest.mark.parametrize(
    ("identity", "dimension", "calibrated"),
    [
        (EmbeddingIdentity("openai", "openai/text-embedding-3-small", "https://api.openai.com/v1", "text-embedding-3-small"), 1536, True),
        (EmbeddingIdentity("openai", "openai/text-embedding-3-small", "https://api.openai.com/v1", "text-embedding-3-small", 512), 512, False),
        (EmbeddingIdentity("openai", "openai/text-embedding-3-large", "https://api.openai.com/v1", "text-embedding-3-large"), 3072, False),
        (EmbeddingIdentity("ollama", "ollama/nomic-embed-text", "http://localhost:11434", "nomic-embed-text"), 768, False),
    ],
    ids=["calibrated_model", "shortened", "another_openai_model", "local_model"],
)
def test_a_profile_says_whether_the_similarity_thresholds_were_tuned_on_its_model(identity, dimension, calibrated) -> None:
    assert identity.profile(dimension).thresholds_calibrated is calibrated


# -- Storage: each index once, one in use -------------------------------------------


def _record(profile: EmbeddingProfile, name: str, *, index_id: str, adopted: bool = False) -> dict:
    return {
        "id": index_id,
        "fingerprint": profile.fingerprint,
        "collection_name": name,
        "adopted": adopted,
        "dimension": profile.dimension,
        "profile": profile.fields(),
        "vector_count": 0,
    }


async def test_an_installation_has_at_most_one_index_in_use(db) -> None:
    first = IDENTITY_A.profile(3)

    assert await db.get_active_vector_index() is None
    assert await db.record_active_vector_index(_record(first, "memories-first", index_id="first")) is True
    assert await db.record_active_vector_index(_record(IDENTITY_B.profile(4), "memories-second", index_id="second")) is False

    in_use = await db.get_active_vector_index()
    assert (in_use["id"], in_use["fingerprint"], in_use["collection_name"]) == ("first", first.fingerprint, "memories-first")
    assert (in_use["state"], in_use["adopted"], in_use["dimension"]) == ("active", False, 3)
    assert in_use["profile"] == first.fields()
    assert in_use["activated_at"] is not None
    with sqlite3.connect(db.db_path) as raw, pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            """INSERT INTO vector_indexes (id, fingerprint, collection_name, state, dimension, created_at)
               VALUES ('forced', 'f', 'c', 'active', 3, '2026-10-14T00:00:00+00:00')"""
        )


async def test_the_migration_adds_the_index_table_to_a_database_that_has_none(tmp_path) -> None:
    database = Database(str(tmp_path / "earlier.db"))
    await database.connect()
    await database.db.execute("DROP TABLE vector_indexes")
    await database.db.execute("DELETE FROM schema_migrations WHERE version = 114")
    await database.db.commit()
    await database.close()

    reopened = Database(str(tmp_path / "earlier.db"))
    await reopened.connect()
    try:
        assert await reopened.get_active_vector_index() is None
        with sqlite3.connect(reopened.db_path) as raw:
            applied = raw.execute("SELECT description FROM schema_migrations WHERE version = 114").fetchone()
        assert applied == ("Record each Memory vector index with the fingerprint it was built with",)
    finally:
        await reopened.close()


class _PagedVectors:
    """A collection that hands out its vectors a page at a time."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def get(self, *, include, limit, offset):
        page = self._vectors[offset : offset + limit]
        return {"ids": [f"mem-{offset + index}" for index in range(len(page))], "embeddings": page}


def test_deleting_an_index_collection_removes_its_vector_files(tmp_path) -> None:
    chroma_path = str(tmp_path / "chroma")
    kept = create_index_collection(chroma_path, "memories-kept")
    kept.upsert(ids=["mem-kept"], embeddings=[[1.0, 0.0, 0.0]])
    kept_files = {entry.name for entry in Path(chroma_path).iterdir() if entry.is_dir()}
    dropped = create_index_collection(chroma_path, "memories-dropped")
    dropped.upsert(ids=["mem-1", "mem-2"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert len({entry.name for entry in Path(chroma_path).iterdir() if entry.is_dir()}) == 2

    delete_index_collection(chroma_path, "memories-dropped")
    delete_index_collection(chroma_path, "memories-never-created")

    assert {entry.name for entry in Path(chroma_path).iterdir() if entry.is_dir()} == kept_files
    assert _collection_names(chroma_path) == ["memories-kept"]
    assert _stored_vectors(chroma_path, "memories-kept") == {"mem-kept": [1.0, 0.0, 0.0]}
    # The name is free again, and a collection made under it starts empty.
    again = create_index_collection(chroma_path, "memories-dropped")
    again.upsert(ids=["mem-3"], embeddings=[[0.0, 0.0, 1.0]])
    assert again.query(query_embeddings=[[0.0, 0.0, 1.0]], n_results=1)["ids"] == [["mem-3"]]


def test_verification_counts_the_vectors_and_those_of_the_declared_length_with_finite_values(monkeypatch) -> None:
    from memforge.retrieval import vector_index

    monkeypatch.setattr(vector_index, "_VERIFY_PAGE_SIZE", 2)
    vectors = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0],
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
        [0.0, 0.0, 0.0],
    ]

    assert vector_index.count_valid_vectors(_PagedVectors(vectors), dimension=3) == (5, 2)
    assert vector_index.count_valid_vectors(_PagedVectors([]), dimension=3) == (0, 0)


# -- One way to open the index --------------------------------------------------------


async def test_opening_returns_the_index_in_use_with_its_fingerprint(db, tmp_path) -> None:
    chroma_path = str(tmp_path / "chroma")
    profile = IDENTITY_A.profile(3)

    assert await open_memory_vector_index(db, chroma_path) is None
    assert not Path(chroma_path).exists()
    await settle_memory_vector_index(db, chroma_path, profile)

    index = await open_memory_vector_index(db, chroma_path)
    assert index.fingerprint == profile.fingerprint
    assert index.name == collection_name_for(profile.fingerprint) == f"memories-{profile.fingerprint[:32]}"
    assert (index.dimension, index.adopted) == (3, False)
    assert index.collection.name == index.name
    assert (await open_memory_vector_index(db, chroma_path)).collection is index.collection


async def test_every_default_reader_and_writer_opens_the_index_through_the_one_function(
    db, tmp_path, models, monkeypatch
) -> None:
    from memforge.retrieval import vector_index

    opened: list[str] = []
    open_index = vector_index.open_memory_vector_index

    async def recording(database, chroma_path):
        opened.append(chroma_path)
        return await open_index(database, chroma_path)

    monkeypatch.setattr(vector_index, "open_memory_vector_index", recording)
    config = _config(tmp_path)

    await _write_memory(db, config, "mem-opened", "Deploy through the release pipeline")
    after_write = len(opened)
    await _vector_hits(db, config)
    after_search = len(opened)
    health = await runtime.check_runtime_health(db, config)

    assert after_write > 0 and after_search > after_write and len(opened) > after_search
    assert set(opened) == {config.storage.chroma_path}
    assert health.index_consistency.status == "ok"
    source = Path(runtime.__file__).parent
    direct = [
        path.name
        for path in source.rglob("*.py")
        if "get_or_create_collection" in path.read_text(encoding="utf-8") and path.name != "vector_index.py"
    ]
    assert direct == []


# -- The first index of an installation ----------------------------------------------


async def test_an_installation_with_no_vectors_gets_a_new_index_directly(db, tmp_path, models) -> None:
    config = _config(tmp_path)

    await _write_memory(db, config, "mem-first", "Deploy through the release pipeline")

    profile = IDENTITY_A.profile(3)
    in_use = await db.get_active_vector_index()
    assert in_use["fingerprint"] == profile.fingerprint
    assert in_use["collection_name"] == collection_name_for(profile.fingerprint)
    assert in_use["adopted"] is False
    assert _collection_names(config.storage.chroma_path) == [in_use["collection_name"]]
    assert list(_stored_vectors(config.storage.chroma_path, in_use["collection_name"])) == ["mem-first"]
    assert await _vector_hits(db, config) == ["mem-first"]


def _vectors_from_before_indexes_were_recorded(chroma_path: str, *, dimension: int) -> None:
    """Leave the vectors of an installation that never recorded an index."""
    collection = chromadb.PersistentClient(path=chroma_path).get_or_create_collection(
        name=STORED_VECTORS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=["mem-stored"],
        embeddings=[[1.0] * dimension],
        metadatas=[{"status": "active", "visibility": "workspace", "memory_type": "fact", "project_key": ""}],
    )


async def test_stored_vectors_of_the_models_length_are_adopted_where_they_are(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=3)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))

    hits = await _vector_hits(db, config)

    in_use = await db.get_active_vector_index()
    assert in_use["collection_name"] == STORED_VECTORS_COLLECTION
    assert in_use["adopted"] is True
    assert in_use["fingerprint"] == IDENTITY_A.profile(3).fingerprint
    assert in_use["vector_count"] == 1
    assert _collection_names(config.storage.chroma_path) == [STORED_VECTORS_COLLECTION]
    assert _stored_vectors(config.storage.chroma_path, STORED_VECTORS_COLLECTION) == {"mem-stored": [1.0, 1.0, 1.0]}
    assert hits == ["mem-stored"]
    assert models.texts == ["unrelated words only"]


async def test_stored_vectors_of_another_length_are_not_adopted_and_a_reindex_is_required(
    db, tmp_path, models
) -> None:
    config = _config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=8)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())

    hits = await _vector_hits(db, config)
    await settings.validate_active(db, "embedding")
    view = await settings.vector_index(db)

    assert hits == []
    assert await db.get_active_vector_index() is None
    assert (view.status, view.reindex_required, view.reindex_target) == ("reindex_required", True, "active")
    assert (await settings.health(db))["embedding"].vector_index == "reindex_required"
    assert _stored_vectors(config.storage.chroma_path, STORED_VECTORS_COLLECTION) == {"mem-stored": [1.0] * 8}
    with pytest.raises(NoVectorIndex):
        await _write_memory(db, config, "mem-waits", "A write that waits for the index")
    assert await db.get_memory("mem-waits") is None


async def test_validating_the_active_model_settles_the_first_index(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    settings = ModelSettings(config, DefaultRuntimeProvider())

    before = await settings.vector_index(db)
    await settings.validate_active(db, "embedding")
    after = await settings.vector_index(db)

    assert (before.status, before.in_use) == ("awaiting_validation", None)
    # With no vectors stored, the model's first answer leaves nothing to rebuild.
    assert (before.reindex_target, before.serves_search) == (None, False)
    assert (after.status, after.serves_search) == ("ready", True)
    assert after.in_use["fingerprint"] == IDENTITY_A.profile(3).fingerprint
    assert after.reindex_target is None


@pytest.mark.parametrize("dimension", [3, 8], ids=["the_models_length", "another_length"])
async def test_stored_vectors_can_be_rebuilt_for_a_model_that_has_not_been_checked(
    db, tmp_path, models, dimension
) -> None:
    config = _config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=dimension)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())

    before = await settings.vector_index(db)
    asked_by_the_view = list(models.requests)
    finished = await settings.run_reindex(db)

    assert (before.status, before.reindex_target, before.in_use) == ("awaiting_validation", "active", None)
    assert asked_by_the_view == []
    assert (finished["state"], finished["adopted"], finished["vector_count"]) == ("active", False, 1)
    assert await _vector_hits(db, config) == ["mem-stored"]


async def test_an_adopted_index_can_be_rebuilt_when_the_models_latest_check_failed(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=3)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())
    await settings.validate_active(db, "embedding")
    models.failure = RuntimeError("the provider is down")
    await settings.validate_active(db, "embedding")

    unreachable = await settings.vector_index(db)
    with pytest.raises(ModelConfigError) as refused:
        await settings.run_reindex(db)
    models.failure = None
    finished = await settings.run_reindex(db)

    assert (unreachable.status, unreachable.reindex_target, unreachable.in_use["adopted"]) == ("ready", "active", True)
    assert refused.value.code == "embedding_not_validated"
    assert (finished["state"], finished["adopted"]) == ("active", False)


# -- Mismatch is refused, not served -------------------------------------------------


async def test_search_answers_from_its_other_channels_while_the_index_belongs_to_another_model(
    db, tmp_path, models, caplog
) -> None:
    config = _config(tmp_path)
    await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
    assert await _vector_hits(db, config) == ["mem-kept"]

    config.llm.embedding_model = "embed-c"
    engine = await runtime.build_search_engine(db, config)
    with caplog.at_level("WARNING", logger="memforge.retrieval.search"):
        by_vector = await engine.search("unrelated words only", top_k=5)
        by_keyword = await engine.search("release pipeline", top_k=5)

    assert by_vector["results"] == []
    assert [hit.memory_id for hit in by_keyword["results"]] == ["mem-kept"]
    assert engine.serves_another_index is True
    assert [record for record in caplog.records if record.levelname == "ERROR"] == []
    # An engine says once that its vector channel is off, however many searches it answers.
    assert [record.getMessage() for record in caplog.records if record.levelname == "WARNING"] == [
        "Vector search is off: the vector index was built with another embedding model. "
        "Searches answer from their other channels until the index is rebuilt."
    ]
    # The model's first answer shows which space it embeds in; after that no request is made.
    assert models.models().count("openai/embed-c") == 1


async def test_health_and_settings_say_a_reindex_is_required_for_another_model(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    settings = ModelSettings(config, DefaultRuntimeProvider())
    await settings.validate_active(db, "embedding")
    assert (await settings.vector_index(db)).status == "ready"

    config.llm.embedding_model = "embed-c"
    not_checked = await settings.vector_index(db)
    await settings.validate_active(db, "embedding")
    checked = await settings.vector_index(db)

    # A model that has not answered yet is judged by its settings, so the rebuild is on offer at once.
    for view in (not_checked, checked):
        assert (view.status, view.reindex_target, view.serves_search) == ("reindex_required", "active", False)
    assert checked.in_use["fingerprint"] == IDENTITY_A.profile(3).fingerprint
    assert (await settings.health(db))["embedding"].vector_index == "reindex_required"


async def test_a_vector_write_for_another_model_is_refused_and_leaves_nothing_behind(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
    in_use = await db.get_active_vector_index()

    config.llm.embedding_model = "embed-c"
    with pytest.raises(EmbeddingIndexMismatch):
        await _write_memory(db, config, "mem-other-space", "A fact embedded by another model")

    assert await db.get_memory("mem-other-space") is None
    assert list(_stored_vectors(config.storage.chroma_path, in_use["collection_name"])) == ["mem-kept"]
    config.llm.embedding_model = "embed-a"
    await _write_memory(db, config, "mem-other-space", "A fact embedded by another model")
    assert sorted(_stored_vectors(config.storage.chroma_path, in_use["collection_name"])) == ["mem-kept", "mem-other-space"]


@pytest.mark.parametrize(
    ("stored_dimension", "error_code"),
    [(None, "reindex_required"), (8, "vector_index_missing")],
    ids=["index_of_another_model", "stored_vectors_of_another_length"],
)
async def test_the_api_answers_a_write_that_waits_for_the_index_with_a_code(
    db, tmp_path, models, stored_dimension, error_code
) -> None:
    import httpx

    from memforge.server.admin_api import create_admin_app

    config = _config(tmp_path)
    if stored_dimension is None:
        await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
        config.llm.embedding_model = "embed-c"
    else:
        _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=stored_dimension)
    app = create_admin_app(db=db, config=config, principal_resolver=lambda _request: "author@example.test")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/memories/create",
            json={
                "content": "A fact that needs a vector.",
                "provenance": "Confirmed by the release owner during the rollout review.",
                "memory_type": "fact",
                "client": "codex",
            },
        )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error"] == error_code
    assert await db.count_searchable_memories() == (1 if stored_dimension is None else 0)


async def test_pending_vector_work_waits_for_the_index_without_using_up_its_attempts(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
    config.llm.embedding_model = "embed-c"
    sync_runtime = await runtime.build_sync_runtime(db, config)
    store = sync_runtime.memory_store
    waited: list[str] = []

    class Task:
        id = "task-1"
        memory_id = "mem-kept"
        attempts = 0

    async def fail(task_id: str, error: str, next_attempt_at: str) -> None:
        waited.append(task_id)

    async def complete(task_id: str) -> None:
        waited.append(task_id)

    async def ready() -> list[Task]:
        return [Task()]

    result = await store._attempt_durable_vector_delivery(
        scope="all",
        owner_kind="Lifecycle",
        load_ready=ready,
        load_remaining=ready,
        complete=complete,
        fail=fail,
    )

    assert result.pending is True
    assert (result.delivered_tasks, result.failed_tasks) == (0, 0)
    assert result.error_types == ("EmbeddingIndexMismatch",)
    assert waited == []


async def test_index_repair_leaves_a_memory_unrepaired_when_the_model_is_not_the_indexes(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
    await db.insert_memory(_memory("mem-no-vector", "A fact that lost its vector"))
    index = await open_memory_vector_index(db, config.storage.chroma_path)
    config.llm.embedding_model = "embed-c"

    result = await MemoryIndexRepairer(
        db=db,
        memory_collection=index.collection,
        embedding=await runtime.build_active_embedding_runtime(db, config, DefaultRuntimeProvider()),
        index_fingerprint=index.fingerprint,
    ).repair()

    assert result.unrepaired_memories == ["mem-no-vector"]
    assert _stored_vectors(config.storage.chroma_path, index.name) == {
        "mem-kept": models.vector("openai/embed-a", "Fact: Deploy through the release pipeline\nEntities: "),
    }


# -- The pairing of a runtime and an index ------------------------------------------


def _runtime(identity: EmbeddingIdentity | None, dimension: int, calls: list[list[str]] | None = None) -> EmbeddingRuntime:
    def embed(texts: list[str]) -> list[list[float]]:
        if calls is not None:
            calls.append(texts)
        return [[float(len(text))] * dimension for text in texts]

    return EmbeddingRuntime(
        EmbeddingSettings(invocation=embedding_invocation(), identity=identity),
        transport=texts_transport(embed),
    )


class _Collection:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        self.vectors.update(zip(ids, embeddings))

    def get(self, **_kwargs):
        return {"ids": []}


async def test_a_runtime_knows_its_fingerprint_once_its_model_has_answered() -> None:
    embedding = _runtime(IDENTITY_A, 3)
    index = SqliteVectorStore(_Collection(), index_fingerprint=IDENTITY_A.profile(3).fingerprint)

    assert embedding.fingerprint is None
    assert embedding.embeds_for(IDENTITY_A.profile(3).fingerprint) is None
    vectors = await embed_documents_for_index(embedding, index, ["one"])

    assert vectors == [[3.0, 3.0, 3.0]]
    assert embedding.fingerprint == IDENTITY_A.profile(3).fingerprint
    assert embedding.embeds_for(IDENTITY_A.profile(3).fingerprint) is True
    assert embedding.embeds_for(IDENTITY_B.profile(3).fingerprint) is False


async def test_vectors_of_another_model_or_another_length_never_reach_an_index() -> None:
    index = SqliteVectorStore(_Collection(), index_fingerprint=IDENTITY_A.profile(3).fingerprint)
    another_model = _runtime(IDENTITY_B, 3)
    another_length = _runtime(IDENTITY_A, 4)
    unidentified = _runtime(None, 3)

    for embedding in (another_model, another_length, unidentified):
        with pytest.raises(EmbeddingIndexMismatch):
            await embed_documents_for_index(embedding, index, ["one"])
        with pytest.raises(EmbeddingIndexMismatch):
            await embed_query_for_index(embedding, index, "one")


async def test_an_index_that_records_no_fingerprint_takes_any_runtime() -> None:
    index = SqliteVectorStore(_Collection())

    assert await index.index_fingerprint() is None
    assert await embed_documents_for_index(_runtime(IDENTITY_B, 5), index, ["one"]) == [[3.0] * 5]
    assert await embed_query_for_index(_runtime(None, 2), index, "one") == [3.0, 3.0]


async def test_a_cached_vector_is_never_served_to_an_index_of_another_profile() -> None:
    calls: list[list[str]] = []
    embedding = _runtime(IDENTITY_A, 3, calls)
    own_index = SqliteVectorStore(_Collection(), index_fingerprint=IDENTITY_A.profile(3).fingerprint)
    other_index = SqliteVectorStore(_Collection(), index_fingerprint=IDENTITY_B.profile(3).fingerprint)

    await embed_query_for_index(embedding, own_index, "cached text")
    with pytest.raises(EmbeddingIndexMismatch):
        await embed_query_for_index(embedding, other_index, "cached text")
    with pytest.raises(EmbeddingIndexMismatch):
        await embed_documents_for_index(embedding, other_index, ["cached text"])

    assert await embed_query_for_index(embedding, own_index, "cached text") == [11.0, 11.0, 11.0]
    assert calls == [["cached text"]]


async def test_two_runtimes_never_share_a_cache() -> None:
    first_calls: list[list[str]] = []
    second_calls: list[list[str]] = []
    first = _runtime(IDENTITY_A, 3, first_calls)
    second = _runtime(IDENTITY_B, 4, second_calls)

    assert await first.embed_query("shared text") == [11.0] * 3
    assert await second.embed_query("shared text") == [11.0] * 4
    assert await first.embed_query("shared text") == [11.0] * 3

    assert (first_calls, second_calls) == ([["shared text"]], [["shared text"]])


async def test_the_search_engine_built_after_a_model_change_has_an_empty_cache(db, tmp_path, models) -> None:
    config = _config(tmp_path)
    await _write_memory(db, config, "mem-kept", "Deploy through the release pipeline")
    first = await runtime.build_search_engine(db, config)
    await first.search("unrelated words only", top_k=5)

    config.llm.embedding_model = "embed-c"
    second = await runtime.build_search_engine(db, config)
    await second.search("unrelated words only", top_k=5)

    assert first._embedding is not second._embedding
    assert models.models()[-2:] == ["openai/embed-a", "openai/embed-c"]
    assert first._embedding.fingerprint != second._embedding.fingerprint


async def test_a_store_with_no_index_in_use_reads_nothing_and_refuses_a_vector(db, tmp_path) -> None:
    async def no_index():
        return None

    store = SqliteVectorStore.of_index_in_use(no_index)

    assert await store.index_fingerprint() is None
    assert await store.open_collection() is None
    assert await store.get_record("mem-none") is None
    assert await store.query_many([[1.0]], [None], [None], 5) == [[]]
    await store.delete(["mem-none"])
    with pytest.raises(NoVectorIndex):
        await store.upsert(["mem-none"], [[1.0]], [{}])


async def test_a_database_opened_read_only_without_an_index_record_has_no_index_in_use(tmp_path) -> None:
    earlier = Database(str(tmp_path / "earlier.db"))
    await earlier.connect()
    await earlier.db.execute("DROP TABLE vector_indexes")
    await earlier.db.commit()
    await earlier.close()

    read_only = Database(str(tmp_path / "earlier.db"))
    await read_only.connect(run_migrations=False)
    try:
        assert await read_only.get_active_vector_index() is None
        assert await read_only.vector_reindex_running() is False
        assert await open_memory_vector_index(read_only, str(tmp_path / "chroma")) is None
    finally:
        await read_only.close()


# -- The first start of a server under this version -----------------------------------


def _embedding_health_once_checked(config: AppConfig, db: Database) -> dict:
    """Start a server, let it check its models, and return what health says of the embedding model."""
    from fastapi.testclient import TestClient

    from memforge.server.admin_api import create_admin_app

    with TestClient(create_admin_app(db=db, config=config)) as client:
        for _ in range(400):
            health = client.get("/api/v1/health").json()["models"]["embedding"]
            if health["state"] == "ready" and health["vector_index"] != "awaiting_validation":
                return health
            time.sleep(0.01)
    raise AssertionError(f"the embedding model was not checked: {health}")


@pytest.mark.parametrize(
    ("stored_dimension", "vector_index", "collection", "adopted"),
    [
        (None, "ready", collection_name_for(IDENTITY_A.profile(3).fingerprint), False),
        (3, "ready", STORED_VECTORS_COLLECTION, True),
        (8, "reindex_required", None, None),
    ],
    ids=["no_vectors", "vectors_of_the_models_length", "vectors_of_another_length"],
)
def test_a_starting_server_settles_the_index_of_an_installation_that_recorded_none(
    tmp_path, models, model_checks_at_server_start, stored_dimension, vector_index, collection, adopted
) -> None:
    config = _config(tmp_path)
    if stored_dimension is not None:
        _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=stored_dimension)

    async def started() -> tuple[dict, dict | None]:
        database = Database(str(tmp_path / "start.db"))
        await database.connect()
        try:
            health = await asyncio.to_thread(_embedding_health_once_checked, config, database)
            return health, await database.get_active_vector_index()
        finally:
            await database.close()

    health, in_use = asyncio.run(started())

    assert health == {
        "configured": True,
        "state": "ready",
        "vector_index": vector_index,
        "reindex_required": vector_index == "reindex_required",
    }
    if collection is None:
        assert in_use is None
        assert _collection_names(config.storage.chroma_path) == [STORED_VECTORS_COLLECTION]
    else:
        assert (in_use["collection_name"], in_use["adopted"]) == (collection, adopted)
        assert in_use["fingerprint"] == IDENTITY_A.profile(3).fingerprint
