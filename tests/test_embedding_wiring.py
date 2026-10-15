"""Every consumer embeds through the Embedding Runtime built from the active embedding binding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memforge import runtime
from memforge.config import AppConfig
from memforge.llm.embedding import EmbeddingRuntime
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.llm.providers import ResolvedInvocation
from memforge.runtime import build_embedding_runtime, resolve_models
from memforge.storage.database import Database
from tests.embedding_fixture import embedding_response
from tests.vector_index_fixture import use_collection_as_index

EMBEDDING_MODEL = "wiring-embedding-model"
EMBEDDING_BASE_URL = "https://embed.wiring.example.test/v1"
EMBEDDING_API_KEY = "wiring-embedding-key"


class ProviderRejection(Exception):
    status_code = 401


class RecordingProvider:
    """Stand in for the provider behind LiteLLM and record every embedding request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.failure: Exception | None = None

    @property
    def texts(self) -> list[str]:
        return [text for request in self.requests for text in request["input"]]

    async def __call__(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return embedding_response([_vector(text) for text in request["input"]])


def _vector(text: str) -> list[float]:
    return [float(len(text)), 1.0, 0.5]


class RecordingCollection:
    def __init__(self) -> None:
        self.embeddings: dict[str, list[float]] = {}

    def query(self, **kwargs):
        count = len(kwargs.get("query_embeddings") or ())
        return {"ids": [[] for _ in range(count)], "distances": [[] for _ in range(count)]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            if embeddings:
                self.embeddings[record_id] = list(embeddings[index])

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.embeddings.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        return {"ids": []}


@pytest.fixture
def provider(monkeypatch) -> RecordingProvider:
    recording = RecordingProvider()
    monkeypatch.setattr("litellm.aembedding", recording)
    return recording


@pytest.fixture
def collection(monkeypatch) -> RecordingCollection:
    """The Memory vector collection every default adapter set is bound to."""
    recording = RecordingCollection()
    use_collection_as_index(monkeypatch, recording)
    return recording


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "wiring.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = "wiring-generation-key"
    config.llm.embedding_model = EMBEDDING_MODEL
    config.llm.embedding_base_url = EMBEDDING_BASE_URL
    config.llm.embedding_api_key = EMBEDDING_API_KEY
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        created_at=now,
        updated_at=now,
        status="active",
    )


async def _document(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source="src-wiring",
            source_url="https://docs.wiring.example.test/1",
            title="Wiring",
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


def _assert_sent_to_the_configured_endpoint(request: dict[str, Any]) -> None:
    assert request["model"] == f"openai/{EMBEDDING_MODEL}"
    assert request["api_base"] == EMBEDDING_BASE_URL
    assert request["api_key"] == EMBEDDING_API_KEY


def test_the_runtime_is_built_from_the_resolved_invocation(tmp_path) -> None:
    config = _config(tmp_path)
    config.retrieval.embedding_cache_size = 7
    invocation = ResolvedInvocation(
        route=f"openai/{EMBEDDING_MODEL}",
        api_base=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
    )

    settings = build_embedding_runtime(invocation, config).settings

    assert settings.invocation is invocation
    assert settings.cache_size == 7


def test_a_runtime_built_without_an_invocation_is_not_configured(tmp_path) -> None:
    assert build_embedding_runtime(None, _config(tmp_path)).configured is False


async def test_the_environment_settings_resolve_to_the_configured_endpoint(db, tmp_path) -> None:
    models = await resolve_models(db, _config(tmp_path))

    assert models.embedding == ResolvedInvocation(
        route=f"openai/{EMBEDDING_MODEL}",
        api_base=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
        provider="openai_compatible",
    )


async def test_a_provider_route_without_endpoint_or_key_is_left_for_the_provider_to_resolve(
    db, tmp_path
) -> None:
    config = _config(tmp_path)
    config.llm.embedding_model = "ollama/nomic-embed-text"
    config.llm.embedding_base_url = ""
    config.llm.embedding_api_key = ""

    models = await resolve_models(db, config)

    assert models.embedding == ResolvedInvocation(route="ollama/nomic-embed-text", provider="litellm")
    assert build_embedding_runtime(models.embedding, config).configured is True


async def test_the_sync_runtime_writes_memories_through_the_configured_runtime(
    db, tmp_path, provider, collection
) -> None:
    sync_runtime = await runtime.build_sync_runtime(db, _config(tmp_path))
    await _document(db, "doc-wiring")

    outcome = await sync_runtime.memory_store.deduplicate_and_insert(
        _memory("mem-wiring", "Deploy through the release pipeline"),
        "doc-wiring",
        "manual",
        source_updated_at=None,
    )

    assert outcome == "inserted"
    embedded_text = "Fact: Deploy through the release pipeline\nEntities: "
    assert provider.texts == [embedded_text]
    _assert_sent_to_the_configured_endpoint(provider.requests[0])
    assert collection.embeddings["mem-wiring"] == _vector(embedded_text)


async def test_the_sync_runtime_resolves_entities_with_the_memory_stores_runtime(
    db, tmp_path, collection
) -> None:
    sync_runtime = await runtime.build_sync_runtime(db, _config(tmp_path))

    embedding = sync_runtime.memory_store.embedding

    assert isinstance(embedding, EmbeddingRuntime)
    assert embedding.configured is True
    assert sync_runtime.memory_engine.entity_resolver.embedding is embedding


async def test_the_search_engine_embeds_the_query_through_the_configured_runtime(
    db, tmp_path, provider, collection
) -> None:
    engine = await runtime.build_search_engine(db, _config(tmp_path))

    await engine.search("release pipeline", top_k=5)

    assert provider.texts == ["release pipeline"]
    _assert_sent_to_the_configured_endpoint(provider.requests[0])


async def test_search_answers_from_its_other_channels_when_the_query_cannot_be_embedded(
    db, tmp_path, provider, collection
) -> None:
    await db.insert_memory(_memory("mem-lexical", "Deploy through the release pipeline"))
    provider.failure = ProviderRejection("invalid api key")
    engine = await runtime.build_search_engine(db, _config(tmp_path))

    result = await engine.search("release pipeline", top_k=5)

    assert len(provider.requests) == 1
    assert [hit.memory_id for hit in result["results"]] == ["mem-lexical"]


async def test_a_memory_created_through_the_api_is_embedded_at_the_configured_endpoint(
    db, tmp_path, provider, collection
) -> None:
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        principal_resolver=lambda _request: "author@example.test",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memories/create",
            json={
                "content": "Deploy through the release pipeline.",
                "provenance": "Confirmed by the release owner during the rollout review.",
                "memory_type": "fact",
                "client": "codex",
            },
        )

    assert response.status_code == 200, response.text
    embedded_text = "Fact: Deploy through the release pipeline.\nEntities: "
    assert provider.texts == [embedded_text]
    _assert_sent_to_the_configured_endpoint(provider.requests[0])
    assert collection.embeddings[response.json()["memory_id"]] == _vector(embedded_text)
