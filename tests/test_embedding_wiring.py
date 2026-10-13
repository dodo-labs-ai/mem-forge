"""Every consumer embeds through the Embedding Runtime built from the effective settings."""

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
from memforge.runtime import EffectiveLlmConfig, build_embedding_runtime
from memforge.storage.database import Database
from tests.embedding_fixture import embedding_response

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
    monkeypatch.setattr(runtime, "get_chroma_collection", lambda **_kwargs: recording)
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


def _llm(**embedding: str) -> EffectiveLlmConfig:
    return EffectiveLlmConfig(
        enrichment_model="wiring-generation-model",
        enrichment_base_url="https://generate.wiring.example.test",
        enrichment_api_key="wiring-generation-key",
        request_timeout_s=30.0,
        embedding_model=embedding.get("model", EMBEDDING_MODEL),
        embedding_base_url=embedding.get("base_url", EMBEDDING_BASE_URL),
        embedding_api_key=embedding.get("api_key", EMBEDDING_API_KEY),
    )


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


def test_the_runtime_is_built_from_the_effective_embedding_settings(tmp_path) -> None:
    config = _config(tmp_path)
    config.retrieval.embedding_cache_size = 7

    settings = build_embedding_runtime(_llm(), config).settings

    assert settings.model == EMBEDDING_MODEL
    assert settings.base_url == EMBEDDING_BASE_URL
    assert settings.api_key == EMBEDDING_API_KEY
    assert settings.cache_size == 7


def test_blank_endpoint_settings_are_left_for_the_provider_route_to_resolve(tmp_path) -> None:
    llm = _llm(model="ollama/nomic-embed-text", base_url="", api_key="")

    built = build_embedding_runtime(llm, _config(tmp_path))

    assert built.settings.base_url is None
    assert built.settings.api_key is None
    assert built.configured is True


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
