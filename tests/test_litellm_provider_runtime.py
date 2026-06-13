"""Generic LiteLLM provider runtime behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_llm_base_url_env_can_be_cleared_for_provider_env_auth(monkeypatch):
    from memforge.config import AppConfig

    monkeypatch.setenv("MEMFORGE_ENRICHMENT_MODEL", "provider/chat-model")
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_BASE_URL", "")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "provider/embedding-model")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", "")

    config = AppConfig()

    assert config.llm.enrichment_model == "provider/chat-model"
    assert config.llm.enrichment_base_url == ""
    assert config.llm.embedding_model == "provider/embedding-model"
    assert config.llm.embedding_base_url == ""


@pytest.mark.asyncio
async def test_structured_client_omits_empty_base_url_and_api_key(monkeypatch):
    from memforge.llm import structured
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig

    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"entity_ids": [42]}'),
                )
            ]
        )

    monkeypatch.setattr(structured.litellm, "acompletion", fake_acompletion)

    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="provider/chat-model",
            base_url=None,
            api_key=None,
            timeout_s=3.0,
        )
    )

    response = await client.detect_query_entities("Find the matching entity")

    assert response.entity_ids == [42]
    assert captured["model"] == "provider/chat-model"
    assert "api_base" not in captured
    assert "api_key" not in captured


def test_provider_prefixed_embedding_uses_litellm_without_empty_credentials(monkeypatch):
    from memforge.retrieval import embeddings

    captured = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ]
        )

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)

    vectors = embeddings.embed_texts(
        ["alpha", "beta"],
        base_url="",
        api_key="",
        model="provider/embedding-model",
    )

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["model"] == "provider/embedding-model"
    assert captured["input"] == ["alpha", "beta"]
    assert "api_base" not in captured
    assert "api_key" not in captured
