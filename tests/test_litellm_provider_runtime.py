"""Generic LiteLLM provider runtime behavior."""

from __future__ import annotations

import logging
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


@pytest.mark.parametrize("variable", ["MEMFORGE_MEMORY_EXTRACTION_MAX_TOKENS", "MEMFORGE_ENRICHMENT_MAX_TOKENS"])
def test_a_task_output_limit_variable_is_warned_about_and_has_no_effect(monkeypatch, caplog, variable):
    from dataclasses import asdict

    from memforge.config import AppConfig

    unset = asdict(AppConfig().llm)
    monkeypatch.setenv(variable, "24576")

    with caplog.at_level(logging.WARNING, logger="memforge.config"):
        config = AppConfig()

    assert asdict(config.llm) == unset
    [warning] = [record for record in caplog.records if record.name == "memforge.config"]
    assert warning.levelno == logging.WARNING
    assert f"{variable} has no effect" in warning.getMessage()


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
                    message=SimpleNamespace(content='{"ranking": [42]}'),
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

    response = await client.rerank_memories("Rank the matching memories")

    assert response.ranking == [42]
    assert captured["model"] == "provider/chat-model"
    assert "api_base" not in captured
    assert "api_key" not in captured


def _embedding_runtime(route: str, *, base_url: str | None = None, api_key: str | None = None):
    from memforge.llm.embedding import EmbeddingRuntime, EmbeddingSettings
    from memforge.llm.providers import ResolvedInvocation

    return EmbeddingRuntime(
        EmbeddingSettings(
            invocation=ResolvedInvocation(route=route, api_base=base_url, api_key=api_key),
        )
    )


@pytest.mark.asyncio
async def test_provider_prefixed_embedding_uses_litellm_without_empty_credentials(monkeypatch):
    captured = {}

    async def fake_aembedding(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ]
        )

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    vectors = await _embedding_runtime("ollama/embedding-model").embed_documents(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["model"] == "ollama/embedding-model"
    assert captured["input"] == ["alpha", "beta"]
    assert captured["max_retries"] == 0
    assert "api_base" not in captured
    assert "api_key" not in captured


@pytest.mark.asyncio
async def test_openai_route_embedding_is_sent_to_the_configured_endpoint(monkeypatch):
    captured = {}

    async def fake_aembedding(**kwargs):
        captured.update(kwargs)
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    vector = await _embedding_runtime(
        "openai/text-embedding-3-small",
        base_url="https://embedding.example.test/v1",
        api_key="key",
    ).embed_query("alpha")

    assert vector == [0.1, 0.2]
    assert captured["model"] == "openai/text-embedding-3-small"
    assert captured["api_base"] == "https://embedding.example.test/v1"
    assert captured["api_key"] == "key"
    assert captured["max_retries"] == 0


@pytest.mark.asyncio
async def test_provider_prefixed_embedding_batches_inputs_at_request_limit(monkeypatch):
    calls: list[list[str]] = []

    async def fake_aembedding(**kwargs):
        inputs = list(kwargs["input"])
        calls.append(inputs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(text.removeprefix("item-"))])
                for index, text in enumerate(inputs)
            ]
        )

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)
    texts = [f"item-{index}" for index in range(70)]

    vectors = await _embedding_runtime("ollama/embedding-model").embed_documents(texts)

    assert [len(batch) for batch in calls] == [32, 32, 6]
    assert vectors == [[float(index)] for index in range(70)]


@pytest.mark.asyncio
async def test_provider_prefixed_embedding_rejects_incomplete_batch_response(monkeypatch):
    async def fake_aembedding(**_kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])])

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    with pytest.raises(
        ValueError,
        match=r"Embedding response count does not match input count: expected 2, got 1",
    ):
        await _embedding_runtime("ollama/embedding-model").embed_documents(["alpha", "beta"])


@pytest.mark.parametrize(
    "data",
    [
        [
            SimpleNamespace(index=0, embedding=[0.1]),
            SimpleNamespace(index=0, embedding=[0.2]),
        ],
        [
            SimpleNamespace(index=0, embedding=[0.1]),
            SimpleNamespace(embedding=[0.2]),
        ],
        [
            SimpleNamespace(index=0, embedding=[0.1]),
            SimpleNamespace(index=2, embedding=[0.2]),
        ],
    ],
    ids=["duplicate", "missing", "out_of_range"],
)
@pytest.mark.asyncio
async def test_provider_prefixed_embedding_rejects_invalid_response_indexes(
    monkeypatch, data
):
    async def fake_aembedding(**_kwargs):
        return SimpleNamespace(data=data)

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    with pytest.raises(ValueError, match=r"Embedding response indices"):
        await _embedding_runtime("ollama/embedding-model").embed_documents(["alpha", "beta"])


@pytest.mark.asyncio
async def test_openai_compatible_embedding_rejects_invalid_response_indexes(monkeypatch):
    async def fake_aembedding(**_kwargs):
        return {
            "data": [
                {"index": 0, "embedding": [0.1]},
                {"index": 0, "embedding": [0.2]},
            ]
        }

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    with pytest.raises(ValueError, match=r"Embedding response indices"):
        await _embedding_runtime(
            "openai/text-embedding-3-small",
            base_url="https://embedding.example.test/v1",
            api_key="key",
        ).embed_documents(["alpha", "beta"])


def test_rerank_model_follows_the_generation_model() -> None:
    from memforge.config import AppConfig
    from memforge.llm.providers import ResolvedInvocation
    from memforge.runtime import ResolvedModels, _retrieval_config_for_models

    config = AppConfig()
    models = ResolvedModels(
        generation=ResolvedInvocation(route="sap/anthropic--claude-4.6-sonnet"),
        embedding=ResolvedInvocation(route="sap/text-embedding-3-small"),
    )

    retrieval = _retrieval_config_for_models(config, models)

    assert retrieval.rerank_model == "sap/anthropic--claude-4.6-sonnet"
    assert config.retrieval.rerank_model != retrieval.rerank_model


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://api.example.test/v1", ("https", "api.example.test", 443)),
        ("HTTPS://API.Example.Test:443/other?x=1", ("https", "api.example.test", 443)),
        ("https://api.example.test./v1", ("https", "api.example.test", 443)),
        ("http://api.example.test", ("http", "api.example.test", 80)),
        ("http://localhost:11434/v1", ("http", "localhost", 11434)),
        ("http://[::1]:8080/v1", ("http", "::1", 8080)),
        ("  https://api.example.test/v1  ", ("https", "api.example.test", 443)),
        ("ftp://api.example.test/v1", None),
        ("api.example.test/v1", None),
        ("https://api.example.test:not-a-port/v1", None),
        ("", None),
        (None, None),
    ],
)
def test_endpoint_origin_normalizes_scheme_host_and_port(url, origin) -> None:
    from memforge.llm.providers import endpoint_origin

    assert endpoint_origin(url) == origin
