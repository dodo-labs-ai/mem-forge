"""The model settings API: descriptors, bindings, validation, discovery and health."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.llm import bindings
from memforge.llm.bindings import resolve_active_binding
from memforge.llm.providers import ModelConfigError, ResolvedInvocation
from memforge.model_settings import CandidateDraft, ModelSettings
from memforge.models import Memory, content_hash
from memforge.runtime import DefaultRuntimeProvider, SourceSyncWorker, resolve_models
from memforge.server import model_settings_api
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database
from tests.stalling_endpoint_fixture import GIVES_UP_AFTER_S, STALL_MODES, stalling_endpoint
from tests.test_model_bindings import _database_with_llm_config

SECRET_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
GENERATION_KEY = "generation-secret-key-1111"
EMBEDDING_KEY = "embedding-secret-key-2222"
ENVIRONMENT_KEY = "environment-secret-key-3333"
OTHER_KEY = "replacement-secret-key-4444"
ALL_KEYS = (GENERATION_KEY, EMBEDDING_KEY, ENVIRONMENT_KEY, OTHER_KEY)
API = "/api/v1/model-settings"
EXPECTED_REPLY = {"probe": {"label": "memforge", "confirmed": True, "items": ["alpha", "beta", "gamma"]}}
# The number the image of every capability check shows here.
IMAGE_DIGITS = "7319"

OPENAI_GENERATION = {"provider": "openai", "model_id": "gpt-5-mini"}
COMPATIBLE_EMBEDDING = {
    "provider": "openai_compatible",
    "endpoint": "https://embed.example.test/v1",
    "model_id": "embed-model",
}


@pytest.fixture(autouse=True)
def secret_key(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", SECRET_KEY)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "settings.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_model = "claude-sonnet-4"
    config.llm.enrichment_base_url = "https://api.anthropic.com"
    config.llm.enrichment_api_key = ""
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_base_url = "https://api.openai.com/v1"
    config.llm.embedding_api_key = ""
    config.llm.decision_model = ""
    # The generation models of these tests are not in LiteLLM's registry, so
    # their token limits come from the environment.
    config.llm.max_input_tokens = 64_000
    config.llm.context_window_tokens = 64_000
    config.llm.max_output_tokens = 8_000
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


class Providers:
    """Stand in for the model providers behind LiteLLM.

    Every generation and embedding request is recorded. A failure set on a kind
    is raised for each of its requests until it is cleared. The generation model
    reads images: it names the number of a check's image.
    """

    def __init__(self, monkeypatch) -> None:
        self.generation_requests: list[dict[str, Any]] = []
        self.embedding_requests: list[dict[str, Any]] = []
        self.generation_failure: Exception | None = None
        self.embedding_failure: Exception | None = None
        self.generation_gate: threading.Event | None = None
        self.dimension = 3
        self.image_reply: dict[str, Any] = {"digits": IMAGE_DIGITS}
        monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", self.complete)
        monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: True)
        monkeypatch.setattr("memforge.llm.capability._new_probe_digits", lambda: IMAGE_DIGITS)
        monkeypatch.setattr("litellm.aembedding", self.embed)

    async def complete(self, **request: Any) -> Any:
        import asyncio

        self.generation_requests.append(request)
        while self.generation_gate is not None and not self.generation_gate.is_set():
            await asyncio.sleep(0.005)
        if self.generation_failure is not None:
            raise self.generation_failure
        reply = self.image_reply if _carries_image(request) else EXPECTED_REPLY
        return SimpleNamespace(
            model="reported-generation-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(reply)),
                    finish_reason="stop",
                )
            ],
        )

    async def embed(self, **request: Any) -> Any:
        self.embedding_requests.append(request)
        if self.embedding_failure is not None:
            raise self.embedding_failure
        return {
            "model": "reported-embedding-model",
            "data": [
                {"index": index, "embedding": [float(len(text))] + [0.5] * (self.dimension - 1)}
                for index, text in enumerate(request["input"])
            ],
        }


def _carries_image(request: dict[str, Any]) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(block.get("type") == "image_url" for block in message["content"])
        for message in request.get("messages", ())
    )


def text_requests(providers: "Providers") -> list[dict[str, Any]]:
    """The generation requests that are not the image of a check."""
    return [request for request in providers.generation_requests if not _carries_image(request)]


class Rejection(Exception):
    status_code = 401


@pytest.fixture
def providers(monkeypatch) -> Providers:
    return Providers(monkeypatch)


def _rejected_key() -> Exception:
    return litellm.AuthenticationError("invalid api key", "openai", "gpt-5-mini")


def _stored_rows(db: Database) -> list[tuple[str, str]]:
    with sqlite3.connect(db.db_path) as raw:
        return raw.execute("SELECT kind, slot FROM model_bindings ORDER BY kind, slot").fetchall()


def _settings(config: AppConfig) -> ModelSettings:
    return ModelSettings(config, DefaultRuntimeProvider())


# -- Descriptors and reading --------------------------------------------------------


def test_the_api_lists_the_provider_descriptors(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    providers = client.get(f"{API}/providers").json()["providers"]

    assert [provider["id"] for provider in providers] == [
        "openai",
        "anthropic",
        "openrouter",
        "ollama",
        "openai_compatible",
        "litellm",
    ]
    by_id = {provider["id"]: provider for provider in providers}
    assert by_id["openai"] == {
        "id": "openai",
        "label": "OpenAI",
        "kinds": ["generation", "embedding"],
        "auth_strategies": ["api_key", "environment"],
        "default_endpoint": "https://api.openai.com/v1",
        "endpoint_required": False,
        "model_is_route": False,
        "lists_models": True,
        "lists_token_limits": False,
        "sets_context_window": False,
        "options": ["organization"],
        "parameters": {"generation": [], "embedding": ["dimensions"]},
        "token_limits": {
            "generation": ["context_window_tokens", "max_output_tokens"],
            "embedding": [],
        },
    }
    assert by_id["anthropic"]["kinds"] == ["generation"]
    assert by_id["anthropic"]["parameters"] == {"generation": []}
    assert by_id["anthropic"]["token_limits"] == {"generation": ["context_window_tokens", "max_output_tokens"]}
    assert by_id["ollama"]["auth_strategies"] == ["none", "api_key"]
    assert by_id["openai_compatible"]["endpoint_required"] is True
    assert by_id["openai_compatible"]["default_endpoint"] is None
    assert by_id["litellm"]["model_is_route"] is True
    assert by_id["litellm"]["lists_models"] is False
    assert by_id["litellm"]["label"] == "Other provider (LiteLLM route)"
    assert {provider["id"] for provider in providers if provider["lists_token_limits"]} == {
        "anthropic",
        "openrouter",
        "openai_compatible",
    }
    assert [provider["id"] for provider in providers if provider["sets_context_window"]] == ["ollama"]


def test_a_fresh_installation_reads_as_writable_with_no_bindings(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = client.get(API).json()

    assert payload["writable"] is True
    assert payload["litellm_version"] == bindings.installed_litellm_version()
    assert payload["generation"] == {"active": None, "candidate": None}
    assert payload["embedding"] == {"active": None, "candidate": None}


def test_environment_values_read_as_the_active_binding_without_the_key(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    client = TestClient(create_admin_app(db=db, config=config))

    response = client.get(API)

    active = response.json()["generation"]["active"]
    assert active == {
        "kind": "generation",
        "provider": "anthropic",
        "endpoint": "https://api.anthropic.com",
        "origin": "https://api.anthropic.com",
        "auth_strategy": "api_key",
        "model_id": "claude-sonnet-4",
        "route": "anthropic/claude-sonnet-4",
        "options": {},
        "parameters": {},
        "token_limits": {},
        "native_schema": None,
        "config_source": "environment",
        "state": "not_validated",
        "validation": None,
        "api_key": {"is_set": True, "source": "environment", "last4": "3333"},
        "saved_at": None,
        "profile": None,
        "waits_for_reindex": False,
    }
    assert ENVIRONMENT_KEY not in response.text
    assert _stored_rows(db) == []


# -- Saving a candidate -------------------------------------------------------------


def test_saving_writes_the_candidate_and_leaves_the_active_binding_in_use(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    client = TestClient(create_admin_app(db=db, config=config))

    response = client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})

    payload = response.json()
    assert response.status_code == 200, response.text
    assert payload["active"]["route"] == "anthropic/claude-sonnet-4"
    candidate = payload["candidate"]
    assert candidate["route"] == "openai/gpt-5-mini"
    assert candidate["endpoint"] == "https://api.openai.com/v1"
    assert candidate["auth_strategy"] == "api_key"
    assert candidate["config_source"] == "settings"
    assert candidate["state"] == "not_validated"
    assert candidate["validation"] is None
    assert candidate["api_key"] == {"is_set": True, "source": "saved", "last4": "1111"}
    assert candidate["saved_at"]
    assert GENERATION_KEY not in response.text
    assert _stored_rows(db) == [("generation", "candidate")]
    assert client.get(API).json()["generation"] == payload


async def test_the_runtime_keeps_the_active_binding_while_a_candidate_waits(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    await _settings(config).save_candidate(
        db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY)
    )

    models = await resolve_models(db, config)

    assert models.generation_route == "anthropic/claude-sonnet-4"
    assert models.generation.api_key == ENVIRONMENT_KEY


def test_a_candidate_that_needs_a_key_cannot_be_saved_without_one(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/generation/candidate", json=OPENAI_GENERATION)

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "api_key_required"
    assert _stored_rows(db) == []


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"provider": "bedrock", "model_id": "model-a"}, "unknown_provider"),
        ({"provider": "anthropic", "model_id": ""}, "model_required"),
        ({"provider": "openai_compatible", "model_id": "model-a"}, "endpoint_required"),
        ({"provider": "openai", "model_id": "model-a", "endpoint": "ftp://host"}, "endpoint_invalid"),
        (
            {"provider": "openai", "model_id": "model-a", "endpoint": "https://user:pw-secret@proxy.example.test/v1"},
            "endpoint_invalid",
        ),
        ({"provider": "openai", "model_id": "model-a", "endpoint": "https://token@proxy.example.test/v1"}, "endpoint_invalid"),
        ({"model_id": "model-a"}, "unknown_provider"),
        ({"provider": "openai"}, "model_required"),
        ({"provider": "litellm", "model_id": "bare-model"}, "model_route_invalid"),
        ({"provider": "anthropic", "model_id": "model-a", "auth_strategy": "none"}, "auth_strategy_not_supported"),
        ({"provider": "openai", "model_id": "model-a", "options": {"project": "p"}}, "option_not_supported"),
        ({"provider": "openai", "model_id": "model-a", "parameters": {"dimensions": 8}}, "parameter_not_supported"),
    ],
)
def test_unusable_settings_are_refused_with_a_stable_code(db, tmp_path, body, code) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/generation/candidate", json={**body, "api_key": GENERATION_KEY})

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == code
    assert response.json()["detail"]["message"]
    assert GENERATION_KEY not in response.text
    assert _stored_rows(db) == []


def test_a_credential_written_into_an_endpoint_is_neither_stored_nor_repeated(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(
        f"{API}/embedding/candidate",
        json={
            "provider": "openai_compatible",
            "model_id": "embed-model",
            "endpoint": "https://user:pw-secret@embed.example.test/v1",
            "auth_strategy": "none",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "endpoint_invalid"
    assert "pw-secret" not in response.text
    assert "pw-secret" not in client.get(API).text
    assert _stored_rows(db) == []


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        ("generation", {"provider": "openai", "model_id": "gpt-5-mini", "endpoint": "https://collector.example.test/v1"}),
        ("embedding", {"provider": "openai", "model_id": "embed-a", "endpoint": "http://api.openai.com/v1"}),
        ("generation", {"provider": "anthropic", "model_id": "claude-x", "endpoint": "https://collector.example.test"}),
        ("generation", {"provider": "openrouter", "model_id": "vendor/model", "endpoint": "https://collector.example.test/api/v1"}),
        ("generation", {"provider": "litellm", "model_id": "openai/gpt-x", "endpoint": "https://collector.example.test/v1"}),
        ("embedding", {"provider": "litellm", "model_id": "voyage/voyage-3", "endpoint": "https://collector.example.test"}),
    ],
    ids=["openai", "openai_other_scheme", "anthropic", "openrouter", "litellm_generation", "litellm_embedding"],
)
def test_credentials_from_the_environment_cannot_be_saved_for_an_endpoint_a_request_names(
    db, tmp_path, providers, kind, body
) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/{kind}/candidate", json={**body, "auth_strategy": "environment"})
    validated = client.post(f"{API}/{kind}/candidate/validate")

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "environment_endpoint_not_allowed"
    assert response.json()["detail"]["message"]
    assert validated.status_code == 409
    assert _stored_rows(db) == []
    assert providers.generation_requests == [] and providers.embedding_requests == []


@pytest.mark.parametrize(
    ("body", "endpoint"),
    [
        ({"provider": "openai", "model_id": "gpt-5-mini"}, "https://api.openai.com/v1"),
        ({"provider": "openai", "model_id": "gpt-5-mini", "endpoint": "https://api.openai.com/v1/"}, "https://api.openai.com/v1"),
        ({"provider": "anthropic", "model_id": "claude-x", "endpoint": ""}, "https://api.anthropic.com"),
        ({"provider": "litellm", "model_id": "litellm_proxy/claude"}, None),
    ],
    ids=["openai_default", "openai_default_written_out", "anthropic_default", "litellm_route"],
)
def test_credentials_from_the_environment_are_accepted_for_the_providers_own_endpoint(
    db, tmp_path, body, endpoint
) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/generation/candidate", json={**body, "auth_strategy": "environment"})

    assert response.status_code == 200, response.text
    candidate = response.json()["candidate"]
    assert (candidate["auth_strategy"], candidate["endpoint"]) == ("environment", endpoint)
    assert candidate["api_key"] == {"is_set": False, "source": None, "last4": None}


async def test_an_api_key_lets_a_route_name_its_own_endpoint(db, tmp_path) -> None:
    view = await _settings(_config(tmp_path)).save_candidate(
        db,
        "generation",
        CandidateDraft(
            provider="litellm",
            model_id="openai/gpt-x",
            endpoint="https://proxy.example.test/v1",
            api_key=GENERATION_KEY,
        ),
    )

    assert (view.candidate.binding.auth_strategy, view.candidate.binding.endpoint) == (
        "api_key",
        "https://proxy.example.test/v1",
    )


async def test_the_environment_variables_may_pair_environment_credentials_with_a_base_url(db, tmp_path) -> None:
    """The operator sets both, so the binding the variables describe keeps its base URL."""
    config = _config(tmp_path)
    config.llm.enrichment_model = "openai/gpt-x"
    config.llm.enrichment_base_url = "https://proxy.example.test/v1"

    models = await resolve_models(db, config)

    assert (models.generation.route, models.generation.api_base, models.generation.api_key) == (
        "openai/gpt-x",
        "https://proxy.example.test/v1",
        None,
    )


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("put", "generation/candidate", {"provider": "openai", "model_id": 7, "api_key": GENERATION_KEY}),
        ("put", "generation/candidate", {"provider": "openai", "model_id": "m", "api_key": [GENERATION_KEY]}),
        ("put", "generation/candidate", {"provider": "openai", "options": GENERATION_KEY, "api_key": GENERATION_KEY}),
        ("put", "reranking/candidate", {"provider": "openai", "model_id": "m", "api_key": GENERATION_KEY}),
        ("post", "generation/discover-models", {"provider": ["openai"], "api_key": GENERATION_KEY}),
        ("post", "generation/discover-models", {"provider": "openai", "api_key": {"value": GENERATION_KEY}}),
    ],
    ids=[
        "model_not_text",
        "key_not_text",
        "options_not_an_object",
        "unknown_kind",
        "discovery_provider_not_text",
        "discovery_key_not_text",
    ],
)
def test_a_malformed_request_is_refused_without_repeating_what_it_carried(db, tmp_path, method, path, body) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.request(method, f"{API}/{path}", json=body)

    assert response.status_code == 422
    assert GENERATION_KEY not in response.text
    errors = response.json()["detail"]
    assert errors and all(set(error) == {"type", "loc", "msg"} for error in errors)
    assert _stored_rows(db) == []


def test_a_request_without_a_provider_or_a_model_is_refused_by_code_and_not_echoed(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    without_model = client.put(
        f"{API}/generation/candidate", json={"provider": "openai", "api_key": GENERATION_KEY}
    )
    without_provider = client.post(f"{API}/generation/discover-models", json={"api_key": GENERATION_KEY})

    assert (without_model.status_code, without_model.json()["detail"]["error"]) == (422, "model_required")
    assert (without_provider.status_code, without_provider.json()["detail"]["error"]) == (422, "unknown_provider")
    assert GENERATION_KEY not in without_model.text + without_provider.text


def test_a_provider_that_does_not_serve_the_kind_is_refused(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(
        f"{API}/embedding/candidate",
        json={"provider": "anthropic", "model_id": "claude-sonnet-4", "api_key": EMBEDDING_KEY},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "kind_not_supported"


def test_an_unknown_kind_is_not_a_route(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    assert client.put(f"{API}/reranking/candidate", json=OPENAI_GENERATION).status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    ["https://embed.example.test/v1", "https://EMBED.example.test:443/other/path/"],
    ids=["same_url", "same_origin_other_spelling"],
)
def test_a_candidate_saved_again_for_the_same_origin_keeps_its_key(db, tmp_path, endpoint) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    response = client.put(
        f"{API}/embedding/candidate",
        json={**COMPATIBLE_EMBEDDING, "endpoint": endpoint, "model_id": "another-model"},
    )

    candidate = response.json()["candidate"]
    assert response.status_code == 200, response.text
    assert candidate["model_id"] == "another-model"
    assert candidate["api_key"] == {"is_set": True, "source": "saved", "last4": "2222"}


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://other.example.test/v1",
        "http://embed.example.test/v1",
        "https://embed.example.test:8443/v1",
    ],
    ids=["other_host", "other_scheme", "other_port"],
)
def test_a_candidate_for_another_origin_has_to_bring_the_key_again(db, tmp_path, endpoint) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    refused = client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "endpoint": endpoint})
    kept = client.get(API).json()["embedding"]["candidate"]
    accepted = client.put(
        f"{API}/embedding/candidate",
        json={**COMPATIBLE_EMBEDDING, "endpoint": endpoint, "api_key": OTHER_KEY},
    )

    assert refused.status_code == 422
    assert refused.json()["detail"]["error"] == "api_key_origin_changed"
    assert "https://embed.example.test" in refused.json()["detail"]["message"]
    assert EMBEDDING_KEY not in refused.text
    assert kept["endpoint"] == "https://embed.example.test/v1"
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["candidate"]["api_key"]["last4"] == "4444"


async def test_a_candidate_keeps_the_active_bindings_key_only_on_its_origin(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(db, "embedding", CandidateDraft(**COMPATIBLE_EMBEDDING, api_key=EMBEDDING_KEY))
    await settings.validate_candidate(db, "embedding")

    same_origin = await settings.save_candidate(
        db, "embedding", CandidateDraft(**{**COMPATIBLE_EMBEDDING, "model_id": "larger-model"})
    )
    await settings.discard_candidate(db, "embedding")
    with pytest.raises(ModelConfigError) as refused:
        await settings.save_candidate(
            db,
            "embedding",
            CandidateDraft(**{**COMPATIBLE_EMBEDDING, "endpoint": "https://collector.example.test/v1"}),
        )

    assert same_origin.candidate.key.last4 == "2222"
    assert same_origin.candidate.binding.secret_ref == same_origin.active.binding.secret_ref
    assert refused.value.code == "api_key_origin_changed"


async def test_a_candidate_may_keep_the_environment_key_of_the_active_binding_on_its_origin(
    db, tmp_path
) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    settings = _settings(config)

    view = await settings.save_candidate(
        db, "generation", CandidateDraft(provider="anthropic", model_id="claude-opus-4")
    )
    with pytest.raises(ModelConfigError) as refused:
        await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION))

    assert view.candidate.binding.secret_ref == "env:MEMFORGE_ENRICHMENT_API_KEY"
    assert view.candidate.key.source == "environment"
    assert refused.value.code == "api_key_origin_changed"


def test_a_candidate_can_be_discarded(db, tmp_path) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})

    response = client.delete(f"{API}/generation/candidate")

    assert response.status_code == 200
    assert response.json() == {"active": None, "candidate": None}
    assert _stored_rows(db) == []
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM model_secrets").fetchone() == (0,)


# -- Validation and promotion -------------------------------------------------------


def test_a_candidate_becomes_active_only_after_its_check_passes(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})
    providers.generation_failure = _rejected_key()

    failed = client.post(f"{API}/generation/candidate/validate")
    after_failure = client.get(API).json()["generation"]
    providers.generation_failure = None
    passed = client.post(f"{API}/generation/candidate/validate")

    assert failed.status_code == 200, failed.text
    assert failed.json()["passed"] is False
    assert failed.json()["promoted"] is False
    assert failed.json()["error_code"] == "authentication_failed"
    assert after_failure["active"] is None
    assert after_failure["candidate"]["state"] == "not_validated"
    assert after_failure["candidate"]["validation"]["passed"] is False
    assert after_failure["candidate"]["validation"]["error_code"] == "authentication_failed"
    assert after_failure["candidate"]["validation"]["current"] is True

    payload = passed.json()
    assert passed.status_code == 200, passed.text
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["error_code"] is None
    assert payload["settings"]["candidate"] is None
    active = payload["settings"]["active"]
    assert active["route"] == "openai/gpt-5-mini"
    assert active["state"] == "ready"
    assert active["api_key"] == {"is_set": True, "source": "saved", "last4": "1111"}
    assert active["validation"]["passed"] is True
    assert active["validation"]["current"] is True
    assert active["validation"]["facts"] == {
        "latency_ms": active["validation"]["facts"]["latency_ms"],
        "token_limits_known": active["validation"]["facts"]["token_limits_known"],
        "structured_output_mode": "native_schema",
        "schema_fallback": False,
        "provider_model": "reported-generation-model",
        "image_input": True,
    }
    assert _stored_rows(db) == [("generation", "active")]
    for response in (failed, passed):
        assert GENERATION_KEY not in response.text


def test_the_check_is_made_with_the_candidates_route_endpoint_and_key(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})

    client.post(f"{API}/generation/candidate/validate")

    request = providers.generation_requests[0]
    assert request["model"] == "openai/gpt-5-mini"
    assert request["api_base"] == "https://api.openai.com/v1"
    assert request["api_key"] == GENERATION_KEY
    assert request["num_retries"] == 0


async def test_the_runtime_uses_a_promoted_binding(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    before = await resolve_models(db, config)

    outcome = await settings.validate_candidate(db, "generation")
    after = await resolve_models(db, config)

    assert before.generation is None
    assert outcome.promoted is True
    assert after.generation == ResolvedInvocation(
        route="openai/gpt-5-mini",
        api_base="https://api.openai.com/v1",
        api_key=GENERATION_KEY,
        provider="openai",
        image_input=True,
    )


def test_a_validated_embedding_candidate_becomes_active_at_once(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(
        f"{API}/embedding/candidate",
        json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY, "parameters": {"dimensions": 3}},
    )

    response = client.post(f"{API}/embedding/candidate/validate")

    payload = response.json()
    assert payload["promoted"] is True
    active = payload["settings"]["active"]
    assert active["state"] == "ready"
    assert active["parameters"] == {"dimensions": 3}
    assert active["validation"]["facts"] == {
        "dimension": 3,
        "provider_model": "reported-embedding-model",
        "latency_ms": active["validation"]["facts"]["latency_ms"],
    }
    assert len(providers.embedding_requests) == 2
    assert providers.embedding_requests[0]["dimensions"] == 3
    assert providers.embedding_requests[0]["api_key"] == EMBEDDING_KEY
    assert EMBEDDING_KEY not in response.text


def test_an_embedding_candidate_whose_vectors_are_not_the_requested_size_stays_a_candidate(
    db, tmp_path, providers
) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(
        f"{API}/embedding/candidate",
        json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY, "parameters": {"dimensions": 8}},
    )

    response = client.post(f"{API}/embedding/candidate/validate")

    assert response.json()["error_code"] == "embedding_dimension_mismatch"
    assert response.json()["settings"]["active"] is None
    assert response.json()["settings"]["candidate"]["validation"]["facts"]["dimension"] == 3


def test_a_keyless_candidate_is_checked_without_a_credential_from_the_environment(
    db, tmp_path, providers, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    saved = client.put(
        f"{API}/embedding/candidate",
        json={**COMPATIBLE_EMBEDDING, "endpoint": "http://localhost:11434/v1", "auth_strategy": "none"},
    )

    response = client.post(f"{API}/embedding/candidate/validate")

    assert saved.json()["candidate"]["api_key"] == {"is_set": False, "source": None, "last4": None}
    assert response.json()["promoted"] is True
    assert {request["api_key"] for request in providers.embedding_requests} == {"no-api-key"}


def test_validating_without_a_candidate_is_refused(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.post(f"{API}/generation/candidate/validate")

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "candidate_not_found"
    assert providers.generation_requests == []


def test_promotion_makes_the_server_build_its_search_engine_again(db, tmp_path, providers) -> None:
    app = create_admin_app(db=db, config=_config(tmp_path))
    client = TestClient(app)
    app.state.memory_search_engine = object()
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    client.post(f"{API}/embedding/candidate/validate")

    assert app.state.memory_search_engine is None


async def test_promotion_makes_the_sync_worker_embed_with_the_binding_now_active(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    app = create_admin_app(db=db, config=config)
    settings: ModelSettings = app.state.model_settings
    await settings.save_candidate(db, "embedding", CandidateDraft(**COMPATIBLE_EMBEDDING, api_key=EMBEDDING_KEY))
    await settings.validate_candidate(db, "embedding")
    worker = SourceSyncWorker(db, config)
    app.state.sync_worker = worker
    worker._vector_runtime = await worker._build_runtime()
    worker._relation_runtime = worker._vector_runtime
    before = worker._vector_runtime.memory_store.embedding.settings.invocation

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        await client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": OTHER_KEY})
        promoted = await client.post(f"{API}/embedding/candidate/validate")

    assert promoted.json()["promoted"] is True
    assert (worker._vector_runtime, worker._relation_runtime) == (None, None)
    after = (await worker._build_runtime()).memory_store.embedding.settings.invocation
    assert (before.route, before.api_base, before.api_key) == (
        "openai/embed-model",
        "https://embed.example.test/v1",
        EMBEDDING_KEY,
    )
    assert (after.route, after.api_base, after.api_key) == (
        "openai/embed-model",
        "https://embed.example.test/v1",
        OTHER_KEY,
    )


async def test_a_first_embedding_model_waits_for_a_rebuild_that_embeds_the_memories_kept_without_vectors(
    db, tmp_path, providers
) -> None:
    config = _config(tmp_path)
    config.llm.embedding_model = ""
    content = "Deploy through the release pipeline."
    await db.insert_memory(Memory(id="mem-kept", memory_type="fact", content=content, content_hash=content_hash(content)))
    settings = _settings(config)
    await settings.save_candidate(db, "embedding", CandidateDraft(**COMPATIBLE_EMBEDDING, api_key=EMBEDDING_KEY))

    outcome = await settings.validate_candidate(db, "embedding")
    waiting = await settings.vector_index(db)
    finished = await settings.run_reindex(db)
    view = await settings.view(db, "embedding")

    assert (outcome.promoted, outcome.reindex_required) == (False, True)
    assert (waiting.status, waiting.reindex_target) == ("not_configured", "candidate")
    assert (finished["state"], finished["vector_count"]) == ("active", 1)
    assert (view.active is not None, view.candidate) == (True, None)


async def test_a_failed_validation_leaves_the_sync_workers_runtimes_in_place(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    app = create_admin_app(db=db, config=config)
    worker = SourceSyncWorker(db, config)
    app.state.sync_worker = worker
    worker._vector_runtime = kept = await worker._build_runtime()
    providers.embedding_failure = Rejection("invalid api key")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        await client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})
        refused = await client.post(f"{API}/embedding/candidate/validate")

    assert refused.json()["promoted"] is False
    assert worker._vector_runtime is kept


async def test_a_runtime_built_while_the_models_change_is_built_again(db, tmp_path) -> None:
    config = _config(tmp_path)
    built: list[object] = []

    class ChangingProvider(DefaultRuntimeProvider):
        async def build_sync_runtime(self, *args: Any, **kwargs: Any) -> Any:
            runtime = await super().build_sync_runtime(*args, **kwargs)
            built.append(runtime)
            if len(built) == 1:
                worker.models_changed()
            return runtime

    worker = SourceSyncWorker(db, config, runtime_provider=ChangingProvider())

    runtime = await worker._build_runtime()

    assert len(built) == 2
    assert runtime is built[1]


async def test_simultaneous_checks_of_the_active_binding_make_one_model_request(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    settings = _settings(config)
    providers.generation_gate = threading.Event()

    checks = [asyncio.create_task(settings.validate_active(db, "generation")) for _ in range(4)]
    while not providers.generation_requests:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)
    providers.generation_gate.set()
    outcomes = await asyncio.gather(*checks)

    assert len(text_requests(providers)) == 1
    assert {outcome.state for outcome in outcomes} == {"ready"}
    assert len({outcome.validation.checked_at for outcome in outcomes}) == 1
    assert (await settings.health(db))["generation"].state == "ready"


async def test_a_check_requested_after_one_finished_is_made_again(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    settings = _settings(config)

    await settings.validate_active(db, "generation")
    await settings.validate_active(db, "generation")

    assert len(text_requests(providers)) == 2


async def test_a_check_that_started_first_is_the_one_whose_outcome_is_stored(db, tmp_path, providers) -> None:
    """A later request does not start a second check whose result an earlier one could overwrite."""
    config = _config(tmp_path)
    settings = _settings(config)
    await _ready_generation(db, settings)
    providers.generation_requests.clear()
    providers.generation_gate = threading.Event()
    providers.generation_failure = _rejected_key()

    first = asyncio.create_task(settings.validate_active(db, "generation"))
    while not providers.generation_requests:
        await asyncio.sleep(0.005)
    second = asyncio.create_task(settings.validate_active(db, "generation"))
    await asyncio.sleep(0.05)
    providers.generation_gate.set()
    outcomes = await asyncio.gather(first, second)

    view = await settings.view(db, "generation")
    assert len(text_requests(providers)) == 1
    assert [outcome.state for outcome in outcomes] == ["needs_revalidation", "needs_revalidation"]
    assert view.active.binding.state == "needs_revalidation"
    assert view.active.binding.validation.error_code == "authentication_failed"


async def test_simultaneous_validations_of_a_candidate_promote_it_once(db, tmp_path, providers) -> None:
    settings = _settings(_config(tmp_path))
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    providers.generation_gate = threading.Event()

    checks = [asyncio.create_task(settings.validate_candidate(db, "generation")) for _ in range(3)]
    while not providers.generation_requests:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)
    providers.generation_gate.set()
    outcomes = await asyncio.gather(*checks)

    view = await settings.view(db, "generation")
    assert len(text_requests(providers)) == 1
    assert [outcome.promoted for outcome in outcomes] == [True, True, True]
    assert (view.active.binding.state, view.candidate) == ("ready", None)


async def test_a_binding_that_replaced_the_checked_one_gets_a_check_of_its_own(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await _ready_generation(db, settings)
    providers.generation_requests.clear()
    providers.generation_gate = threading.Event()

    first = asyncio.create_task(settings.validate_active(db, "generation"))
    while not providers.generation_requests:
        await asyncio.sleep(0.005)
    await settings.save_candidate(db, "generation", CandidateDraft(provider="openai", model_id="gpt-5"))
    promotion = asyncio.create_task(settings.validate_candidate(db, "generation"))
    while len(text_requests(providers)) < 2:
        await asyncio.sleep(0.005)
    providers.generation_gate.set()
    await asyncio.gather(first, promotion)
    providers.generation_requests.clear()
    await settings.validate_active(db, "generation")

    view = await settings.view(db, "generation")
    assert [request["model"] for request in text_requests(providers)] == ["openai/gpt-5"]
    assert (view.active.binding.model_id, view.active.binding.state) == ("gpt-5", "ready")


async def test_a_candidate_saved_again_while_it_is_checked_is_not_promoted(
    db, tmp_path, providers, monkeypatch
) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    answer = providers.complete

    async def save_again_then_answer(**request: Any) -> Any:
        await settings.save_candidate(db, "generation", CandidateDraft(provider="openai", model_id="gpt-5"))
        return await answer(**request)

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", save_again_then_answer)
    with pytest.raises(ModelConfigError) as refused:
        await settings.validate_candidate(db, "generation")

    view = await settings.view(db, "generation")
    assert refused.value.code == "candidate_changed"
    assert view.active is None
    assert view.candidate.binding.model_id == "gpt-5"
    assert view.candidate.binding.validation is None


# -- States of the active binding ---------------------------------------------------


async def _ready_generation(db: Database, settings: ModelSettings) -> None:
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    assert (await settings.validate_candidate(db, "generation")).promoted is True


async def test_a_model_its_check_finds_reading_text_alone_gets_no_image_at_runtime(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    providers.image_reply = {"digits": ""}
    await _ready_generation(db, _settings(config))

    stored = bindings.binding_from_record(await db.get_model_binding("generation", "active"))
    models = await resolve_models(db, config)

    assert stored.validation.facts["image_input"] is False
    assert models.generation.image_input is False
    assert DefaultRuntimeProvider().build_structured_llm_client(models, max_concurrent=1).reads_images is False


@pytest.mark.parametrize("config_source", ["environment", "deployment"])
async def test_a_model_the_environment_names_gets_no_image_once_its_check_finds_it_reads_text_alone(
    db, tmp_path, providers, config_source
) -> None:
    if config_source == "deployment":
        config = _deployment_config(tmp_path)
    else:
        config = _config(tmp_path)
        config.llm.enrichment_api_key = GENERATION_KEY
    providers.image_reply = {"digits": ""}

    await _settings(config).validate_unready_active_bindings(db)
    models = await resolve_models(db, config)

    assert await db.get_model_binding("generation", "active") is None
    assert models.generation.image_input is False


async def _ready_generation_never_asked_about_images(
    db: Database, settings: ModelSettings, providers: Providers
) -> bindings.ValidationResult:
    await _ready_generation(db, settings)
    record = await db.get_model_binding("generation", "active")
    checked = bindings.binding_from_record(record).validation
    unasked = {key: value for key, value in checked.facts.items() if key != "image_input"}
    record["validation"] = {**record["validation"], "facts": unasked}
    await db.put_model_binding("generation", "active", record)
    providers.generation_requests.clear()
    return replace(checked, facts=unasked)


async def test_a_starting_server_asks_a_ready_model_once_whether_it_reads_images(db, tmp_path, providers) -> None:
    settings = _settings(_config(tmp_path))
    unasked = await _ready_generation_never_asked_about_images(db, settings, providers)

    await settings.validate_unready_active_bindings(db)
    asked = bindings.binding_from_record(await db.get_model_binding("generation", "active"))
    await settings.validate_unready_active_bindings(db)

    assert asked.state == "ready"
    assert asked.validation == replace(unasked, facts={**unasked.facts, "image_input": True})
    assert len(providers.generation_requests) == 1
    assert text_requests(providers) == []


async def test_a_check_requested_while_a_starting_server_asks_about_images_is_made(
    db, tmp_path, providers
) -> None:
    settings = _settings(_config(tmp_path))
    await _ready_generation_never_asked_about_images(db, settings, providers)
    providers.generation_gate = threading.Event()

    asking = asyncio.create_task(settings.validate_unready_active_bindings(db))
    while not providers.generation_requests:
        await asyncio.sleep(0.005)
    check = asyncio.create_task(settings.validate_active(db, "generation"))
    await asyncio.sleep(0.05)
    providers.generation_gate.set()
    await asking
    outcome = await check

    assert len(text_requests(providers)) == 1
    assert outcome.validation.facts["image_input"] is True
    assert outcome.state == "ready"


async def test_a_provider_down_at_start_leaves_a_ready_model_ready(db, tmp_path, providers) -> None:
    settings = _settings(_config(tmp_path))
    unasked = await _ready_generation_never_asked_about_images(db, settings, providers)
    providers.generation_failure = litellm.ServiceUnavailableError("no upstream available", "openai", "gpt-5-mini")

    await settings.validate_unready_active_bindings(db)
    asked = bindings.binding_from_record(await db.get_model_binding("generation", "active"))

    assert asked.state == "ready"
    assert asked.validation == replace(unasked, facts={**unasked.facts, "image_input": None})


async def test_a_failed_check_moves_a_ready_binding_to_needs_revalidation_and_it_stays_in_use(
    db, tmp_path, providers
) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await _ready_generation(db, settings)
    providers.generation_failure = _rejected_key()

    outcome = await settings.validate_active(db, "generation")
    view = await settings.view(db, "generation")
    models = await resolve_models(db, config)

    assert outcome.promoted is False
    assert outcome.state == "needs_revalidation"
    assert view.active.binding.state == "needs_revalidation"
    assert view.active.binding.validation.error_code == "authentication_failed"
    assert models.generation_route == "openai/gpt-5-mini"

    providers.generation_failure = None
    assert (await settings.validate_active(db, "generation")).state == "ready"


def test_the_active_binding_can_be_checked_again_through_the_api(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    client = TestClient(create_admin_app(db=db, config=config))

    response = client.post(f"{API}/generation/active/validate")
    unconfigured = client.post(f"{API}/embedding/active/validate")

    assert response.status_code == 200, response.text
    assert response.json()["passed"] is True
    assert response.json()["promoted"] is False
    assert response.json()["settings"]["active"]["state"] == "ready"
    assert providers.generation_requests[0]["api_key"] == ENVIRONMENT_KEY
    assert ENVIRONMENT_KEY not in response.text
    assert unconfigured.status_code == 409
    assert unconfigured.json()["detail"]["error"] == "model_not_configured"
    assert _stored_rows(db) == []


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint": "https://embed.example.test/v2"},
        {"model_id": "larger-model"},
        {"parameters": {"dimensions": 3}},
        {"auth_strategy": "none"},
    ],
    ids=["endpoint", "model", "dimensions", "auth_strategy"],
)
async def test_changing_what_a_check_vouched_for_leaves_a_ready_candidate_to_be_validated_again(
    db, tmp_path, change
) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    view = await settings.save_candidate(
        db, "embedding", CandidateDraft(**COMPATIBLE_EMBEDDING, api_key=EMBEDDING_KEY)
    )
    binding = view.candidate.binding
    passed = bindings.validation_record(
        bindings.ValidationResult(checked_at="2026-10-14T00:00:00+00:00", digest=binding.digest, facts={"dimension": 3})
    )
    await db.record_model_binding_validation(
        "embedding", "candidate", updated_at=binding.saved_at, state="ready", validation=passed
    )

    unchanged = await settings.save_candidate(db, "embedding", CandidateDraft(**COMPATIBLE_EMBEDDING))
    changed = await settings.save_candidate(
        db, "embedding", CandidateDraft(**{**COMPATIBLE_EMBEDDING, **change})
    )

    assert unchanged.candidate.binding.state == "ready"
    assert changed.candidate.binding.state == "needs_revalidation"
    assert changed.candidate.binding.validation.digest != changed.candidate.binding.digest


async def test_a_failed_check_is_forgotten_once_the_settings_it_failed_with_change(
    db, tmp_path, providers
) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    providers.generation_failure = _rejected_key()
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    await settings.validate_candidate(db, "generation")

    same = await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION))
    new_key = await settings.save_candidate(
        db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=OTHER_KEY)
    )

    assert same.candidate.binding.validation.error_code == "authentication_failed"
    assert new_key.candidate.binding.validation is None
    assert new_key.candidate.binding.state == "not_validated"


# -- At server start ----------------------------------------------------------------


async def _translated_generation(db: Database) -> None:
    """Store an active binding as the settings of an earlier version are carried over."""
    binding = bindings.build_binding(
        kind="generation",
        provider="anthropic",
        model_id="claude-sonnet-4",
        secret_ref="env:MEMFORGE_ENRICHMENT_API_KEY",
        config_source="settings",
    )
    await db.put_model_binding("generation", "active", bindings.binding_record(binding))


def _wait_for_state(client: TestClient, kind: str, state: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        health = client.get("/api/v1/health").json()["models"]
        if health[kind]["state"] == state:
            return health
        time.sleep(0.01)
    raise AssertionError(f"the {kind} model did not become {state}: {health}")


async def test_a_starting_server_checks_carried_over_bindings_without_waiting_for_the_check(
    db, tmp_path, providers, model_checks_at_server_start
) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    await _translated_generation(db)
    providers.generation_gate = threading.Event()

    with TestClient(create_admin_app(db=db, config=config)) as client:
        while_checking = client.get("/api/v1/health").json()["models"]
        providers.generation_gate.set()
        checked = _wait_for_state(client, "generation", "ready")

    assert while_checking["generation"] == {"configured": True, "state": "not_validated"}
    assert checked["generation"] == {"configured": True, "state": "ready"}
    assert len(text_requests(providers)) == 1
    stored = bindings.binding_from_record(await db.get_model_binding("generation", "active"))
    assert stored.state == "ready"
    assert stored.validation.facts["structured_output_mode"] == "native_schema"


async def test_a_carried_over_binding_that_fails_its_check_stays_in_use(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    await _translated_generation(db)
    providers.generation_failure = _rejected_key()

    settings = _settings(config)
    await settings.validate_unready_active_bindings(db)

    view = await settings.view(db, "generation")
    assert view.active.binding.state == "not_validated"
    assert view.active.binding.validation.error_code == "authentication_failed"
    assert (await resolve_models(db, config)).generation_route == "anthropic/claude-sonnet-4"


async def test_a_starting_server_checks_again_a_binding_validated_under_another_litellm(
    db, tmp_path, providers, monkeypatch
) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await _ready_generation(db, settings)
    providers.generation_requests.clear()

    await settings.validate_unready_active_bindings(db)
    assert providers.generation_requests == []

    monkeypatch.setattr(bindings, "installed_litellm_version", lambda: "9.9.9")
    assert (await settings.health(db))["generation"].state == "needs_revalidation"
    await settings.validate_unready_active_bindings(db)

    assert len(text_requests(providers)) == 1
    assert (await settings.health(db))["generation"].state == "ready"


async def test_a_failing_check_of_one_kind_does_not_stop_the_check_of_the_other(
    db, tmp_path, providers
) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    config.llm.embedding_api_key = ENVIRONMENT_KEY
    settings = _settings(config)
    validate_active = settings.validate_active

    async def validate(database, kind):
        if kind == "generation":
            raise RuntimeError("the check could not be made")
        return await validate_active(database, kind)

    settings.validate_active = validate
    await settings.validate_unready_active_bindings(db)

    health = await settings.health(db)
    assert health["generation"].state == "not_validated"
    assert health["embedding"].state == "ready"


async def test_a_starting_server_translates_earlier_settings_with_the_configuration_it_was_given(
    tmp_path, providers
) -> None:
    """A host hands its own configuration to the app; that one completes the stored settings."""
    config = _config(tmp_path)
    config.llm.enrichment_model = "claude-from-the-configuration"
    config.llm.enrichment_api_key = ENVIRONMENT_KEY
    config.llm.embedding_api_key = ENVIRONMENT_KEY
    await _database_with_llm_config(
        Path(config.storage.db_path),
        {
            "enrichment_model": "claude-opus-ui",
            "enrichment_base_url": "http://proxy.internal:6655/anthropic",
            "embedding_model": "text-embedding-3-large",
            "embedding_api_key": EMBEDDING_KEY,
        },
    )

    with TestClient(create_admin_app(config=config)) as client:
        payload = client.get(API).json()

    generation, embedding = payload["generation"]["active"], payload["embedding"]["active"]
    assert (generation["route"], generation["endpoint"], generation["config_source"], generation["state"]) == (
        "anthropic/claude-opus-ui",
        "http://proxy.internal:6655/anthropic",
        "settings",
        "not_validated",
    )
    assert generation["api_key"] == {"is_set": True, "source": "environment", "last4": "3333"}
    assert (embedding["route"], embedding["endpoint"]) == (
        "openai/text-embedding-3-large",
        "https://api.openai.com/v1",
    )
    assert embedding["api_key"] == {"is_set": True, "source": "saved", "last4": "2222"}
    database_bytes = b"".join(
        file.read_bytes() for file in Path(config.storage.db_path).parent.glob("memforge.db*")
    )
    assert EMBEDDING_KEY.encode() not in database_bytes


def test_a_server_with_nothing_configured_makes_no_model_request_at_start(
    db, tmp_path, providers, model_checks_at_server_start
) -> None:
    with TestClient(create_admin_app(db=db, config=_config(tmp_path))) as client:
        health = client.get("/api/v1/health").json()["models"]

    assert health == {
        "generation": {"configured": False, "state": None},
        "embedding": {
            "configured": False,
            "state": None,
            "vector_index": "not_configured",
            "reindex_required": False,
        },
    }
    assert providers.generation_requests == []
    assert providers.embedding_requests == []


# -- A deployment that manages its model settings -----------------------------------


def _deployment_config(tmp_path: Path) -> AppConfig:
    config = _config(tmp_path)
    config.server.llm_config_writable = False
    config.llm.enrichment_model = "litellm_proxy/claude-sonnet"
    config.llm.enrichment_base_url = ""
    config.llm.embedding_model = "litellm_proxy/text-embedding-3-small"
    config.llm.embedding_base_url = ""
    return config


def test_a_deployment_managed_installation_reads_its_bindings_and_refuses_every_write(
    db, tmp_path, providers
) -> None:
    client = TestClient(create_admin_app(db=db, config=_deployment_config(tmp_path)))

    read = client.get(API)
    writes = [
        client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY}),
        client.delete(f"{API}/generation/candidate"),
        client.post(f"{API}/generation/candidate/validate"),
    ]

    payload = read.json()
    assert payload["writable"] is False
    assert payload["generation"]["candidate"] is None
    active = payload["generation"]["active"]
    assert (active["provider"], active["route"], active["auth_strategy"], active["config_source"]) == (
        "litellm",
        "litellm_proxy/claude-sonnet",
        "environment",
        "deployment",
    )
    assert active["api_key"] == {"is_set": False, "source": None, "last4": None}
    assert payload["embedding"]["active"]["route"] == "litellm_proxy/text-embedding-3-small"
    for response in writes:
        assert response.status_code == 405
        assert response.json()["detail"]["error"] == "model_settings_read_only"
    assert providers.generation_requests == []
    assert _stored_rows(db) == []


async def test_a_deployment_managed_installation_keeps_its_checks_in_memory(
    db, tmp_path, providers, model_checks_at_server_start
) -> None:
    config = _deployment_config(tmp_path)

    with TestClient(create_admin_app(db=db, config=config)) as client:
        health = _wait_for_state(client, "embedding", "ready")
        _wait_for_state(client, "generation", "ready")
        active = client.get(API).json()["generation"]["active"]

    assert health["embedding"] == {
        "configured": True,
        "state": "ready",
        "vector_index": "ready",
        "reindex_required": False,
    }
    assert active["state"] == "ready"
    assert active["validation"]["passed"] is True
    assert active["saved_at"] is None
    assert "api_key" not in providers.generation_requests[0]
    assert "api_base" not in providers.generation_requests[0]
    assert _stored_rows(db) == []
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM model_secrets").fetchone() == (0,)


async def test_a_checked_deployment_binding_needs_revalidation_once_its_environment_changes(
    db, tmp_path, providers
) -> None:
    config = _deployment_config(tmp_path)
    settings = _settings(config)
    await settings.validate_unready_active_bindings(db)
    assert (await settings.health(db))["generation"].state == "ready"

    config.llm.enrichment_model = "litellm_proxy/another-model"
    changed = await settings.view(db, "generation")

    assert changed.active.binding.route == "litellm_proxy/another-model"
    assert changed.active.binding.state == "needs_revalidation"
    assert changed.active.binding.validation.digest != changed.active.binding.digest


def test_a_key_that_cannot_be_encrypted_is_refused_and_not_stored(db, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", "not-a-usable-secret-key")
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "secret_storage_unavailable"
    assert GENERATION_KEY not in response.text
    assert _stored_rows(db) == []


# -- Health -------------------------------------------------------------------------


def test_health_tells_when_no_generation_model_is_ready(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    first_run = client.get("/api/v1/health").json()
    client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})
    with_candidate = client.get("/api/v1/health").json()
    client.post(f"{API}/generation/candidate/validate")
    validated = client.get("/api/v1/health").json()

    assert first_run["models"]["generation"] == {"configured": False, "state": None}
    assert with_candidate["models"]["generation"] == {"configured": False, "state": None}
    assert validated["models"]["generation"] == {"configured": True, "state": "ready"}
    assert validated["models"]["embedding"] == {
        "configured": False,
        "state": None,
        "vector_index": "not_configured",
        "reindex_required": False,
    }
    assert first_run["status"] == validated["status"]


# -- Token limits -------------------------------------------------------------------

UNLISTED_GENERATION = {
    "provider": "openai_compatible",
    "endpoint": "https://generate.example.test/v1",
    "model_id": "a-model-litellm-does-not-list",
}
TOKEN_LIMITS = {"context_window_tokens": 32_768, "max_output_tokens": 4_096}


def _config_without_environment_limits(tmp_path: Path) -> AppConfig:
    config = _config(tmp_path)
    config.llm.max_input_tokens = None
    config.llm.context_window_tokens = None
    config.llm.max_output_tokens = None
    return config


def test_a_model_litellm_has_no_limits_for_fails_its_check_until_the_limits_are_given(
    db, tmp_path, providers
) -> None:
    client = TestClient(create_admin_app(db=db, config=_config_without_environment_limits(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**UNLISTED_GENERATION, "api_key": GENERATION_KEY})

    without = client.post(f"{API}/generation/candidate/validate").json()
    saved = client.put(f"{API}/generation/candidate", json={**UNLISTED_GENERATION, "token_limits": TOKEN_LIMITS})
    with_limits = client.post(f"{API}/generation/candidate/validate").json()

    assert (without["passed"], without["promoted"], without["error_code"]) == (False, False, "token_limits_required")
    assert without["validation"]["facts"]["token_limits_known"] is False
    assert without["settings"]["candidate"]["token_limits"] == {}
    assert saved.status_code == 200, saved.text
    assert saved.json()["candidate"]["token_limits"] == TOKEN_LIMITS
    assert saved.json()["candidate"]["validation"] is None
    assert (with_limits["passed"], with_limits["promoted"], with_limits["error_code"]) == (True, True, None)
    assert with_limits["validation"]["facts"]["token_limits_known"] is True
    assert with_limits["settings"]["active"]["token_limits"] == TOKEN_LIMITS
    assert len(text_requests(providers)) == 2


async def test_the_runtime_sizes_its_requests_with_the_limits_of_the_binding(db, tmp_path, providers) -> None:
    config = _config_without_environment_limits(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(
        db,
        "generation",
        CandidateDraft(**UNLISTED_GENERATION, api_key=GENERATION_KEY, token_limits=TOKEN_LIMITS),
    )
    assert (await settings.validate_candidate(db, "generation")).promoted is True

    models = await resolve_models(db, config)
    budget = DefaultRuntimeProvider().build_structured_llm_client(models, max_concurrent=1).request_budget()

    assert (models.context_window_tokens, models.max_output_tokens, models.max_input_tokens) == (32_768, 4_096, 32_768)
    assert (budget.context_limit, budget.output_limit, budget.input_limit) == (32_768, 4_096, 32_768)
    assert "context_window_tokens" not in providers.generation_requests[0]
    assert "max_output_tokens" not in providers.generation_requests[0]


async def test_limits_from_the_environment_apply_to_what_the_binding_leaves_open(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(db, "generation", CandidateDraft(**UNLISTED_GENERATION, api_key=GENERATION_KEY))
    from_environment = await settings.validate_candidate(db, "generation")
    environment_only = await resolve_models(db, config)

    await settings.save_candidate(
        db,
        "generation",
        CandidateDraft(**UNLISTED_GENERATION, token_limits={"context_window_tokens": 16_000}),
    )
    partly_stated = await settings.validate_candidate(db, "generation")
    with_binding = await resolve_models(db, config)

    assert (from_environment.promoted, partly_stated.promoted) == (True, True)
    assert (
        environment_only.context_window_tokens,
        environment_only.max_output_tokens,
        environment_only.max_input_tokens,
    ) == (64_000, 8_000, 64_000)
    assert (
        with_binding.context_window_tokens,
        with_binding.max_output_tokens,
        with_binding.max_input_tokens,
    ) == (16_000, 8_000, 64_000)


def test_changing_the_token_limits_of_a_ready_binding_needs_a_new_check(db, tmp_path, providers) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**UNLISTED_GENERATION, "api_key": GENERATION_KEY})
    client.post(f"{API}/generation/candidate/validate")

    saved = client.put(f"{API}/generation/candidate", json={**UNLISTED_GENERATION, "token_limits": TOKEN_LIMITS})

    candidate = saved.json()["candidate"]
    assert (candidate["state"], candidate["token_limits"]) == ("not_validated", TOKEN_LIMITS)
    assert saved.json()["active"]["token_limits"] == {}
    assert saved.json()["active"]["state"] == "ready"


@pytest.mark.parametrize(
    ("kind", "body", "error_code"),
    [
        ("generation", {**UNLISTED_GENERATION, "token_limits": {"context_window_tokens": 0}}, "token_limit_invalid"),
        ("generation", {**UNLISTED_GENERATION, "token_limits": {"max_output_tokens": "4096"}}, "token_limit_invalid"),
        ("generation", {**UNLISTED_GENERATION, "token_limits": {"max_input_tokens": 1}}, "token_limit_not_supported"),
        (
            "embedding",
            {**COMPATIBLE_EMBEDDING, "token_limits": {"context_window_tokens": 8_000}},
            "token_limit_not_supported",
        ),
    ],
    ids=["zero", "text", "unknown_limit", "embedding_model"],
)
def test_a_token_limit_a_binding_cannot_state_is_refused(db, tmp_path, kind, body, error_code) -> None:
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.put(f"{API}/{kind}/candidate", json={**body, "api_key": GENERATION_KEY})

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == error_code
    assert _stored_rows(db) == []


# -- Model discovery ----------------------------------------------------------------


class Endpoint:
    """Stand in for the HTTP client of a model listing request and record what it was asked."""

    def __init__(self, monkeypatch, respond=None) -> None:
        self.client_options: list[dict] = []
        self.requests: list[tuple[str, dict]] = []
        self._respond = respond or _model_list
        monkeypatch.setattr(model_settings_api.httpx, "AsyncClient", self)

    def __call__(self, **options):
        self.client_options.append(options)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, headers):
        self.requests.append((url, dict(headers)))
        return self._respond(url, httpx.Request("GET", url))


def _model_list(url, request):
    return httpx.Response(
        200,
        json={"data": [{"id": "model-a"}, {"id": "model-b", "display_name": "Model B"}]},
        request=request,
    )


def _discover(client: TestClient, kind: str = "embedding", **body: Any) -> dict[str, Any]:
    response = client.post(f"{API}/{kind}/discover-models", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_discovery_lists_the_models_of_an_endpoint(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test/v1", api_key=OTHER_KEY
    )

    assert payload["ok"] is True
    assert payload["models_supported"] is True
    assert payload["models"] == [
        {
            "id": "model-a",
            "label": None,
            "context_window_tokens": None,
            "max_output_tokens": None,
            "native_schema": None,
            "image_input": None,
        },
        {
            "id": "model-b",
            "label": "Model B",
            "context_window_tokens": None,
            "max_output_tokens": None,
            "native_schema": None,
            "image_input": None,
        },
    ]
    assert payload["kind_filtered"] is False
    assert payload["message"] == "Connected. Found 2 models; the endpoint does not say which are embedding models."
    assert payload["error_code"] is None
    assert endpoint.requests == [
        (
            "https://proxy.example.test/v1/models",
            {"Accept": "application/json", "Authorization": f"Bearer {OTHER_KEY}"},
        )
    ]
    assert [options["follow_redirects"] for options in endpoint.client_options] == [False]


OPENROUTER_CHAT_MODELS = {
    "data": [
        {
            "id": "anthropic/claude-haiku-5.5",
            "name": "Anthropic: Claude Haiku 5.5",
            "context_length": 1_000_000,
            "top_provider": {"context_length": 1_000_000, "max_completion_tokens": 128_000},
            "supported_parameters": ["max_tokens", "response_format", "structured_outputs", "tools"],
            "architecture": {"input_modalities": ["text", "image", "file"]},
        },
        {
            "id": "openai/gpt-5-mini",
            "name": "OpenAI: GPT-5 Mini",
            "top_provider": {"max_completion_tokens": None},
            "supported_parameters": ["max_tokens", "response_format"],
            "architecture": {"input_modalities": ["text"]},
        },
        {"id": "meituan/longcat-2.0", "name": "Meituan: LongCat 2.0"},
    ]
}
OPENROUTER_EMBEDDING_MODELS = {
    "data": [
        {
            "id": "openai/text-embedding-3-small",
            "name": "OpenAI: Text Embedding 3 Small",
            "context_length": 8_192,
            "supported_parameters": ["dimensions"],
        },
        {"id": "qwen/qwen3-embedding-8b", "name": "Qwen: Qwen3 Embedding 8B"},
    ]
}


def _openrouter(url, request):
    if url == "https://openrouter.ai/api/v1/embeddings/models":
        return httpx.Response(200, json=OPENROUTER_EMBEDDING_MODELS, request=request)
    if url == "https://openrouter.ai/api/v1/models":
        return httpx.Response(200, json=OPENROUTER_CHAT_MODELS, request=request)
    return httpx.Response(404, json={}, request=request)


def test_openrouter_lists_each_kind_from_its_own_listing(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch, _openrouter)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    embedding = _discover(client, "embedding", provider="openrouter", api_key=OTHER_KEY)
    generation = _discover(client, "generation", provider="openrouter", api_key=OTHER_KEY)

    assert [url for url, _headers in endpoint.requests] == [
        "https://openrouter.ai/api/v1/embeddings/models",
        "https://openrouter.ai/api/v1/models",
    ]
    assert [model["id"] for model in embedding["models"]] == ["openai/text-embedding-3-small", "qwen/qwen3-embedding-8b"]
    assert [model["native_schema"] for model in embedding["models"]] == [None, None]
    assert (embedding["kind_filtered"], embedding["message"]) == (True, "Connected. Found 2 embedding models.")
    assert [model["id"] for model in generation["models"]] == [
        "anthropic/claude-haiku-5.5",
        "openai/gpt-5-mini",
        "meituan/longcat-2.0",
    ]
    assert generation["message"] == "Connected. Found 3 generation models."


def test_discovery_carries_the_token_limits_response_schema_support_and_image_input_a_listing_gives(
    db, tmp_path, monkeypatch
) -> None:
    """OpenRouter declares a model's response schema support by ``structured_outputs``; JSON mode alone is not it."""
    Endpoint(monkeypatch, _openrouter)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    generation = _discover(client, "generation", provider="openrouter", api_key=OTHER_KEY)

    assert generation["models"] == [
        {
            "id": "anthropic/claude-haiku-5.5",
            "label": "Anthropic: Claude Haiku 5.5",
            "context_window_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "native_schema": True,
            "image_input": True,
        },
        {
            "id": "openai/gpt-5-mini",
            "label": "OpenAI: GPT-5 Mini",
            "context_window_tokens": None,
            "max_output_tokens": None,
            "native_schema": False,
            "image_input": False,
        },
        {
            "id": "meituan/longcat-2.0",
            "label": "Meituan: LongCat 2.0",
            "context_window_tokens": None,
            "max_output_tokens": None,
            "native_schema": None,
            "image_input": None,
        },
    ]


def test_a_listing_that_declares_no_response_schema_support_gives_none(db, tmp_path, monkeypatch) -> None:
    """Only a listing known to declare it is read for it, whatever other entries carry."""

    def respond(url, request):
        return httpx.Response(
            200,
            json={"data": [{"id": "served-model", "supported_parameters": ["structured_outputs"]}]},
            request=request,
        )

    Endpoint(monkeypatch, respond)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    generation = _discover(
        client, "generation", provider="openai_compatible", endpoint="https://models.example.test/v1"
    )

    assert [(model["id"], model["native_schema"]) for model in generation["models"]] == [("served-model", None)]


OPENROUTER_GENERATION = {"provider": "openrouter", "model_id": "anthropic/claude-haiku-5.5"}


@pytest.mark.parametrize(
    ("declared", "native"),
    [(True, True), (False, False), (None, True)],
    ids=["declared", "declined", "undeclared-the-model-map-decides"],
)
async def test_a_model_chosen_from_the_listing_is_asked_for_its_schema_as_the_listing_declares(
    db, tmp_path, providers, declared, native
) -> None:
    """The fixture's LiteLLM model map takes a schema for every model; a declaration overrides it."""
    config = _config(tmp_path)
    settings = _settings(config)
    await settings.save_candidate(
        db,
        "generation",
        CandidateDraft(**OPENROUTER_GENERATION, api_key=GENERATION_KEY, native_schema=declared),
    )

    outcome = await settings.validate_candidate(db, "generation")
    models = await resolve_models(db, config)

    assert outcome.promoted is True
    assert outcome.validation.facts["structured_output_mode"] == ("native_schema" if native else "json_text")
    assert ("response_format" in providers.generation_requests[0]) is native
    assert models.generation.native_schema is declared


def test_the_declared_response_schema_support_is_saved_and_shown_for_a_generation_model_only(
    db, tmp_path, providers, monkeypatch
) -> None:
    Endpoint(monkeypatch, lambda url, request: httpx.Response(503, json={}, request=request))
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/generation/candidate", json={**OPENROUTER_GENERATION, "api_key": GENERATION_KEY})
    client.post(f"{API}/generation/candidate/validate")

    saved = client.put(f"{API}/generation/candidate", json={**OPENROUTER_GENERATION, "native_schema": True})
    refused = client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "native_schema": True})

    assert saved.status_code == 200, saved.text
    assert (saved.json()["active"]["native_schema"], saved.json()["active"]["state"]) == (None, "ready")
    assert (saved.json()["candidate"]["native_schema"], saved.json()["candidate"]["state"]) == (True, "not_validated")
    assert refused.status_code == 422
    assert refused.json()["detail"]["error"] == "native_schema_not_supported"


def test_a_pasted_listing_path_is_read_as_the_base_url_it_follows(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch, _openrouter)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client,
        "embedding",
        provider="openrouter",
        endpoint="https://openrouter.ai/api/v1/models",
        api_key=OTHER_KEY,
    )
    saved = client.put(
        f"{API}/generation/candidate",
        json={
            "provider": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "model_id": "anthropic/claude-haiku-5.5",
            "api_key": OTHER_KEY,
        },
    )

    assert payload["message"] == "Connected. Found 2 embedding models."
    assert [url for url, _headers in endpoint.requests] == ["https://openrouter.ai/api/v1/embeddings/models"]
    assert saved.status_code == 200, saved.text
    assert saved.json()["candidate"]["endpoint"] == "https://openrouter.ai/api/v1"


class Listing:
    """Stand in for what the model listing declares, and record the bindings and keys it was asked about."""

    def __init__(self, declared: bool | None = None, failure: Exception | None = None) -> None:
        self.declared = declared
        self.failure = failure
        self.asked: list[tuple[str, str | None]] = []

    async def __call__(self, binding, api_key):
        self.asked.append((binding.model_id, api_key))
        if self.failure is not None:
            raise self.failure
        return self.declared


def _listing_settings(config: AppConfig, listing: Listing) -> ModelSettings:
    return ModelSettings(config, DefaultRuntimeProvider(), listed_native_schema=listing)


async def _active_openrouter_model(db, settings: ModelSettings, native_schema: bool | None = None) -> None:
    await settings.save_candidate(
        db, "generation", CandidateDraft(**OPENROUTER_GENERATION, api_key=GENERATION_KEY, native_schema=native_schema)
    )
    assert (await settings.validate_candidate(db, "generation")).promoted is True


@pytest.mark.parametrize(
    ("saved", "listed", "native"),
    [(None, True, True), (True, False, False), (False, True, True)],
    ids=["undeclared-now-declared", "declared-now-dropped", "declined-now-added"],
)
async def test_save_and_validate_checks_the_model_with_what_the_listing_declares_now(
    db, tmp_path, providers, saved, listed, native
) -> None:
    config = _config(tmp_path)
    listing = Listing(listed)

    await _active_openrouter_model(db, _listing_settings(config, listing), saved)
    active = bindings.binding_from_record(await db.get_model_binding("generation", "active"))

    assert listing.asked == [("anthropic/claude-haiku-5.5", GENERATION_KEY)]
    assert ("response_format" in providers.generation_requests[0]) is native
    assert (active.native_schema, active.state) == (listed, "ready")
    assert (await resolve_models(db, config)).generation.native_schema is listed


@pytest.mark.parametrize("listed", [True, False])
async def test_a_starting_server_refreshes_a_ready_model_from_the_listing_without_checking_it_again(
    db, tmp_path, providers, listed
) -> None:
    config = _config(tmp_path)
    await _active_openrouter_model(db, _settings(config), not listed)
    before = bindings.binding_from_record(await db.get_model_binding("generation", "active"))
    listing = Listing(listed)

    await _listing_settings(config, listing).validate_unready_active_bindings(db)
    after = bindings.binding_from_record(await db.get_model_binding("generation", "active"))

    assert listing.asked == [("anthropic/claude-haiku-5.5", GENERATION_KEY)]
    assert (after.native_schema, after.state, after.saved_at) == (listed, "ready", before.saved_at)
    assert (after.digest, after.validation) == (before.digest, before.validation)
    assert len(text_requests(providers)) == 1


async def test_checking_the_model_in_use_again_refreshes_it_from_the_listing(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    await _active_openrouter_model(db, _settings(config))

    await _listing_settings(config, Listing(True)).validate_active(db, "generation")

    assert bindings.binding_from_record(await db.get_model_binding("generation", "active")).native_schema is True
    assert "response_format" in providers.generation_requests[-1]


@pytest.mark.parametrize(
    "listing",
    [Listing(None), Listing(failure=httpx.ConnectError("offline"))],
    ids=["declares-nothing", "unreachable"],
)
async def test_a_listing_that_says_nothing_keeps_the_stored_support(db, tmp_path, providers, listing) -> None:
    config = _config(tmp_path)
    await _active_openrouter_model(db, _settings(config), True)

    await _listing_settings(config, listing).validate_unready_active_bindings(db)
    await _listing_settings(config, listing).validate_active(db, "generation")
    after = bindings.binding_from_record(await db.get_model_binding("generation", "active"))

    assert len(listing.asked) == 2
    assert (after.native_schema, after.state) == (True, "ready")
    assert "response_format" in providers.generation_requests[-1]


async def test_only_a_model_whose_listing_declares_response_schema_support_is_refreshed(
    db, tmp_path, providers
) -> None:
    config = _config(tmp_path)
    listing = Listing(True)
    settings = _listing_settings(config, listing)
    await settings.save_candidate(db, "generation", CandidateDraft(**OPENAI_GENERATION, api_key=GENERATION_KEY))
    await settings.validate_candidate(db, "generation")
    await settings.validate_unready_active_bindings(db)
    await settings.validate_active(db, "generation")

    assert listing.asked == []
    assert bindings.binding_from_record(await db.get_model_binding("generation", "active")).native_schema is None


async def test_a_starting_server_refreshes_the_stored_model_from_the_provider_listing(
    db, tmp_path, providers, monkeypatch, model_checks_at_server_start
) -> None:
    config = _config(tmp_path)
    await _active_openrouter_model(db, _settings(config))
    Endpoint(monkeypatch, _openrouter)

    with TestClient(create_admin_app(db=db, config=config)) as client:
        deadline = time.monotonic() + 5
        while client.get(API).json()["generation"]["active"]["native_schema"] is None:
            assert time.monotonic() < deadline, "the starting server did not refresh the listed support"
            time.sleep(0.01)

    stored = bindings.binding_from_record(await db.get_model_binding("generation", "active"))
    assert (stored.native_schema, stored.state) == (True, "ready")
    assert len(text_requests(providers)) == 1


async def test_the_listing_is_asked_what_it_declares_about_the_stored_model(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch, _openrouter)
    listed = {
        model_id: bindings.build_binding(
            kind="generation",
            provider="openrouter",
            model_id=model_id,
            auth_strategy="environment",
            config_source="settings",
        )
        for model_id in ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini", "meituan/longcat-2.0", "unlisted/model")
    }

    declared = {model_id: await model_settings_api.listed_native_schema(binding, None) for model_id, binding in listed.items()}

    assert declared == {
        "anthropic/claude-haiku-5.5": True,
        "openai/gpt-5-mini": False,
        "meituan/longcat-2.0": None,
        "unlisted/model": None,
    }
    assert {url for url, _headers in endpoint.requests} == {"https://openrouter.ai/api/v1/models"}


async def test_a_listing_that_fails_declares_nothing(db, tmp_path, monkeypatch) -> None:
    Endpoint(monkeypatch, lambda url, request: httpx.Response(503, json={}, request=request))
    binding = bindings.build_binding(
        kind="generation",
        provider="openrouter",
        model_id="anthropic/claude-haiku-5.5",
        auth_strategy="environment",
        config_source="settings",
    )

    assert await model_settings_api.listed_native_schema(binding, None) is None


OPENAI_MODELS = {
    "object": "list",
    "data": [
        {"id": "gpt-5-mini", "object": "model", "owned_by": "system"},
        {"id": "text-embedding-3-small", "object": "model", "owned_by": "system"},
        {"id": "dall-e-3", "object": "model", "owned_by": "system"},
        {"id": "whisper-1", "object": "model", "owned_by": "system"},
        {"id": "gpt-4o-mini-tts", "object": "model", "owned_by": "system"},
        {"id": "omni-moderation-latest", "object": "model", "owned_by": "system"},
        {"id": "text-embedding-3-large", "object": "model", "owned_by": "system"},
        {"id": "o4-mini", "object": "model", "owned_by": "system"},
    ],
}


def test_openai_tells_the_kinds_apart_by_model_id_in_its_one_listing(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch, lambda url, request: httpx.Response(200, json=OPENAI_MODELS, request=request))
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    embedding = _discover(client, "embedding", provider="openai", api_key=OTHER_KEY)
    generation = _discover(client, "generation", provider="openai", api_key=OTHER_KEY)

    assert [url for url, _headers in endpoint.requests] == ["https://api.openai.com/v1/models"] * 2
    assert [model["id"] for model in embedding["models"]] == ["text-embedding-3-small", "text-embedding-3-large"]
    assert [model["id"] for model in generation["models"]] == ["gpt-5-mini", "o4-mini"]
    assert (embedding["kind_filtered"], generation["kind_filtered"]) == (True, True)


OLLAMA_TAGS = {
    "models": [
        {"name": "qwen3:14b", "model": "qwen3:14b", "details": {"family": "qwen3", "families": ["qwen3"]}},
        {"name": "bge-m3:latest", "model": "bge-m3:latest", "details": {"family": "bert", "families": ["bert"]}},
        {
            "name": "qwen3-embedding:8b",
            "model": "qwen3-embedding:8b",
            "details": {"family": "qwen3", "families": ["qwen3"]},
        },
        {"name": "custom-encoder:1", "model": "custom-encoder:1", "capabilities": ["embedding"]},
        {"name": "custom-chat:1", "model": "custom-chat:1", "capabilities": ["completion", "tools"]},
    ]
}


def test_ollama_lists_its_tags_and_tells_the_kinds_apart_by_capability_family_or_name(
    db, tmp_path, monkeypatch
) -> None:
    endpoint = Endpoint(
        monkeypatch,
        lambda url, request: (
            httpx.Response(200, json=OLLAMA_TAGS, request=request)
            if url == "http://localhost:11434/api/tags"
            else httpx.Response(404, json={}, request=request)
        ),
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    embedding = _discover(client, "embedding", provider="ollama")
    generation = _discover(client, "generation", provider="ollama", endpoint="http://localhost:11434/v1")

    assert [url for url, _headers in endpoint.requests] == ["http://localhost:11434/api/tags"] * 2
    assert [model["id"] for model in embedding["models"]] == [
        "bge-m3:latest",
        "qwen3-embedding:8b",
        "custom-encoder:1",
    ]
    assert [model["id"] for model in generation["models"]] == ["qwen3:14b", "custom-chat:1"]


def test_a_listing_with_no_model_of_the_kind_says_so_instead_of_offering_the_others(
    db, tmp_path, monkeypatch
) -> None:
    Endpoint(
        monkeypatch,
        lambda url, request: httpx.Response(200, json={"data": [{"id": "gpt-5-mini"}, {"id": "o4-mini"}]}, request=request),
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, "embedding", provider="openai", api_key=OTHER_KEY)

    assert (payload["ok"], payload["models_supported"], payload["kind_filtered"]) == (True, True, True)
    assert payload["models"] == []
    assert payload["message"] == (
        "Connected, but this endpoint listed no embedding models. Type the model id if you know it."
    )


def test_a_provider_is_not_asked_for_a_kind_it_does_not_serve(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, "embedding", provider="anthropic", api_key=OTHER_KEY)

    assert (payload["ok"], payload["models_supported"]) == (True, False)
    assert payload["message"] == "Anthropic serves no embedding models."
    assert endpoint.requests == []


def test_discovery_uses_the_default_endpoint_and_key_header_of_the_provider(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(
        monkeypatch,
        lambda url, request: (
            _model_list(url, request)
            if url.endswith("/v1/models")
            else httpx.Response(404, json={}, request=request)
        ),
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, "generation", provider="anthropic", api_key=OTHER_KEY)

    assert payload["ok"] is True
    assert [url for url, _headers in endpoint.requests] == [
        "https://api.anthropic.com/models",
        "https://api.anthropic.com/v1/models",
    ]
    assert endpoint.requests[0][1] == {
        "Accept": "application/json",
        "x-api-key": OTHER_KEY,
        "anthropic-version": "2023-06-01",
    }


def test_a_provider_without_a_model_list_is_not_asked(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, provider="litellm", api_key=OTHER_KEY)

    assert payload["ok"] is True
    assert payload["models_supported"] is False
    assert endpoint.requests == []


@pytest.mark.parametrize(
    "requested",
    ["https://embed.example.test/v1", "https://EMBED.Example.Test:443/other/path"],
    ids=["same_url", "same_origin_other_spelling"],
)
def test_discovery_uses_the_saved_key_for_its_own_origin(db, tmp_path, monkeypatch, requested) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    payload = _discover(client, provider="openai_compatible", endpoint=requested)

    assert payload["ok"] is True
    assert endpoint.requests[0][1]["Authorization"] == f"Bearer {EMBEDDING_KEY}"
    assert EMBEDDING_KEY not in json.dumps(payload)


@pytest.mark.parametrize(
    "requested",
    [
        "https://other.example.test/v1",
        "http://embed.example.test/v1",
        "https://embed.example.test:8443/v1",
    ],
    ids=["other_host", "other_scheme", "other_port"],
)
def test_discovery_asks_for_the_key_again_on_another_origin(db, tmp_path, monkeypatch, requested) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    payload = _discover(client, provider="openai_compatible", endpoint=requested)

    assert payload["ok"] is False
    assert payload["stage"] == "auth"
    assert payload["error_code"] == "api_key_origin_changed"
    assert "Enter the API key for this endpoint" in payload["message"]
    assert endpoint.requests == []


def test_discovery_sends_a_key_entered_for_another_origin(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://other.example.test/v1", api_key=OTHER_KEY
    )

    assert payload["ok"] is True
    assert endpoint.requests[0][1]["Authorization"] == f"Bearer {OTHER_KEY}"


def test_discovery_uses_the_environment_key_only_for_the_environment_endpoint(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    config = _config(tmp_path)
    config.llm.embedding_model = "embed-model"
    config.llm.embedding_base_url = "https://configured.example.test/v1"
    config.llm.embedding_api_key = ENVIRONMENT_KEY
    client = TestClient(create_admin_app(db=db, config=config))

    same_origin = _discover(client, provider="openai_compatible", endpoint="https://configured.example.test/v1")
    other_origin = _discover(client, provider="openai_compatible", endpoint="https://other.example.test/v1")

    assert same_origin["ok"] is True
    assert other_origin["ok"] is False
    assert other_origin["error_code"] == "api_key_origin_changed"
    assert [headers["Authorization"] for _url, headers in endpoint.requests] == [f"Bearer {ENVIRONMENT_KEY}"]


def test_discovery_with_the_none_strategy_sends_no_saved_key(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://embed.example.test/v1", auth_strategy="none"
    )

    assert payload["ok"] is True
    assert endpoint.requests[0][1] == {"Accept": "application/json"}


def test_discovery_reports_a_redirect_to_another_origin_as_failed(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(
        monkeypatch,
        lambda url, request: httpx.Response(
            302,
            headers={"location": "https://collector.example.test/v1/models?from=probe"},
            request=request,
        ),
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test/v1", api_key=OTHER_KEY
    )

    assert payload["ok"] is False
    assert payload["stage"] == "http"
    assert payload["error_code"] == "redirect_to_another_origin"
    assert payload["status"] == 302
    assert "redirected to https://collector.example.test, another origin" in payload["message"]
    assert "from=probe" not in payload["message"]
    assert [url for url, _headers in endpoint.requests] == ["https://proxy.example.test/v1/models"]


@pytest.mark.parametrize(
    "location",
    ["http://proxy.example.test/v1/models", "https://proxy.example.test:8443/v1/models"],
    ids=["other_scheme", "other_port"],
)
def test_discovery_treats_a_changed_port_or_scheme_as_another_origin(db, tmp_path, monkeypatch, location) -> None:
    Endpoint(monkeypatch, lambda url, request: httpx.Response(307, headers={"location": location}, request=request))
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test/v1", api_key=OTHER_KEY
    )

    assert payload["ok"] is False
    assert payload["error_code"] == "redirect_to_another_origin"


def test_discovery_does_not_follow_a_redirect_within_the_origin(db, tmp_path, monkeypatch) -> None:
    def respond(url, request):
        if url.endswith("/v1/models"):
            return _model_list(url, request)
        return httpx.Response(301, headers={"location": "/elsewhere/models"}, request=request)

    endpoint = Endpoint(monkeypatch, respond)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test", api_key=OTHER_KEY
    )

    assert payload["ok"] is True
    assert [url for url, _headers in endpoint.requests] == [
        "https://proxy.example.test/models",
        "https://proxy.example.test/v1/models",
    ]


def test_discovery_fails_when_the_endpoint_only_redirects(db, tmp_path, monkeypatch) -> None:
    Endpoint(
        monkeypatch,
        lambda url, request: httpx.Response(308, headers={"location": "/moved"}, request=request),
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test", api_key=OTHER_KEY
    )

    assert payload["ok"] is False
    assert payload["stage"] == "http"
    assert payload["error_code"] == "redirect_not_followed"
    assert payload["status"] == 308


def test_discovery_treats_a_missing_model_list_as_a_model_to_enter_by_hand(db, tmp_path, monkeypatch) -> None:
    Endpoint(monkeypatch, lambda url, request: httpx.Response(404, json={"error": "not found"}, request=request))
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test", api_key=OTHER_KEY
    )

    assert payload["ok"] is True
    assert payload["models_supported"] is False
    assert payload["models"] == []
    assert "does not expose a model list" in payload["message"]


def test_discovery_falls_through_from_an_html_answer_to_the_v1_path(db, tmp_path, monkeypatch) -> None:
    def respond(url, request):
        if url.endswith("/v1/models"):
            return httpx.Response(200, json={"data": [{"id": "model-from-v1"}]}, request=request)
        return httpx.Response(200, content=b"<html>not json</html>", request=request)

    Endpoint(monkeypatch, respond)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test", api_key=OTHER_KEY
    )

    assert [model["id"] for model in payload["models"]] == ["model-from-v1"]


def test_discovery_suggests_the_docker_host_for_an_unreachable_localhost(db, tmp_path, monkeypatch) -> None:
    def refuse(url, request):
        raise httpx.ConnectError("connection refused", request=request)

    Endpoint(monkeypatch, refuse)
    monkeypatch.setattr(model_settings_api, "_is_running_in_container", lambda: True)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, provider="ollama", endpoint="http://localhost:11434")

    assert payload["ok"] is False
    assert payload["stage"] == "connect"
    assert payload["error_code"] == "provider_unreachable"
    assert payload["suggested_endpoint"] == "http://host.docker.internal:11434"


def test_discovery_without_a_key_says_to_add_one_when_the_endpoint_wants_one(db, tmp_path, monkeypatch) -> None:
    Endpoint(monkeypatch, lambda url, request: httpx.Response(401, json={"error": "unauthorized"}, request=request))
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    without_key = _discover(client, provider="openai_compatible", endpoint="https://proxy.example.test/v1")
    with_key = _discover(
        client, provider="openai_compatible", endpoint="https://proxy.example.test/v1", api_key=OTHER_KEY
    )

    assert (without_key["stage"], without_key["error_code"]) == ("auth", "api_key_required")
    assert without_key["message"] == "Add an API key, then try again."
    assert (with_key["stage"], with_key["error_code"]) == ("auth", "authentication_failed")
    assert OTHER_KEY not in json.dumps(with_key)


def test_discovery_with_environment_credentials_says_the_model_has_to_be_entered(
    db, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    endpoint = Endpoint(
        monkeypatch, lambda url, request: httpx.Response(401, json={"error": "unauthorized"}, request=request)
    )
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, kind="generation", provider="openai", auth_strategy="environment")

    assert (payload["ok"], payload["stage"], payload["status"]) == (False, "auth", 401)
    assert payload["error_code"] == "listing_needs_api_key"
    assert payload["message"] == (
        "This endpoint lists its models only for an API key, and credentials from the "
        "server environment are not sent with this request. Enter the model yourself."
    )
    assert endpoint.requests == [("https://api.openai.com/v1/models", {"Accept": "application/json"})]


def test_discovery_with_environment_credentials_lists_an_endpoint_that_asks_for_no_key(
    db, tmp_path, monkeypatch
) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, kind="generation", provider="openrouter", auth_strategy="environment")

    assert payload["ok"] is True
    assert [model["id"] for model in payload["models"]] == ["model-a", "model-b"]
    assert endpoint.requests == [("https://openrouter.ai/api/v1/models", {"Accept": "application/json"})]


def test_discovery_refuses_an_endpoint_that_is_not_a_url(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    payload = _discover(client, provider="openai_compatible", endpoint="proxy.example.test/v1")

    assert (payload["ok"], payload["stage"], payload["error_code"]) == (False, "validation", "endpoint_invalid")
    assert endpoint.requests == []


def test_discovery_refuses_an_endpoint_that_carries_a_credential(db, tmp_path, monkeypatch) -> None:
    endpoint = Endpoint(monkeypatch)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    response = client.post(
        f"{API}/embedding/discover-models",
        json={"provider": "openai_compatible", "endpoint": "https://user:pw-secret@embed.example.test/v1"},
    )

    payload = response.json()
    assert (payload["ok"], payload["stage"], payload["error_code"]) == (False, "validation", "endpoint_invalid")
    assert "pw-secret" not in response.text
    assert endpoint.requests == []


@pytest.mark.parametrize("mode", STALL_MODES)
def test_discovery_of_an_endpoint_that_does_not_finish_answering_ends_as_a_timeout(
    db, tmp_path, monkeypatch, mode
) -> None:
    monkeypatch.setattr(model_settings_api, "_LISTING_TIMEOUT_S", 0.5)
    client = TestClient(create_admin_app(db=db, config=_config(tmp_path)))

    with stalling_endpoint(mode) as base_url:
        started = time.monotonic()
        payload = _discover(client, provider="openai_compatible", endpoint=f"{base_url}/v1", api_key=OTHER_KEY)
        elapsed_s = time.monotonic() - started

    assert (payload["ok"], payload["stage"], payload["error_code"]) == (False, "timeout", "timeout")
    assert elapsed_s < GIVES_UP_AFTER_S / 2


# -- Keys never leave ---------------------------------------------------------------


def test_no_response_and_no_log_line_contains_a_key(db, tmp_path, providers, monkeypatch, caplog) -> None:
    Endpoint(monkeypatch)
    config = _config(tmp_path)
    config.llm.embedding_model = "embed-model"
    config.llm.embedding_base_url = "https://configured.example.test/v1"
    config.llm.embedding_api_key = ENVIRONMENT_KEY
    client = TestClient(create_admin_app(db=db, config=config))
    responses = []

    with caplog.at_level("DEBUG"):
        responses.append(client.get(API))
        responses.append(
            client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": GENERATION_KEY})
        )
        providers.generation_failure = _rejected_key()
        responses.append(client.post(f"{API}/generation/candidate/validate"))
        providers.generation_failure = None
        responses.append(client.post(f"{API}/generation/candidate/validate"))
        responses.append(
            client.put(f"{API}/generation/candidate", json={**OPENAI_GENERATION, "api_key": OTHER_KEY})
        )
        responses.append(
            client.put(
                f"{API}/generation/candidate",
                json={"provider": "openai_compatible", "endpoint": "https://moved.example.test", "model_id": "m"},
            )
        )
        responses.append(
            client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})
        )
        providers.embedding_failure = Rejection("invalid api key")
        responses.append(client.post(f"{API}/embedding/candidate/validate"))
        providers.embedding_failure = None
        responses.append(client.post(f"{API}/embedding/candidate/validate"))
        responses.append(client.post(f"{API}/embedding/active/validate"))
        responses.append(
            client.post(
                f"{API}/embedding/discover-models",
                json={"provider": "openai_compatible", "endpoint": "https://embed.example.test/v1"},
            )
        )
        responses.append(client.get(API))
        responses.append(client.get("/api/v1/health"))
        responses.append(client.delete(f"{API}/generation/candidate"))

    assert [response.status_code for response in responses] == [
        200, 200, 200, 200, 200, 422, 200, 200, 200, 200, 200, 200, 200, 200,
    ]
    sent_keys = {request["api_key"] for request in providers.generation_requests + providers.embedding_requests}
    assert sent_keys == {GENERATION_KEY, EMBEDDING_KEY}
    for key in ALL_KEYS:
        assert all(key not in response.text for response in responses)
        assert key not in caplog.text
    final = responses[-3].json()
    assert final["generation"]["active"]["api_key"]["last4"] == "1111"
    assert final["generation"]["candidate"]["api_key"]["last4"] == "4444"
    assert final["embedding"]["active"]["api_key"]["last4"] == "2222"


async def test_the_resolved_binding_of_the_api_is_the_one_the_runtime_resolves(db, tmp_path, providers) -> None:
    config = _config(tmp_path)
    client = TestClient(create_admin_app(db=db, config=config))
    client.put(f"{API}/embedding/candidate", json={**COMPATIBLE_EMBEDDING, "api_key": EMBEDDING_KEY})
    client.post(f"{API}/embedding/candidate/validate")

    shown = client.get(API).json()["embedding"]["active"]
    resolved = await resolve_active_binding(db, config, "embedding")

    assert (shown["route"], shown["endpoint"], shown["state"]) == (resolved.route, resolved.endpoint, resolved.state)
