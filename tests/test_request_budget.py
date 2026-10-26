"""How a request budget is resolved for a model LiteLLM has no metadata for."""

from __future__ import annotations

import logging

import litellm
import pytest

from memforge.config import AppConfig, LlmConfig
from memforge.llm.request_budget import RequestBudget

LOGGER = "memforge.llm.request_budget"


@pytest.fixture
def unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_metadata(model: str) -> dict:
        raise Exception(f"This model isn't mapped yet. model={model}")

    monkeypatch.setattr(litellm, "get_model_info", no_metadata)


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == LOGGER]


def test_stated_limits_of_an_unknown_model_are_logged_at_debug(unknown_model, caplog) -> None:
    config = LlmConfig(max_input_tokens=32768, context_window_tokens=32768, max_output_tokens=4096)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        budget = RequestBudget.resolve("ollama_chat/qwen3:8b", config)

    assert (budget.input_limit, budget.context_limit, budget.output_limit) == (32768, 32768, 4096)
    assert budget.metadata_source == "operator_caps"
    records = _records(caplog)
    assert [record.levelno for record in records] == [logging.DEBUG]
    assert "using_explicit_operator_caps=true" in records[0].getMessage()


def test_a_limit_nothing_states_is_logged_as_a_warning(unknown_model, caplog) -> None:
    config = LlmConfig(max_input_tokens=32768, context_window_tokens=32768)

    with caplog.at_level(logging.DEBUG, logger=LOGGER), pytest.raises(ValueError, match="max_output_tokens"):
        RequestBudget.resolve("ollama_chat/qwen3:8b", config)

    records = _records(caplog)
    assert [record.levelno for record in records] == [logging.WARNING]
    assert "unconfigured_limits=max_output_tokens" in records[0].getMessage()


def test_a_known_model_logs_nothing(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"max_input_tokens": 128000, "context_window": 128000, "max_output_tokens": 16384},
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        budget = RequestBudget.resolve("openai/gpt-4o", LlmConfig())

    assert budget.metadata_source == "litellm_and_operator_caps"
    assert _records(caplog) == []


def test_a_window_sized_per_request_is_only_the_stated_one(monkeypatch) -> None:
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"max_input_tokens": 131072, "context_window": 131072, "max_output_tokens": 32768},
    )

    with pytest.raises(ValueError, match="context_window_tokens"):
        RequestBudget.resolve("ollama_chat/qwen3:8b", LlmConfig(), catalog_window=False)
    budget = RequestBudget.resolve(
        "ollama_chat/qwen3:8b", LlmConfig(context_window_tokens=16384), catalog_window=False,
    )

    assert (budget.input_limit, budget.context_limit, budget.output_limit) == (131072, 16384, 32768)
    assert budget.output_tokens == 4096


def _budget(*, input_limit: int, context_limit: int, output_limit: int, correction_reserve: int = 0) -> RequestBudget:
    return RequestBudget(
        "fixture/model", input_limit, context_limit, output_limit, 1.0, "fixture", correction_reserve,
    )


@pytest.mark.parametrize(
    ("context_limit", "output_limit", "expected"),
    [
        (200_000, 16_384, 16_384),  # the model's output limit binds
        (32_768, 16_384, 8_192),  # a quarter of the context window binds
        (3, 16_384, 1),  # a tiny window still asks for one token
    ],
)
def test_output_tokens_are_the_output_limit_within_a_quarter_of_the_context(
    context_limit, output_limit, expected,
) -> None:
    budget = _budget(input_limit=context_limit, context_limit=context_limit, output_limit=output_limit)

    assert budget.output_tokens == expected


def test_available_input_leaves_the_output_capacity_and_the_correction_reserve() -> None:
    budget = _budget(input_limit=32_768, context_limit=32_768, output_limit=16_384, correction_reserve=1_024)

    assert budget.available_input(reserve_correction=False) == 32_768 - 8_192
    assert budget.available_input() == 32_768 - 8_192 - 1_024
    assert budget.fits(32_768 - 8_192 - 1_024)
    assert not budget.fits(32_768 - 8_192 - 1_023)
    assert budget.fits(32_768 - 8_192, reserve_correction=False)


def test_available_input_is_bounded_by_the_input_limit() -> None:
    budget = _budget(input_limit=10_000, context_limit=200_000, output_limit=16_384)

    assert budget.available_input() == 10_000


def test_identity_follows_the_limits_a_request_is_sized_by() -> None:
    first = _budget(input_limit=1_000, context_limit=4_000, output_limit=500)

    assert first.identity == _budget(input_limit=1_000, context_limit=4_000, output_limit=500).identity
    assert first.identity != _budget(input_limit=1_000, context_limit=4_000, output_limit=600).identity


def test_a_route_the_deployment_maps_takes_the_limits_of_the_model_it_names(monkeypatch) -> None:
    looked_up: list[str] = []

    def catalog(model: str) -> dict:
        looked_up.append(model)
        if model != "anthropic/claude-sonnet-4-6":
            raise Exception(f"This model isn't mapped yet. model={model}")
        return {"max_input_tokens": 1_000_000, "context_window": 1_000_000, "max_output_tokens": 64_000}

    monkeypatch.setattr(litellm, "get_model_info", catalog)
    monkeypatch.setenv("MEMFORGE_LLM_METADATA_MODELS", '{"gateway/claude-sonnet": "anthropic/claude-sonnet-4-6"}')

    budget = RequestBudget.resolve("gateway/claude-sonnet", LlmConfig(max_output_tokens=32_000))
    unmapped = RequestBudget.resolve("anthropic/claude-sonnet-4-6", LlmConfig())

    assert looked_up == ["anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"]
    assert budget.model == "gateway/claude-sonnet"
    assert (budget.input_limit, budget.context_limit, budget.output_limit) == (1_000_000, 1_000_000, 32_000)
    assert budget.metadata_source == "litellm_and_operator_caps"
    assert unmapped.output_limit == 64_000


@pytest.mark.parametrize(
    "value",
    ["not json", "{'gateway/claude-sonnet': 'anthropic/claude-sonnet-4-6'}", '["gateway/claude-sonnet"]', '{"gateway/claude-sonnet": ""}'],
)
def test_a_malformed_metadata_model_mapping_stops_config_load(monkeypatch, tmp_path, value) -> None:
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("MEMFORGE_LLM_METADATA_MODELS", value)

    with pytest.raises(ValueError, match="MEMFORGE_LLM_METADATA_MODELS"):
        AppConfig()
