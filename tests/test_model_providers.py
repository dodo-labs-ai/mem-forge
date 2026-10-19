"""Provider descriptors, and the invocation a binding compiles to."""

from __future__ import annotations

import pytest

from memforge.llm.bindings import build_binding, compile_binding, embedding_identity
from memforge.llm.embedding_profile import EmbeddingIdentity
from memforge.llm.providers import (
    KEYLESS_API_KEY,
    PROVIDER_DESCRIPTORS,
    ModelConfigError,
    ResolvedInvocation,
    names_litellm_route,
    provider_descriptor,
)


def _binding(provider: str, kind: str = "generation", **settings):
    settings.setdefault("model_id", "model-a")
    return build_binding(kind=kind, provider=provider, config_source="settings", **settings)


def test_the_six_descriptors_are_offered_in_order() -> None:
    assert [descriptor.id for descriptor in PROVIDER_DESCRIPTORS] == [
        "openai",
        "anthropic",
        "openrouter",
        "ollama",
        "openai_compatible",
        "litellm",
    ]


@pytest.mark.parametrize(
    ("provider", "kinds", "auth_strategies", "default_endpoint", "endpoint_required", "lists_models"),
    [
        ("openai", ("generation", "embedding"), ("api_key", "environment"), "https://api.openai.com/v1", False, True),
        ("anthropic", ("generation",), ("api_key", "environment"), "https://api.anthropic.com", False, True),
        (
            "openrouter",
            ("generation", "embedding"),
            ("api_key", "environment"),
            "https://openrouter.ai/api/v1",
            False,
            True,
        ),
        ("ollama", ("generation", "embedding"), ("none", "api_key"), "http://localhost:11434", False, True),
        ("openai_compatible", ("generation", "embedding"), ("api_key", "none"), None, True, True),
        ("litellm", ("generation", "embedding"), ("api_key", "environment"), None, False, False),
    ],
)
def test_each_descriptor_states_what_it_serves_and_how_it_is_reached(
    provider, kinds, auth_strategies, default_endpoint, endpoint_required, lists_models
) -> None:
    descriptor = provider_descriptor(provider)

    assert descriptor.kinds == kinds
    assert descriptor.auth_strategies == auth_strategies
    assert descriptor.default_endpoint == default_endpoint
    assert descriptor.endpoint_required is endpoint_required
    assert descriptor.lists_models is lists_models


@pytest.mark.parametrize(
    ("provider", "lists_token_limits"),
    [
        ("openai", False),
        ("anthropic", True),
        ("openrouter", True),
        ("ollama", False),
        ("openai_compatible", True),
        ("litellm", False),
    ],
)
def test_only_a_listing_whose_entries_can_carry_token_limits_offers_them(provider, lists_token_limits) -> None:
    assert provider_descriptor(provider).lists_token_limits is lists_token_limits


@pytest.mark.parametrize(
    ("provider", "pasted", "base"),
    [
        ("openrouter", "https://openrouter.ai/api/v1/models", "https://openrouter.ai/api/v1"),
        ("openrouter", "https://openrouter.ai/api/v1/embeddings/models/", "https://openrouter.ai/api/v1"),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "https://openrouter.ai/api/v1"),
        ("openai", "https://api.openai.com/v1/embeddings", "https://api.openai.com/v1"),
        ("openai", "https://api.openai.com/v1/Models", "https://api.openai.com/v1"),
        ("openai_compatible", "http://localhost:8000/v1/completions", "http://localhost:8000/v1"),
        ("openai_compatible", "https://proxy.example.test/models", "https://proxy.example.test"),
        ("openai_compatible", "https://proxy.example.test/v1/models?tenant=a", "https://proxy.example.test/v1/models?tenant=a"),
        ("anthropic", "https://api.anthropic.com/v1/messages", "https://api.anthropic.com"),
        ("anthropic", "https://api.anthropic.com/v1", "https://api.anthropic.com"),
        ("ollama", "http://localhost:11434/api/tags", "http://localhost:11434"),
        ("ollama", "http://localhost:11434/v1", "http://localhost:11434"),
        ("ollama", "http://localhost:11434/v1/chat/completions", "http://localhost:11434"),
        ("litellm", "https://gateway.example.test/v1/models", "https://gateway.example.test/v1/models"),
        ("openai", " https://api.openai.com/v1/ ", "https://api.openai.com/v1"),
    ],
    ids=[
        "openrouter_models",
        "openrouter_embedding_models",
        "openrouter_chat",
        "openai_embeddings",
        "openai_models_any_case",
        "compatible_completions",
        "compatible_models_at_root",
        "query_kept",
        "anthropic_messages",
        "anthropic_version",
        "ollama_tags",
        "ollama_openai_api",
        "ollama_openai_chat",
        "litellm_route_decides",
        "whitespace_and_slash",
    ],
)
def test_a_base_url_pasted_with_a_request_path_is_read_without_it(provider, pasted, base) -> None:
    descriptor = provider_descriptor(provider)

    assert descriptor.base_endpoint(pasted) == base
    if provider != "litellm" and "?" not in pasted:
        assert _binding(provider, endpoint=pasted).endpoint == base


@pytest.mark.parametrize(
    ("provider", "kind", "model_id", "route"),
    [
        ("openai", "generation", "gpt-5-mini", "openai/gpt-5-mini"),
        ("openai", "embedding", "text-embedding-3-small", "openai/text-embedding-3-small"),
        ("anthropic", "generation", "claude-sonnet-4", "anthropic/claude-sonnet-4"),
        ("openrouter", "generation", "anthropic/claude-sonnet-4", "openrouter/anthropic/claude-sonnet-4"),
        ("openrouter", "embedding", "openai/text-embedding-3-small", "openrouter/openai/text-embedding-3-small"),
        ("ollama", "generation", "llama3.2", "ollama_chat/llama3.2"),
        ("ollama", "embedding", "nomic-embed-text", "ollama/nomic-embed-text"),
        ("openai_compatible", "generation", "local-model", "openai/local-model"),
        ("openai_compatible", "embedding", "BAAI/bge-m3", "openai/BAAI/bge-m3"),
        ("litellm", "generation", "litellm_proxy/claude-sonnet", "litellm_proxy/claude-sonnet"),
        ("litellm", "embedding", " voyage/voyage-3 ", "voyage/voyage-3"),
    ],
)
def test_a_model_id_compiles_to_the_litellm_route_of_its_provider(provider, kind, model_id, route) -> None:
    assert provider_descriptor(provider).compile_route(kind, model_id) == route


@pytest.mark.parametrize(
    ("provider", "kind", "model_id", "code"),
    [
        ("anthropic", "embedding", "claude-sonnet-4", "kind_not_supported"),
        ("openai", "generation", "  ", "model_required"),
        ("litellm", "generation", "bare-model", "model_route_invalid"),
        ("litellm", "generation", "provider/", "model_route_invalid"),
        ("litellm", "embedding", "https://host/model", "model_route_invalid"),
    ],
)
def test_a_model_the_provider_cannot_route_is_refused_by_code(provider, kind, model_id, code) -> None:
    with pytest.raises(ModelConfigError) as refused:
        provider_descriptor(provider).compile_route(kind, model_id)

    assert refused.value.code == code


def test_an_unknown_provider_is_refused_by_code() -> None:
    with pytest.raises(ModelConfigError) as refused:
        provider_descriptor("bedrock")

    assert refused.value.code == "unknown_provider"


@pytest.mark.parametrize(
    ("model", "is_route"),
    [
        ("ollama/nomic-embed-text", True),
        ("openrouter/openai/text-embedding-3-small", True),
        ("litellm_proxy/text-embedding-3-small", True),
        ("text-embedding-3-small", False),
        ("BAAI/bge-m3", False),
        ("ollama/", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_first_segment_litellm_knows_makes_a_name_a_route(model, is_route) -> None:
    assert names_litellm_route(model) is is_route


def test_a_provider_registered_with_litellm_makes_a_name_a_route(monkeypatch) -> None:
    monkeypatch.setattr(
        "litellm.custom_provider_map",
        [{"provider": "house-models", "custom_handler": object()}],
    )

    assert names_litellm_route("house-models/embed-1") is True


def test_an_api_key_binding_compiles_with_its_key_and_default_endpoint() -> None:
    binding = _binding("anthropic", model_id="claude-sonnet-4", secret_ref="secret:one")

    invocation = compile_binding(binding, "anthropic-key")

    assert invocation == ResolvedInvocation(
        route="anthropic/claude-sonnet-4",
        api_base="https://api.anthropic.com",
        api_key="anthropic-key",
        provider="anthropic",
    )


def test_an_api_key_binding_without_a_credential_does_not_compile() -> None:
    binding = _binding("openai", secret_ref="secret:one")

    with pytest.raises(ModelConfigError) as refused:
        compile_binding(binding, None)

    assert refused.value.code == "credential_unavailable"


def test_the_none_strategy_sends_an_explicit_keyless_value_and_ignores_any_credential() -> None:
    binding = _binding("ollama", kind="embedding", model_id="nomic-embed-text")

    invocation = compile_binding(binding, "a-key-that-must-not-be-sent")

    assert binding.auth_strategy == "none"
    assert invocation.api_key == KEYLESS_API_KEY
    assert invocation.request_kwargs() == {
        "api_base": "http://localhost:11434/v1",
        "api_key": KEYLESS_API_KEY,
    }


@pytest.mark.parametrize(
    ("endpoint", "api_base"),
    [
        (None, "http://localhost:11434/v1"),
        ("https://ollama.example.test/v1/embeddings", "https://ollama.example.test/v1"),
        ("https://gateway.example.test/ollama/", "https://gateway.example.test/ollama/v1"),
    ],
)
def test_an_ollama_embedding_is_sent_to_its_openai_compatible_api_under_the_route_that_names_it(
    endpoint, api_base
) -> None:
    binding = _binding(
        "ollama",
        kind="embedding",
        model_id="nomic-embed-text",
        endpoint=endpoint,
        auth_strategy="api_key",
        secret_ref="secret:one",
        parameters={"dimensions": 256},
    )

    invocation = compile_binding(binding, "ollama-key")

    # The model, and the vector space its Embedding Profile names, are the binding's.
    assert embedding_identity(binding) == EmbeddingIdentity(
        provider="ollama",
        route="ollama/nomic-embed-text",
        endpoint=api_base.removesuffix("/v1"),
        model_id="nomic-embed-text",
        requested_dimensions=256,
    )
    assert invocation == ResolvedInvocation(
        route="openai/nomic-embed-text",
        api_base=api_base,
        api_key="ollama-key",
        extra_params={"dimensions": 256},
        provider="ollama",
    )
    assert invocation.request_kwargs()["allowed_openai_params"] == ["dimensions"]


def test_an_ollama_generation_model_is_sent_on_its_own_route() -> None:
    invocation = compile_binding(_binding("ollama", model_id="qwen3"), None)

    assert (invocation.route, invocation.api_base) == ("ollama_chat/qwen3", "http://localhost:11434")


def test_the_environment_strategy_passes_no_key_so_the_sdk_resolves_it() -> None:
    binding = _binding("litellm", model_id="litellm_proxy/claude-sonnet", auth_strategy="environment")

    invocation = compile_binding(binding, None)

    assert invocation == ResolvedInvocation(route="litellm_proxy/claude-sonnet", provider="litellm")
    assert invocation.request_kwargs() == {}


def test_options_and_parameters_reach_the_sdk_as_extra_parameters() -> None:
    binding = _binding(
        "openai",
        kind="embedding",
        model_id="text-embedding-3-large",
        options={"organization": "org-team"},
        parameters={"dimensions": 512},
        secret_ref="secret:one",
    )

    invocation = compile_binding(binding, "openai-key")

    assert invocation.request_kwargs() == {
        "api_base": "https://api.openai.com/v1",
        "api_key": "openai-key",
        "organization": "org-team",
        "dimensions": 512,
        "allowed_openai_params": ["dimensions"],
    }


@pytest.mark.parametrize(
    ("provider", "model_id", "parameters", "allowed"),
    [
        ("openai_compatible", "BAAI/bge-m3", {"dimensions": 256}, True),
        ("litellm", "openai/custom-embed", {"dimensions": 256}, True),
        ("openai_compatible", "BAAI/bge-m3", {}, False),
        ("openrouter", "vendor/embed", {"dimensions": 256}, False),
        ("litellm", "voyage/voyage-3", {"dimensions": 256}, False),
    ],
    ids=["compatible", "openai_route", "no_dimension", "openrouter", "another_route"],
)
def test_a_set_dimension_is_named_for_the_sdk_to_send_on_the_openai_route_only(
    provider, model_id, parameters, allowed
) -> None:
    binding = _binding(
        provider,
        kind="embedding",
        model_id=model_id,
        endpoint="https://embed.example.test/v1",
        parameters=parameters,
        secret_ref="secret:one",
    )

    kwargs = compile_binding(binding, "endpoint-key").request_kwargs()

    assert ("allowed_openai_params" in kwargs) is allowed
    if allowed:
        assert kwargs["allowed_openai_params"] == ["dimensions"]


@pytest.mark.parametrize(
    "extra_params",
    [
        {"drop_params": True},
        {"api_base": "https://elsewhere.example.test"},
        {"dimensions": "512"},
        {"dimensions": True},
    ],
)
def test_an_invocation_carries_only_allowlisted_extra_parameters(extra_params) -> None:
    with pytest.raises(ModelConfigError) as refused:
        ResolvedInvocation(route="openai/model-a", extra_params=extra_params)

    assert refused.value.code == "parameter_not_supported"


def test_an_invocation_is_frozen_and_does_not_show_its_key() -> None:
    invocation = ResolvedInvocation(route="openai/model-a", api_key="secret-key", extra_params={"dimensions": 8})

    with pytest.raises(AttributeError):
        invocation.route = "openai/model-b"
    with pytest.raises(TypeError):
        invocation.extra_params["dimensions"] = 16
    assert "secret-key" not in repr(invocation)


@pytest.mark.parametrize(
    ("provider", "kind", "settings", "code"),
    [
        ("openai_compatible", "generation", {}, "endpoint_required"),
        ("openai", "generation", {"endpoint": "api.openai.com/v1"}, "endpoint_invalid"),
        ("anthropic", "generation", {"auth_strategy": "none"}, "auth_strategy_not_supported"),
        ("ollama", "generation", {"auth_strategy": "environment"}, "auth_strategy_not_supported"),
        ("anthropic", "generation", {"options": {"organization": "org"}}, "option_not_supported"),
        ("openai", "generation", {"options": {"organization": 7}}, "option_invalid"),
        ("openai", "generation", {"parameters": {"dimensions": 256}}, "parameter_not_supported"),
        ("openai", "embedding", {"parameters": {"dimensions": 0}}, "parameter_invalid"),
        ("openai", "embedding", {"parameters": {"dimensions": "256"}}, "parameter_invalid"),
        ("openai", "embedding", {"parameters": {"temperature": 1}}, "parameter_not_supported"),
    ],
)
def test_settings_a_provider_does_not_accept_are_refused_by_code(provider, kind, settings, code) -> None:
    with pytest.raises(ModelConfigError) as refused:
        _binding(provider, kind=kind, **settings)

    assert refused.value.code == code


def test_a_binding_takes_the_default_endpoint_and_the_first_auth_strategy() -> None:
    binding = _binding("openrouter", model_id="anthropic/claude-sonnet-4")

    assert binding.endpoint == "https://openrouter.ai/api/v1"
    assert binding.origin == "https://openrouter.ai"
    assert binding.auth_strategy == "api_key"
    assert binding.state == "not_validated"


def test_a_binding_keeps_no_key_reference_for_a_strategy_that_uses_none() -> None:
    binding = _binding("ollama", secret_ref="secret:one")

    assert binding.secret_ref is None


@pytest.mark.parametrize(
    ("endpoint", "origin"),
    [
        ("https://Proxy.Example.Test:443/v1/", "https://proxy.example.test"),
        ("http://localhost:8080/openai", "http://localhost:8080"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_a_binding_is_bound_to_the_normalized_origin_of_its_endpoint(endpoint, origin) -> None:
    binding = _binding("openai_compatible", endpoint=endpoint)

    assert binding.origin == origin
    assert not (binding.endpoint or "").endswith("/")


def test_a_route_without_an_endpoint_is_bound_to_the_provider_of_its_route() -> None:
    assert _binding("litellm", model_id="groq/llama-3").origin == "provider:groq"
    assert _binding("litellm", model_id="voyage/voyage-3", kind="embedding").origin == "provider:voyage"


@pytest.mark.parametrize("model_id", ["litellm_proxy/claude-sonnet", "litellm_proxy/gpt-5-mini"])
def test_a_litellm_route_through_a_gateway_is_asked_for_nothing_about_reasoning(model_id) -> None:
    """A gateway may pass a request on to any provider, so no reasoning parameter is safe to send it."""
    descriptor = provider_descriptor("litellm")

    assert descriptor.reasoning_off(descriptor.compile_route("generation", model_id)) == {}


def test_reasoning_off_parameters_are_a_fresh_value_for_every_request() -> None:
    """LiteLLM merges into ``extra_body`` in place, so no request may share it with the next."""
    descriptor = provider_descriptor("openrouter")
    first = descriptor.reasoning_off("openrouter/anthropic/claude-haiku-5.5")
    first["extra_body"]["reasoning"]["enabled"] = True

    assert descriptor.reasoning_off("openrouter/anthropic/claude-haiku-5.5") == {
        "extra_body": {"reasoning": {"enabled": False}}
    }


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter", "ollama", "openai_compatible", "litellm"])
def test_a_compiled_invocation_names_the_descriptor_it_was_compiled_from(provider) -> None:
    model_id = "litellm_proxy/claude-sonnet" if provider == "litellm" else "model-a"
    endpoint = "https://models.example.test/v1" if provider == "openai_compatible" else None
    binding = _binding(provider, model_id=model_id, endpoint=endpoint, secret_ref="secret:one")

    assert compile_binding(binding, "key").provider == provider
