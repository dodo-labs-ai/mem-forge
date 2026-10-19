"""Every provider states what its model listing declares, read from the provider's own entry shape."""

import pytest

from memforge.llm.model_listings import DeclaredModel, ModelListing
from memforge.llm.providers import PROVIDER_DESCRIPTORS, ProviderDescriptor

# One listing per provider, in the shape the provider answers with, and what
# it declares for each kind. A provider added to PROVIDER_DESCRIPTORS with a
# listing needs an entry here.
LISTINGS = {
    "openai": (
        {
            "object": "list",
            "data": [
                {"id": "gpt-5-mini", "object": "model", "created": 1754425777, "owned_by": "system"},
                {"id": "text-embedding-3-small", "object": "model", "created": 1705948997, "owned_by": "system"},
                {"id": "dall-e-3", "object": "model", "created": 1698785189, "owned_by": "system"},
            ],
        },
        {
            "generation": [DeclaredModel(id="gpt-5-mini", kinds=frozenset({"generation"}))],
            "embedding": [DeclaredModel(id="text-embedding-3-small", kinds=frozenset({"embedding"}))],
        },
    ),
    "anthropic": (
        {
            "data": [
                {
                    "type": "model",
                    "id": "claude-haiku-5-5",
                    "display_name": "Claude Haiku 5.5",
                    "created_at": "2026-09-21T00:00:00Z",
                    "max_input_tokens": 200000,
                    "max_tokens": 64000,
                }
            ],
            "has_more": False,
        },
        {
            "generation": [
                DeclaredModel(
                    id="claude-haiku-5-5",
                    label="Claude Haiku 5.5",
                    context_window_tokens=200000,
                    max_output_tokens=64000,
                )
            ]
        },
    ),
    "openrouter": (
        {
            "data": [
                {
                    "id": "anthropic/claude-haiku-5.5",
                    "name": "Anthropic: Claude Haiku 5.5",
                    "context_length": 200000,
                    "top_provider": {"max_completion_tokens": 64000},
                    "supported_parameters": ["max_tokens", "structured_outputs", "tools"],
                },
                {
                    "id": "qwen/qwen3-32b",
                    "name": "Qwen: Qwen3 32B",
                    "context_length": 40960,
                    "top_provider": {"max_completion_tokens": None},
                    "supported_parameters": ["max_tokens", "tools"],
                },
            ]
        },
        {
            "generation": [
                DeclaredModel(
                    id="anthropic/claude-haiku-5.5",
                    label="Anthropic: Claude Haiku 5.5",
                    context_window_tokens=200000,
                    max_output_tokens=64000,
                    native_schema=True,
                ),
                DeclaredModel(
                    id="qwen/qwen3-32b",
                    label="Qwen: Qwen3 32B",
                    context_window_tokens=40960,
                    native_schema=False,
                ),
            ],
            # The embedding listing has the same entry shape; schema support is a generation matter.
            "embedding": [
                DeclaredModel(
                    id="anthropic/claude-haiku-5.5",
                    label="Anthropic: Claude Haiku 5.5",
                    context_window_tokens=200000,
                    max_output_tokens=64000,
                ),
                DeclaredModel(id="qwen/qwen3-32b", label="Qwen: Qwen3 32B", context_window_tokens=40960),
            ],
        },
    ),
    "ollama": (
        {
            "models": [
                {
                    "name": "qwen3:0.6b",
                    "model": "qwen3:0.6b",
                    "details": {"family": "qwen3", "families": ["qwen3"]},
                    "capabilities": ["completion", "tools", "thinking"],
                },
                {
                    "name": "nomic-embed-text:latest",
                    "model": "nomic-embed-text:latest",
                    "details": {"family": "nomic-bert", "families": ["nomic-bert"]},
                },
            ]
        },
        {
            "generation": [DeclaredModel(id="qwen3:0.6b", kinds=frozenset({"generation"}))],
            "embedding": [DeclaredModel(id="nomic-embed-text:latest", kinds=frozenset({"embedding"}))],
        },
    ),
    "openai_compatible": (
        {"object": "list", "data": [{"id": "Qwen/Qwen3-8B", "object": "model", "max_model_len": 32768}]},
        {
            "generation": [DeclaredModel(id="Qwen/Qwen3-8B", context_window_tokens=32768)],
            "embedding": [DeclaredModel(id="Qwen/Qwen3-8B", context_window_tokens=32768)],
        },
    ),
}

# Providers that have no one listing to ask.
UNLISTED = {"litellm"}


def test_every_provider_states_its_listing_and_each_listing_has_a_contract_case() -> None:
    for descriptor in PROVIDER_DESCRIPTORS:
        if descriptor.listing is None:
            assert descriptor.id in UNLISTED, f"{descriptor.id} states no listing"
        else:
            assert isinstance(descriptor.listing, ModelListing)
            assert descriptor.id in LISTINGS, f"{descriptor.id} has no listing contract case"
            assert set(descriptor.listing.paths) == set(descriptor.kinds)
    assert set(LISTINGS) | UNLISTED == {descriptor.id for descriptor in PROVIDER_DESCRIPTORS}


def test_a_descriptor_cannot_leave_its_listing_unstated() -> None:
    with pytest.raises(TypeError, match="listing"):
        ProviderDescriptor(id="new", label="New", kinds=("generation",), auth_strategies=("api_key",))


@pytest.mark.parametrize(
    ("provider_id", "kind"),
    [(provider_id, kind) for provider_id, (_, expected) in LISTINGS.items() for kind in expected],
)
def test_a_listing_declares_what_its_provider_entries_say(provider_id, kind) -> None:
    payload, expected = LISTINGS[provider_id]
    listing = next(d for d in PROVIDER_DESCRIPTORS if d.id == provider_id).listing

    models, listed = listing.declared_models(payload, kind)

    assert models == expected[kind]
    assert listed == len(payload.get("data") or payload.get("models"))


@pytest.mark.parametrize("provider_id", list(LISTINGS))
def test_a_listing_says_it_gives_exactly_what_its_entries_declare(provider_id) -> None:
    _, expected = LISTINGS[provider_id]
    listing = next(d for d in PROVIDER_DESCRIPTORS if d.id == provider_id).listing
    models = [model for kind_models in expected.values() for model in kind_models]

    assert listing.gives_token_limits == any(
        model.context_window_tokens is not None or model.max_output_tokens is not None for model in models
    )
    assert listing.gives_native_schema == any(model.native_schema is not None for model in models)


def test_a_listing_reads_only_the_fields_its_provider_writes() -> None:
    listing = next(d for d in PROVIDER_DESCRIPTORS if d.id == "openai").listing
    # Fields other providers write: OpenRouter's limits, Ollama's capabilities, Anthropic's display name.
    entry = {"id": "gpt-5-mini", "context_length": 400000, "capabilities": ["embedding"], "display_name": "GPT"}

    models, _ = listing.declared_models({"data": [entry], "models": [{"name": "qwen3:0.6b"}]}, "generation")

    assert models == [DeclaredModel(id="gpt-5-mini", kinds=frozenset({"generation"}))]


def test_an_openai_compatible_listing_takes_the_shapes_any_server_writes() -> None:
    listing = next(d for d in PROVIDER_DESCRIPTORS if d.id == "openai_compatible").listing

    from_list, listed = listing.declared_models(["local-model", "", {"object": "model"}, "local-model"], "generation")
    from_models, _ = listing.declared_models({"models": [{"model": "served", "label": "Served model"}]}, "generation")

    assert from_list == [DeclaredModel(id="local-model")]
    assert listed == 1
    assert from_models == [DeclaredModel(id="served", label="Served model")]
