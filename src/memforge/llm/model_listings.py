"""What a provider's model listing declares about the models it lists.

``ModelListing`` is the one contract every provider that lists its models
implements, in a class of its own: where to ask (``request_urls``), how to ask
(``request_headers``), and how the answer reads (``entries`` and
``declared_model``, into a ``DeclaredModel``). Providers keep the same facts
in different fields, or give none, so each implementation reads only the
fields its provider writes. Everything else, the request with its origin and
redirect rules, Settings and the runtime, uses the contract alone and never
asks which provider it is.

The contract grows in two directions. A new provider implements it and states
it on its descriptor, with no change to shared code. A new fact is a
``DeclaredModel`` field that declares nothing by default, with a ``gives_*``
flag: each implementation reads it from its provider's own field or leaves it
undeclared, and code that uses the fact reads it from ``DeclaredModel`` only.

A declaration is what the provider observes about a model. It guides the
choice of a model and the requests made to it; a capability check, not the
listing, establishes what a configured model can do.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from memforge.llm.providers import ModelKind

__all__ = [
    "AnthropicListing",
    "DeclaredModel",
    "KindSignal",
    "ModelListing",
    "OllamaListing",
    "OpenAiCompatibleListing",
    "OpenAiListing",
    "OpenRouterListing",
]

# How a model listing tells the models of one kind from the other.
# ``listing``: each kind has a listing of its own, which holds only models of
# that kind. ``entry``: one listing holds every model, and each entry says, by
# its declared capabilities, its family or its name, which kind it is.
# ``none``: the listing does not say, so every listed model is offered.
KindSignal = Literal["listing", "entry", "none"]

# Ollama families of models that only embed text; Ollama names a model's family
# by its architecture.
_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "xlm-roberta"})
# A word in a model id that names a model which does not answer in text: an
# image, audio or moderation model of an OpenAI-style listing.
_NON_TEXT_MODEL = re.compile(
    r"(?<![a-z0-9])(dall-e|image|whisper|tts|transcribe|audio|realtime|moderation|sora)(?![a-z])"
)


@dataclass(frozen=True)
class DeclaredModel:
    """One listed model and what its entry declares; ``None`` where the entry declares nothing."""

    id: str
    label: str | None = None
    # The kinds the entry declares it serves; ``None`` when the listing, not
    # the entry, decides the kind.
    kinds: frozenset[ModelKind] | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    # Whether the model answers in a JSON schema the request gives.
    native_schema: bool | None = None
    # Whether the generation model takes images as input.
    image_input: bool | None = None


class ModelListing(ABC):
    """Where an endpoint lists its models, and how one entry of the listing reads."""

    # The path, after the base URL, that lists the models of each served kind.
    paths: ClassVar[Mapping[ModelKind, str]]
    kind_signal: ClassVar[KindSignal]
    # The field of the listing's answer that holds its entries.
    entries_field: ClassVar[str] = "data"
    # Whether a base URL entered without an API version is also asked under
    # ``/v1``, where OpenAI-style servers keep their API.
    tries_v1: ClassVar[bool] = True
    # Whether an entry can give the token limits of its model: the context
    # window and the largest output.
    gives_token_limits: ClassVar[bool] = False
    # Whether an entry declares if its model answers in a JSON schema the
    # request gives.
    gives_native_schema: ClassVar[bool] = False
    # Whether an entry declares if its generation model takes images as input.
    gives_image_input: ClassVar[bool] = False

    @abstractmethod
    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        """Read one entry listed for ``kind``; ``None`` for an entry that names no model."""

    def request_urls(self, endpoint: str, kind: ModelKind) -> list[str]:
        """The addresses that may answer with the listing of ``kind``, in the order they are asked."""
        root = endpoint.strip().rstrip("/")
        path = self.paths[kind]
        urls = [f"{root}{path}"]
        if self.tries_v1 and not urlsplit(root).path.rstrip("/").endswith("/v1"):
            urls.append(f"{root}/v1{path}")
        return list(dict.fromkeys(urls))

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        """The headers of a listing request: the API key, when there is one, as a bearer token."""
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @property
    def kind_filtered(self) -> bool:
        """Whether the listing tells the models of one kind from the other."""
        return self.kind_signal != "none"

    def entries(self, payload: Any) -> list[Any]:
        """The entries of a listing's answer."""
        entries = payload.get(self.entries_field) if isinstance(payload, dict) else None
        return entries if isinstance(entries, list) else []

    def declared_models(self, payload: Any, kind: ModelKind) -> tuple[list[DeclaredModel], int]:
        """Read the models of ``kind`` from a listing, with the number of models it listed.

        A listing of one kind holds only models of that kind, and a listing
        that does not tell the kinds apart gives every model it lists.
        """
        seen: set[str] = set()
        listed = 0
        models: list[DeclaredModel] = []
        for entry in self.entries(payload):
            model = self.declared_model(entry, kind)
            if model is None or model.id in seen:
                continue
            seen.add(model.id)
            listed += 1
            if self.kind_signal != "entry" or (model.kinds is not None and kind in model.kinds):
                models.append(model)
        return models, listed


class OpenAiListing(ModelListing):
    """OpenAI's ``/models``: every model in one listing, each entry an ``id`` and its owner."""

    paths = {"generation": "/models", "embedding": "/models"}
    kind_signal = "entry"

    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        model_id = _text(entry, "id")
        return None if model_id is None else DeclaredModel(id=model_id, kinds=_kinds_by_name(model_id))


class AnthropicListing(ModelListing):
    """Anthropic's ``/v1/models``: generation models only, each with its ``display_name``.

    ``max_input_tokens`` is the largest input, which stands for the context
    window as it does in LiteLLM's model map, and ``max_tokens`` the largest
    output. ``capabilities.image_input.supported`` says whether the model
    takes image content (https://platform.claude.com/docs/en/api/models/list).
    """

    paths = {"generation": "/models"}
    kind_signal = "listing"
    gives_token_limits = True
    gives_image_input = True

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        """Anthropic takes the API key as ``x-api-key``, with the API version it answers in."""
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        model_id = _text(entry, "id")
        if model_id is None:
            return None
        return DeclaredModel(
            id=model_id,
            label=_label(entry, "display_name", model_id),
            context_window_tokens=_token_count(entry.get("max_input_tokens")),
            max_output_tokens=_token_count(entry.get("max_tokens")),
            image_input=_supported(entry.get("capabilities"), "image_input"),
        )


class OpenRouterListing(ModelListing):
    """OpenRouter's ``/models`` and ``/embeddings/models``, one listing per kind.

    An entry names its model by ``id`` and ``name``. The context window is
    ``context_length`` and the largest output
    ``top_provider.max_completion_tokens``. A model whose
    ``supported_parameters`` name ``structured_outputs`` takes a
    ``json_schema`` response format
    (https://openrouter.ai/docs/guides/features/structured-outputs), and one
    whose ``architecture.input_modalities`` name ``image`` takes images.
    """

    paths = {"generation": "/models", "embedding": "/embeddings/models"}
    kind_signal = "listing"
    gives_token_limits = True
    gives_native_schema = True
    gives_image_input = True

    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        model_id = _text(entry, "id")
        if model_id is None:
            return None
        top_provider = entry.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        return DeclaredModel(
            id=model_id,
            label=_label(entry, "name", model_id),
            context_window_tokens=_token_count(entry.get("context_length")),
            max_output_tokens=_token_count(top_provider.get("max_completion_tokens")),
            native_schema=_declared_native_schema(entry) if kind == "generation" else None,
            image_input=_input_modality(entry, "image") if kind == "generation" else None,
        )


class OllamaListing(ModelListing):
    """Ollama's ``/api/tags``: every local model under ``models``, each named by its tag.

    An entry's ``capabilities`` name its kinds; without them its family, or
    its name, does. ``vision`` among them says a generation model takes
    images; an Ollama that lists no capabilities declares nothing about it.
    """

    paths = {"generation": "/api/tags", "embedding": "/api/tags"}
    kind_signal = "entry"
    entries_field = "models"
    tries_v1 = False
    gives_image_input = True

    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        model_id = _text(entry, "name")
        if model_id is None:
            return None
        capabilities = entry.get("capabilities")
        return DeclaredModel(
            id=model_id,
            kinds=_ollama_kinds(entry, model_id),
            image_input=(
                "vision" in capabilities if kind == "generation" and isinstance(capabilities, list) else None
            ),
        )


class OpenAiCompatibleListing(ModelListing):
    """An OpenAI-compatible ``/models``, which does not say which kind a model is.

    Any server may answer here, so this listing takes the shapes such servers
    write: the entries as a list, or under ``data`` or ``models``; an entry as
    an id alone, or an object naming its model by ``id``, ``name`` or
    ``model``. vLLM and gateways give a model's context window in its entry as
    ``context_length``, ``context_window`` or ``max_model_len``, and its
    largest output as ``top_provider.max_completion_tokens`` or
    ``max_completion_tokens``.
    """

    paths = {"generation": "/models", "embedding": "/models"}
    kind_signal = "none"
    gives_token_limits = True

    def entries(self, payload: Any) -> list[Any]:
        if isinstance(payload, dict):
            entries = payload.get("data")
            if entries is None:
                entries = payload.get("models")
        else:
            entries = payload
        return entries if isinstance(entries, list) else []

    def declared_model(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        if isinstance(entry, str):
            return DeclaredModel(id=entry) if entry else None
        model_id = _text(entry, "id") or _text(entry, "name") or _text(entry, "model")
        if model_id is None:
            return None
        top_provider = entry.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        return DeclaredModel(
            id=model_id,
            label=_label(
                entry, next((key for key in ("display_name", "label", "name") if entry.get(key)), "name"), model_id
            ),
            context_window_tokens=_first_token_count(
                entry.get("context_length"),
                entry.get("context_window"),
                entry.get("max_model_len"),
            ),
            max_output_tokens=_first_token_count(
                top_provider.get("max_completion_tokens"),
                entry.get("max_completion_tokens"),
            ),
        )


def _text(entry: Any, field: str) -> str | None:
    """The text of one field of an entry object; ``None`` when it is missing or empty."""
    value = entry.get(field) if isinstance(entry, dict) else None
    return str(value) if value is not None and str(value) else None


def _label(entry: Mapping[str, Any], field: str, model_id: str) -> str | None:
    """The display name in ``field`` of an entry, when it says more than the model id."""
    label = _text(entry, field)
    return None if label == model_id else label


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _first_token_count(*values: Any) -> int | None:
    for value in values:
        count = _token_count(value)
        if count is not None:
            return count
    return None


def _declared_native_schema(entry: Mapping[str, Any]) -> bool | None:
    """Whether an entry's ``supported_parameters`` name ``structured_outputs``; ``None`` without that list."""
    supported = entry.get("supported_parameters")
    if not isinstance(supported, list):
        return None
    return "structured_outputs" in supported


def _supported(capabilities: Any, name: str) -> bool | None:
    """Anthropic's ``capabilities.<name>.supported``; ``None`` when the entry does not say."""
    capability = capabilities.get(name) if isinstance(capabilities, dict) else None
    supported = capability.get("supported") if isinstance(capability, dict) else None
    return supported if isinstance(supported, bool) else None


def _input_modality(entry: Mapping[str, Any], modality: str) -> bool | None:
    """Whether OpenRouter's ``architecture.input_modalities`` name ``modality``; ``None`` without that list."""
    architecture = entry.get("architecture")
    modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    if not isinstance(modalities, list):
        return None
    return modality in modalities


def _ollama_kinds(entry: Mapping[str, Any], model_id: str) -> frozenset[ModelKind]:
    """The kinds an Ollama entry serves: by its ``capabilities``, else its family, else its name.

    ``completion`` is a generation model and ``embedding`` an embedding model;
    an Ollama family of embedding models makes it an embedding model.
    """
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, list):
        kinds: set[ModelKind] = set()
        if "completion" in capabilities:
            kinds.add("generation")
        if "embedding" in capabilities:
            kinds.add("embedding")
        return frozenset(kinds)
    details = entry.get("details")
    if isinstance(details, dict):
        families = details.get("families")
        named = {details.get("family"), *(families if isinstance(families, list) else ())}
        if named & _EMBEDDING_FAMILIES:
            return frozenset({"embedding"})
    return _kinds_by_name(model_id)


def _kinds_by_name(model_id: str) -> frozenset[ModelKind]:
    """The kinds a model id names: ``embed`` an embedding model, an image, audio or moderation word neither."""
    lowered = model_id.lower()
    if "embed" in lowered:
        return frozenset({"embedding"})
    if _NON_TEXT_MODEL.search(lowered):
        return frozenset()
    return frozenset({"generation"})
