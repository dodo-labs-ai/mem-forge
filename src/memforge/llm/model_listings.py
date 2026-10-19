"""What a provider's model listing declares about the models it lists.

Every provider that lists its models has a ``ModelListing`` of its own: where
the listing is, and how one of its entries reads as a ``DeclaredModel``. The
fields an entry carries are the provider's own, so each listing reads only the
fields its provider writes, and a provider added to ``PROVIDER_DESCRIPTORS``
states its listing, or that it has none, as part of its descriptor.

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


class ModelListing(ABC):
    """Where an endpoint lists its models, and how one entry of the listing reads."""

    # The path, after the base URL, that lists the models of each served kind.
    paths: ClassVar[Mapping[ModelKind, str]]
    kind_signal: ClassVar[KindSignal]
    # Whether a base URL entered without an API version is also asked under
    # ``/v1``, where OpenAI-style servers keep their API.
    tries_v1: ClassVar[bool] = True
    # Whether an entry can give the token limits of its model: the context
    # window and the largest output.
    gives_token_limits: ClassVar[bool] = False
    # Whether an entry declares if its model answers in a JSON schema the
    # request gives.
    gives_native_schema: ClassVar[bool] = False

    @abstractmethod
    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        """Read one entry, listed for ``kind``, of the model ``model_id``."""

    @property
    def kind_filtered(self) -> bool:
        """Whether the listing tells the models of one kind from the other."""
        return self.kind_signal != "none"

    def declared_models(self, payload: Any, kind: ModelKind) -> tuple[list[DeclaredModel], int]:
        """Read the models of ``kind`` from a listing, with the number of models it listed.

        A listing of one kind holds only models of that kind, and a listing
        that does not tell the kinds apart gives every model it lists. An entry
        is an object with an id, or the id alone.
        """
        seen: set[str] = set()
        listed = 0
        models: list[DeclaredModel] = []
        for entry in _listed_entries(payload):
            model = self._declared_entry(entry, kind)
            if model is None or model.id in seen:
                continue
            seen.add(model.id)
            listed += 1
            if self.kind_signal != "entry" or (model.kinds is not None and kind in model.kinds):
                models.append(model)
        return models, listed

    def _declared_entry(self, entry: Any, kind: ModelKind) -> DeclaredModel | None:
        if isinstance(entry, str):
            return self._kinds_of_bare_id(entry) if entry else None
        if not isinstance(entry, dict):
            return None
        raw_id = entry.get("id") or entry.get("name") or entry.get("model")
        if raw_id is None or not str(raw_id):
            return None
        return self.declared_model(entry, str(raw_id), kind)

    def _kinds_of_bare_id(self, model_id: str) -> DeclaredModel:
        return DeclaredModel(id=model_id, kinds=_kinds_by_name(model_id) if self.kind_signal == "entry" else None)


class OpenAiListing(ModelListing):
    """OpenAI's ``/models``: every model in one listing, each entry an id and its owner."""

    paths = {"generation": "/models", "embedding": "/models"}
    kind_signal = "entry"

    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        return DeclaredModel(id=model_id, label=_label(entry, model_id), kinds=_kinds_by_name(model_id))


class AnthropicListing(ModelListing):
    """Anthropic's ``/v1/models``: generation models only."""

    paths = {"generation": "/models"}
    kind_signal = "listing"

    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        return DeclaredModel(id=model_id, label=_label(entry, model_id))


class OpenRouterListing(ModelListing):
    """OpenRouter's ``/models`` and ``/embeddings/models``, one listing per kind.

    The context window is ``context_length`` and the largest output
    ``top_provider.max_completion_tokens``. A model whose
    ``supported_parameters`` name ``structured_outputs`` takes a
    ``json_schema`` response format
    (https://openrouter.ai/docs/guides/features/structured-outputs).
    """

    paths = {"generation": "/models", "embedding": "/embeddings/models"}
    kind_signal = "listing"
    gives_token_limits = True
    gives_native_schema = True

    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        top_provider = entry.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        return DeclaredModel(
            id=model_id,
            label=_label(entry, model_id),
            context_window_tokens=_token_count(entry.get("context_length")),
            max_output_tokens=_token_count(top_provider.get("max_completion_tokens")),
            native_schema=_declared_native_schema(entry) if kind == "generation" else None,
        )


class OllamaListing(ModelListing):
    """Ollama's ``/api/tags``: every local model in one listing.

    An entry's ``capabilities`` name its kinds; without them its family, or
    its name, does.
    """

    paths = {"generation": "/api/tags", "embedding": "/api/tags"}
    kind_signal = "entry"
    tries_v1 = False

    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        return DeclaredModel(id=model_id, label=_label(entry, model_id), kinds=_ollama_kinds(entry, model_id))


class OpenAiCompatibleListing(ModelListing):
    """An OpenAI-compatible ``/models``, which does not say which kind a model is.

    Any server may answer here. vLLM and gateways give a model's context window
    in its entry as ``context_length``, ``context_window`` or
    ``max_model_len``, and its largest output as
    ``top_provider.max_completion_tokens`` or ``max_completion_tokens``.
    """

    paths = {"generation": "/models", "embedding": "/models"}
    kind_signal = "none"
    gives_token_limits = True

    def declared_model(self, entry: Mapping[str, Any], model_id: str, kind: ModelKind) -> DeclaredModel:
        top_provider = entry.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        return DeclaredModel(
            id=model_id,
            label=_label(entry, model_id),
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


def _listed_entries(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        raw_items = payload.get("data")
        if raw_items is None:
            raw_items = payload.get("models")
    else:
        raw_items = payload
    return raw_items if isinstance(raw_items, list) else []


def _label(entry: Mapping[str, Any], model_id: str) -> str | None:
    raw_label = entry.get("display_name") or entry.get("label") or entry.get("name")
    return None if raw_label is None or str(raw_label) == model_id else str(raw_label)


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
