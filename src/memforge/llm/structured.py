"""Structured LLM calls with LiteLLM response schemas."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal, Protocol, get_args, get_origin

import litellm
from pydantic import BaseModel, ConfigDict, Field, model_validator

from memforge.llm.providers import litellm_optional_kwargs

logger = logging.getLogger(__name__)


def _expects_container(annotation: object) -> bool:
    """True when a field annotation resolves to a list/tuple/set or nested model."""
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return True
    if origin is not None:  # Optional[...] / Union[...]
        return any(_expects_container(arg) for arg in get_args(annotation))
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


class StructuredResponseModel(BaseModel):
    """Base for LLM structured-output schemas.

    Some gateway/tool-use responses encode list or nested-object fields as JSON
    strings, for example ``{"memories": "[...]"}``. Normalize those containers
    before field validation so the declared schema still owns correctness.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _decode_stringified_containers(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        decoded: dict[str, object] | None = None
        for name, field in cls.model_fields.items():
            key = field.alias if field.alias and field.alias in data else name
            value = data.get(key)
            if not isinstance(value, str) or not _expects_container(field.annotation):
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                continue
            if decoded is None:
                decoded = dict(data)
            decoded[key] = parsed
        return decoded if decoded is not None else data


class SourceSupportDecision(StructuredResponseModel):
    """One verifier decision for an existing memory candidate."""

    model_config = ConfigDict(extra="ignore")

    memory_id: str = Field(min_length=1)
    supported: bool
    excerpt: str | None = None
    reason: str | None = None


class SourceSupportResponse(StructuredResponseModel):
    """Schema returned by the source-support verifier."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[SourceSupportDecision]


class EnrichmentEntity(StructuredResponseModel):
    """One entity identified during document enrichment."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    type: str = "unknown"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EnrichmentRelationship(StructuredResponseModel):
    """One document relationship identified during enrichment."""

    model_config = ConfigDict(extra="ignore")

    target_title: str = ""
    relation_type: Literal["depends-on", "extends", "supersedes", "references", "related"] = "related"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EnrichmentAliasGroup(StructuredResponseModel):
    """Legacy alias group shape preserved for enrichment compatibility."""

    model_config = ConfigDict(extra="ignore")

    canonical: str = ""
    aliases: list[str] = Field(default_factory=list)
    evidence: str = ""


class EnrichmentResponse(StructuredResponseModel):
    """Schema returned by document enrichment."""

    model_config = ConfigDict(extra="ignore")

    summary: str = "No summary available."
    tags: list[str] = Field(default_factory=list)
    entities: list[EnrichmentEntity] = Field(default_factory=list)
    relationships: list[EnrichmentRelationship] = Field(default_factory=list)
    doc_type: Literal[
        "design-doc",
        "runbook",
        "decision-record",
        "how-to",
        "reference",
        "postmortem",
        "meeting-notes",
        "ticket",
        "discussion",
        "email",
        "unknown",
    ] = "unknown"
    complexity: Literal["low", "medium", "high"] = "medium"
    entity_aliases: list[EnrichmentAliasGroup] = Field(default_factory=list)


class MemoryCandidate(StructuredResponseModel):
    """One memory candidate extracted from a source document."""

    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1)
    memory_type: Literal["fact", "decision", "convention", "procedure"]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    extraction_context: str | None = None
    evidence_quote: str | None = None
    evidence_anchor: Literal["unit", "glossary", "preamble", "outline", "document", "unknown"] = "unknown"


class MemoryExtractionResponse(StructuredResponseModel):
    """Schema returned by memory extraction."""

    model_config = ConfigDict(extra="ignore")

    memories: list[MemoryCandidate]


class ReconciliationDecision(StructuredResponseModel):
    """One same-document memory reconciliation decision."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["ADD", "UPDATE", "SUPERSEDE", "DELETE", "NOOP"]
    index: int | None = None
    memory_id: str | None = None
    updated_content: str | None = None
    reason: str | None = None
    flag_for_review: bool = False


class ReconciliationResponse(StructuredResponseModel):
    """Schema returned by same-document memory reconciliation."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[ReconciliationDecision]


class ContradictionDecision(StructuredResponseModel):
    """One cross-document memory relationship classification."""

    model_config = ConfigDict(extra="ignore")

    pair_index: int
    classification: Literal["contradiction", "temporal", "clarification", "unrelated"]
    reason: str = ""


class ContradictionResponse(StructuredResponseModel):
    """Schema returned by cross-document contradiction detection."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[ContradictionDecision]


class EntityValidationResponse(StructuredResponseModel):
    """Schema returned by entity-match validation."""

    model_config = ConfigDict(extra="ignore")

    same_entity: bool = False
    matched_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None


class QueryEntityDetectionResponse(StructuredResponseModel):
    """Schema returned by query entity detection."""

    model_config = ConfigDict(extra="ignore")

    entity_ids: list[int] = Field(default_factory=list)


class RerankResponse(StructuredResponseModel):
    """Schema returned by memory reranking."""

    model_config = ConfigDict(extra="ignore")

    ranking: list[int] = Field(default_factory=list)


class AgentSessionPackageResponse(StructuredResponseModel):
    """Schema returned by agent-session window package generation."""

    model_config = ConfigDict(extra="ignore")

    result: Literal["package_created", "no_output"] = "no_output"
    title: str | None = None
    summary_markdown: str = ""
    reason: str | None = None


@dataclass(frozen=True)
class StructuredLlmConfig:
    model: str
    base_url: str | None
    api_key: str | None
    timeout_s: float
    # Transparently retry transient gateway/connection blips (e.g. a stale
    # keep-alive connection through an Envoy gateway closed on idle timeout, or a
    # momentary upstream 502). litellm retries 408/429/5xx and connection errors
    # with backoff, so a recoverable hiccup never reaches the caller.
    num_retries: int = 2


class SourceSupportStructuredClient(Protocol):
    async def enrich_document(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> EnrichmentResponse:
        """Return schema-validated document enrichment."""

    async def verify_source_support(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> SourceSupportResponse:
        """Return schema-validated source-support decisions."""

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> MemoryExtractionResponse:
        """Return schema-validated extracted memory candidates."""

    async def reconcile_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> ReconciliationResponse:
        """Return schema-validated same-document reconciliation decisions."""

    async def detect_contradictions(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> ContradictionResponse:
        """Return schema-validated cross-document contradiction decisions."""

    async def validate_entity_match(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        model: str | None = None,
    ) -> EntityValidationResponse:
        """Return schema-validated entity validation."""

    async def detect_query_entities(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        model: str | None = None,
    ) -> QueryEntityDetectionResponse:
        """Return schema-validated query entity ids."""

    async def rerank_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> RerankResponse:
        """Return schema-validated reranking indices."""

    async def generate_agent_session_package(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> AgentSessionPackageResponse:
        """Return a generated package from one agent-session transcript window."""


class StructuredLlmError(RuntimeError):
    """Raised when a required structured LLM call cannot produce valid schema output."""


def _message_content(response) -> object:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise StructuredLlmError(f"missing structured response content: {exc}") from exc
    if content is None:
        raise StructuredLlmError("missing structured response content")
    return content


def litellm_model_name(model: str) -> str:
    """Map existing model names into LiteLLM provider/model notation."""
    if "/" in model:
        return model
    return f"anthropic/{model}"


def _json_text_prompt(prompt: str, response_format: type[BaseModel]) -> str:
    """Append the schema as a text instruction for the no-tool JSON path."""
    schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False)
    return (
        f"{prompt}\n\nReturn ONLY a single JSON object that matches this JSON Schema, "
        f"with no markdown fences and no commentary:\n{schema}"
    )


def _supports_native_response_schema(model_name: str) -> bool:
    try:
        return bool(litellm.supports_response_schema(model=model_name))
    except Exception:
        logger.debug(
            "Unable to determine native response_schema support for model %s",
            model_name,
            exc_info=True,
        )
        return False


def _strip_json_fences(text: str) -> str:
    """Drop a leading ```/```json fence and trailing ``` if the model adds them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        stripped = stripped[newline + 1 :] if newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


class LiteLlmStructuredClient:
    """LiteLLM-backed structured client.

    Native response schemas are the preferred path because gateway aliases can
    enforce them even when LiteLLM's model registry does not recognize the
    alias. If a gateway rejects schema output, the client falls back once to a
    plain JSON prompt with the same schema. Both paths validate against the same
    pydantic model before returning to callers.
    """

    def __init__(self, config: StructuredLlmConfig) -> None:
        self.config = config

    async def enrich_document(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> EnrichmentResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=EnrichmentResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def verify_source_support(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> SourceSupportResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=SourceSupportResponse,
            max_tokens=4096,
            model=model,
        )

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> MemoryExtractionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemoryExtractionResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def reconcile_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> ReconciliationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ReconciliationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def detect_contradictions(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> ContradictionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=ContradictionResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def validate_entity_match(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        model: str | None = None,
    ) -> EntityValidationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=EntityValidationResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def detect_query_entities(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        model: str | None = None,
    ) -> QueryEntityDetectionResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=QueryEntityDetectionResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def rerank_memories(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> RerankResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=RerankResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def generate_agent_session_package(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> AgentSessionPackageResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=AgentSessionPackageResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def _call_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        max_tokens: int,
        model: str | None = None,
    ):
        model_name = litellm_model_name(model or self.config.model)
        if not _supports_native_response_schema(model_name):
            logger.debug(
                "Structured LLM model %s does not advertise native response_schema support; "
                "using JSON-text schema for %s",
                model_name,
                response_format.__name__,
            )
            try:
                return await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                )
            except Exception as exc:
                raise StructuredLlmError(str(exc)) from exc

        try:
            return await self._attempt_schema(
                prompt=prompt,
                response_format=response_format,
                model_name=model_name,
                max_tokens=max_tokens,
                native_schema=True,
            )
        except Exception as schema_exc:
            logger.warning(
                "Structured LLM response_schema attempt failed for model %s and schema %s; "
                "retrying with JSON-text schema",
                model_name,
                response_format.__name__,
                exc_info=True,
            )
            try:
                return await self._attempt_schema(
                    prompt=prompt,
                    response_format=response_format,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    native_schema=False,
                )
            except Exception as exc:
                raise StructuredLlmError(
                    f"{exc} (response_schema attempt failed first: {schema_exc})"
                ) from exc

    async def _attempt_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        model_name: str,
        max_tokens: int,
        native_schema: bool,
    ):
        request_prompt = prompt if native_schema else _json_text_prompt(prompt, response_format)
        response = await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": request_prompt}],
            timeout=self.config.timeout_s,
            max_tokens=max_tokens,
            num_retries=self.config.num_retries,
            **litellm_optional_kwargs(
                api_base=self.config.base_url,
                api_key=self.config.api_key,
            ),
            **({"response_format": response_format} if native_schema else {}),
        )
        raw_content = _message_content(response)
        if isinstance(raw_content, response_format):
            return raw_content
        if isinstance(raw_content, dict):
            return response_format.model_validate(raw_content)
        return response_format.model_validate_json(_strip_json_fences(str(raw_content)))
