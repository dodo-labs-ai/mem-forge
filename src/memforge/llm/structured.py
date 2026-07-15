"""Structured LLM calls with LiteLLM response schemas."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args, get_origin

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class AgentSessionAuthorityDecision(StructuredResponseModel):
    """One semantic authority decision for a candidate agent-session user event."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    is_authoritative: bool
    authority_kind: Literal[
        "durable_user_intent",
        "future_memory_intent",
        "durable_preference",
        "design_decision",
        "rule_or_convention",
        "approval_of_durable_direction",
        "not_authoritative",
    ]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _authority_kind_matches_decision(self):
        if self.is_authoritative and self.authority_kind == "not_authoritative":
            raise ValueError("authoritative decisions require an authoritative authority_kind")
        if not self.is_authoritative and self.authority_kind != "not_authoritative":
            raise ValueError("non-authoritative decisions require authority_kind='not_authoritative'")
        return self


class AgentSessionAuthorityResponse(StructuredResponseModel):
    """Schema returned by agent-session authority classification."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[AgentSessionAuthorityDecision]


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
    source_observation_id: str | None = None
    required_source_observation_ids: list[str] = Field(default_factory=list)


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


class MemoryEquivalenceResponse(StructuredResponseModel):
    """Schema for the semantic proof required before Memory ID reuse."""

    model_config = ConfigDict(extra="ignore")

    equivalent: bool
    reason: str = Field(default="", max_length=1000)


class MemorySupportValidationResponse(StructuredResponseModel):
    """Schema proving whether revised dependencies still support a claim."""

    model_config = ConfigDict(extra="ignore")

    supported: bool
    reason: str = Field(default="", max_length=1000)


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

    async def classify_memory_equivalence(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemoryEquivalenceResponse:
        """Prove whether two claims have identical semantic truth conditions."""

    async def validate_memory_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemorySupportValidationResponse:
        """Prove whether current Primary and Required evidence support a claim."""

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

    async def generate_agent_knowledge_patch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ):
        """Return a private agent-knowledge patch proposal."""

    async def classify_agent_session_evidence_authority(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> AgentSessionAuthorityResponse:
        """Return semantic authority decisions for candidate user evidence."""


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
    if _is_sap_anthropic_model(model_name):
        return True
    try:
        return bool(litellm.supports_response_schema(model=model_name))
    except Exception:
        logger.debug(
            "Unable to determine native response_schema support for model %s",
            model_name,
            exc_info=True,
        )
        return False


def _is_sap_anthropic_model(model_name: str) -> bool:
    """Return true for SAP GenAI Hub Anthropic aliases that accept response_format."""
    return model_name.lower().startswith("sap/anthropic--")


def _strip_json_fences(text: str) -> str:
    """Drop a leading ```/```json fence and trailing ``` if the model adds them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        stripped = stripped[newline + 1 :] if newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _escape_invalid_json_backslashes(text: str) -> str:
    """Preserve literal backslashes that models sometimes emit inside JSON strings."""
    return _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def _validate_structured_json_text(text: str, response_format: type[BaseModel]):
    stripped = _strip_json_fences(text)
    try:
        return response_format.model_validate_json(stripped)
    except ValidationError as exc:
        repaired = _escape_invalid_json_backslashes(stripped)
        if repaired == stripped or "Invalid JSON" not in str(exc):
            raise
        return response_format.model_validate_json(repaired)


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

    async def classify_memory_equivalence(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemoryEquivalenceResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemoryEquivalenceResponse,
            max_tokens=max_tokens,
            model=model,
        )

    async def validate_memory_support(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> MemorySupportValidationResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=MemorySupportValidationResponse,
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

    async def generate_agent_knowledge_patch(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        model: str | None = None,
    ):
        from memforge.agent_knowledge import AgentKnowledgePatchProposal

        return await self._call_schema(
            prompt=prompt,
            response_format=AgentKnowledgePatchProposal,
            max_tokens=max_tokens,
            model=model,
        )

    async def classify_agent_session_evidence_authority(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> AgentSessionAuthorityResponse:
        return await self._call_schema(
            prompt=prompt,
            response_format=AgentSessionAuthorityResponse,
            max_tokens=max_tokens,
            model=model,
            retry_with_json_text=False,
        )

    async def _call_schema(
        self,
        *,
        prompt: str,
        response_format: type[BaseModel],
        max_tokens: int,
        model: str | None = None,
        retry_with_json_text: bool = True,
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
            if not retry_with_json_text:
                raise StructuredLlmError(str(schema_exc)) from schema_exc
            logger.warning(
                "Structured LLM response_schema attempt failed for model %s and schema %s; "
                "retrying with JSON-text schema: %s: %s",
                model_name,
                response_format.__name__,
                type(schema_exc).__name__,
                schema_exc,
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
        messages = [{"role": "user", "content": request_prompt}]
        provider_kwargs: dict[str, Any] = {}
        if model_name.startswith("sap/"):
            # SAP AI Core treats every chat message as a prompt template. Source
            # documents can legitimately contain examples such as {{?input}};
            # pass the complete MemForge prompt as one placeholder value so
            # nested template syntax remains source data rather than SAP input.
            messages = [{"role": "user", "content": "{{?memforge_prompt}}"}]
            provider_kwargs["placeholder_values"] = {"memforge_prompt": request_prompt}
        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            timeout=self.config.timeout_s,
            max_tokens=max_tokens,
            num_retries=self.config.num_retries,
            **litellm_optional_kwargs(
                api_base=self.config.base_url,
                api_key=self.config.api_key,
            ),
            **provider_kwargs,
            **({"response_format": response_format} if native_schema else {}),
        )
        raw_content = _message_content(response)
        if isinstance(raw_content, response_format):
            return raw_content
        if isinstance(raw_content, dict):
            return response_format.model_validate(raw_content)
        return _validate_structured_json_text(str(raw_content), response_format)
