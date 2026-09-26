"""Tests for the entity resolution pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from memforge.llm.structured import EntityBatchValidationDecision, EntityBatchValidationResponse, StructuredLlmError
from memforge.memory.entity_resolver import (
    EntityResolutionContext,
    EntityResolutionPolicy,
    EntityResolver,
    validate_alias,
)
from memforge.models import Entity, EntityAlias, canonicalize_entity_name
from memforge.storage.database import Database
from memforge.storage.adapters.protocols import EntityResolutionScope, EntityUpsert
from tests.llm_fixture import FixtureBudgetClient


_SCOPE = EntityResolutionScope(access_context_hash="access-a")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "entity-resolution.db"))
    await database.connect()
    yield database
    await database.close()


# ---------------------------------------------------------------------------
# canonicalize_entity_name
# ---------------------------------------------------------------------------


class TestCanonicalizeEntityName:
    def test_lowercase(self):
        assert canonicalize_entity_name("PostgreSQL") == "postgresql"

    def test_strip_whitespace(self):
        assert canonicalize_entity_name("  pay-api  ") == "pay api"

    def test_collapse_spaces(self):
        assert canonicalize_entity_name("auth   service") == "auth service"

    def test_hyphen_to_space(self):
        """Hyphens normalize to spaces so 'pay-api' and 'pay api' match."""
        assert canonicalize_entity_name("pay-api") == "pay api"

    def test_underscore_to_space(self):
        """Underscores normalize to spaces."""
        assert canonicalize_entity_name("pay_api") == "pay api"

    def test_mixed_separators(self):
        """Hyphens, underscores, and spaces all become single space."""
        assert canonicalize_entity_name("some-name_test  thing") == "some name test thing"

    def test_all_variants_equal(self):
        """pay-api, pay_api, pay api all produce the same canonical form."""
        expected = "pay api"
        assert canonicalize_entity_name("pay-api") == expected
        assert canonicalize_entity_name("pay_api") == expected
        assert canonicalize_entity_name("pay api") == expected
        assert canonicalize_entity_name("  PAY-API  ") == expected

    def test_no_expansion(self):
        """Abbreviations are NOT expanded — alias table handles them at runtime."""
        assert canonicalize_entity_name("PG") == "pg"
        assert canonicalize_entity_name("K8s") == "k8s"
        assert canonicalize_entity_name("JS") == "js"

    def test_empty_string(self):
        assert canonicalize_entity_name("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input produces empty string."""
        assert canonicalize_entity_name("   ") == ""

    def test_dots_preserved(self):
        """Dots pass through unchanged (not hyphens/underscores)."""
        assert canonicalize_entity_name("auth.service") == "auth.service"

    def test_slashes_preserved(self):
        """Slashes pass through unchanged."""
        assert canonicalize_entity_name("auth/service") == "auth/service"

    def test_special_chars_preserved(self):
        """Characters other than hyphens/underscores are not normalized."""
        assert canonicalize_entity_name("auth@v2") == "auth@v2"


# ---------------------------------------------------------------------------
# validate_alias
# ---------------------------------------------------------------------------


class TestValidateAlias:
    def test_token_overlap_plausible(self):
        """'Project Payroll' and 'OnDemand Payroll' share 'payroll' -> plausible."""
        assert validate_alias("Project Payroll", "OnDemand Payroll") is True

    def test_substring_plausible(self):
        """'ODP Runbook' contains 'odp' which is in 'odp' -> substring."""
        assert validate_alias("ODP", "ODP Runbook") is True

    def test_no_resemblance_rejected(self):
        """'pay-api' has no resemblance to 'payment-service' -> rejected."""
        assert validate_alias("pay-api", "payment-service") is False

    def test_sequence_matcher_similar(self):
        """'postgresql' and 'postgresq' have high SequenceMatcher ratio -> plausible."""
        assert validate_alias("postgresq", "postgresql") is True

    def test_completely_different(self):
        """'MSAL' and 'auth-service' -> no overlap -> rejected."""
        assert validate_alias("MSAL", "auth-service") is False

    def test_abbreviation_no_overlap(self):
        """'ODP' and 'OnDemand Payroll' -> no token overlap, no substring -> rejected."""
        # This is correct behavior — abbreviations with no string similarity
        # should be flagged for admin review
        assert validate_alias("ODP", "OnDemand Payroll") is False


# ---------------------------------------------------------------------------
# Batched entity resolution
# ---------------------------------------------------------------------------


class FakeEntityStore:
    def __init__(self, context: EntityResolutionContext) -> None:
        self.context = context
        self.context_calls = 0
        self.created: list[EntityUpsert] = []
        self.aliases: list[EntityAlias] = []

    async def load_entity_resolution_context(self, canonical_names, *, candidate_limit, scope):
        assert candidate_limit == 10
        assert scope == _SCOPE
        self.context_calls += 1
        assert tuple(canonical_names) == tuple(dict.fromkeys(canonical_names))
        return self.context

    async def upsert_entities(self, entities):
        self.created.extend(entities)
        return {item.canonical_name: 100 + index for index, item in enumerate(entities)}

    async def insert_aliases(self, aliases):
        self.aliases.extend(aliases)


def _cases(prompt: str) -> list[dict]:
    return json.loads(prompt.split("Cases:\n", 1)[1].split("\n\nReturn one decision", 1)[0])


def _mentions(*names: str) -> dict[str, tuple[str, ...]]:
    """Mentions whose referencing Memory text does not matter to the test."""

    return dict.fromkeys(names, ())


@dataclass
class EntityClient(FixtureBudgetClient):
    """Answers through ``respond``; ``max_cases`` limits how many mentions one request fits."""

    max_cases: int | None = None

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def request_fits(self, prompt, **kwargs):
        return (self.max_cases is None or len(_cases(prompt)) <= self.max_cases) and super().request_fits(
            prompt, **kwargs,
        )

    async def validate_entity_batch(self, prompt, *, max_tokens, model):
        return await self.call(prompt, max_tokens=max_tokens, model=model)


def _fixed_client(decisions: list[EntityBatchValidationDecision]) -> EntityClient:
    return EntityClient(respond=lambda _prompt: _entity_batch_response(decisions))


def _sequenced_client(responses: list[list[EntityBatchValidationDecision]]) -> EntityClient:
    pending = list(responses)
    return EntityClient(respond=lambda _prompt: _entity_batch_response(pending.pop(0)))


def _match_first_candidate(prompt: str) -> EntityBatchValidationResponse:
    return _entity_batch_response(
        [
            EntityBatchValidationDecision(mention=case["mention"], matched_id=case["candidates"][0]["id"], confidence=0.99)
            for case in _cases(prompt)
        ]
    )


def _entity_batch_response(
    decisions: list[EntityBatchValidationDecision],
) -> EntityBatchValidationResponse:
    return EntityBatchValidationResponse(decisions=decisions)


@pytest.mark.asyncio
async def test_resolve_many_batches_lookup_embedding_and_ambiguity(monkeypatch):
    pay = Entity(id=1, canonical_name="payroll service", display_name="Payroll Service")
    auth = Entity(id=2, canonical_name="authentication service", display_name="Authentication Service")
    known = Entity(id=3, canonical_name="memforge", display_name="MemForge")
    store = FakeEntityStore(
        EntityResolutionContext(
            exact_matches={"memforge": known},
            alias_matches={},
            candidates={
                "pay service": (pay,),
                "auth svc": (auth,),
                "new component": (),
            },
        )
    )
    client = _fixed_client(
        [
            EntityBatchValidationDecision(mention="pay service", matched_id=1, confidence=0.96),
            EntityBatchValidationDecision(mention="auth svc", matched_id=2, confidence=0.97),
        ]
    )
    embedding_batches: list[list[str]] = []

    def fake_embed_texts(texts, *_args):
        embedding_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    result = await resolver.resolve_many(
        {
            "MemForge": ("MemForge stores Memories.",),
            "pay-service": ("pay-service posts payroll runs to the ledger.",),
            "pay service": ("Payroll runs retry through pay service.", "pay-service posts payroll runs to the ledger."),
            "auth_svc": ("auth_svc issues session tokens.",),
            "new component": ("The new component caches tokens.",),
        },
        scope=_SCOPE,
    )

    assert store.context_calls == 1
    assert len(embedding_batches) == 1
    assert client.calls == 1
    assert {case["mention"]: case["memory_texts"] for case in _cases(client.prompts[0])} == {
        "pay service": [
            "pay-service posts payroll runs to the ledger.",
            "Payroll runs retry through pay service.",
        ],
        "auth svc": ["auth_svc issues session tokens."],
    }
    assert result.entity_id("MemForge") == 3
    assert result.entity_id("pay-service") == 1
    assert result.entity_id("auth_svc") == 2
    assert result.entity_id("new component") == 100
    assert store.created == [EntityUpsert("new component", "new component")]
    assert {(item.alias_normalized, item.canonical_id) for item in store.aliases} == {
        ("auth svc", 2),
        ("pay service", 1),
    }
    assert result.metrics.unique_mentions == 4
    assert result.metrics.embedding_batches == 1
    assert result.metrics.structured_llm_calls == 1
    assert result.metrics.validation_retries == 0


@pytest.mark.asyncio
async def test_resolve_many_binds_adjudication_by_mention_not_position(monkeypatch):
    first = Entity(id=1, canonical_name="first service", display_name="First Service")
    second = Entity(id=2, canonical_name="second service", display_name="Second Service")
    store = FakeEntityStore(
        EntityResolutionContext(
            exact_matches={},
            alias_matches={},
            candidates={"first svc": (first,), "second svc": (second,)},
        )
    )
    client = _fixed_client(
        [
            EntityBatchValidationDecision(mention="second svc", matched_id=2, confidence=0.99),
            EntityBatchValidationDecision(mention="first svc", matched_id=1, confidence=0.99),
        ]
    )
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    result = await resolver.resolve_many(_mentions("first svc", "second svc"), scope=_SCOPE)

    assert client.calls == 1
    assert result.entity_id("first svc") == 1
    assert result.entity_id("second svc") == 2
    assert {(item.alias_normalized, item.canonical_id) for item in store.aliases} == {
        ("first svc", 1),
        ("second svc", 2),
    }


@pytest.mark.asyncio
async def test_resolve_many_rejects_classifier_id_outside_candidate_set(monkeypatch):
    candidate = Entity(id=1, canonical_name="payroll service", display_name="Payroll Service")
    store = FakeEntityStore(
        EntityResolutionContext(
            exact_matches={},
            alias_matches={},
            candidates={"pay service": (candidate,)},
        )
    )
    client = _fixed_client([EntityBatchValidationDecision(mention="pay service", matched_id=999, confidence=1.0)])
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    result = await resolver.resolve_many(_mentions("pay service"), scope=_SCOPE)

    assert result.entity_id("pay service") == 100
    assert store.aliases == []


@pytest.mark.parametrize(
    "mentions",
    (
        ("first svc",),
        ("first svc", "first svc"),
    ),
    ids=("missing", "duplicate_hides_missing"),
)
@pytest.mark.asyncio
async def test_resolve_many_rejects_incomplete_adjudication_before_entity_writes(monkeypatch, mentions):
    first = Entity(id=1, canonical_name="first service", display_name="First Service")
    second = Entity(id=2, canonical_name="second service", display_name="Second Service")
    store = FakeEntityStore(
        EntityResolutionContext(
            exact_matches={},
            alias_matches={},
            candidates={"first svc": (first,), "second svc": (second,)},
        )
    )
    client = _fixed_client([EntityBatchValidationDecision(mention=mention, matched_id=1, confidence=0.99) for mention in mentions])
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    with pytest.raises(RuntimeError, match="output_invalid"):
        await resolver.resolve_many(_mentions("first svc", "second svc"), scope=_SCOPE)

    # Accepted decisions are kept; the missing ones are re-asked once, and the batch then fails.
    assert client.calls == 2
    assert "<correction>" in client.prompts[1]
    assert store.created == []
    assert store.aliases == []


@pytest.mark.asyncio
async def test_resolve_many_retries_incomplete_adjudication_once(monkeypatch):
    first = Entity(id=1, canonical_name="first service", display_name="First Service")
    second = Entity(id=2, canonical_name="second service", display_name="Second Service")
    store = FakeEntityStore(
        EntityResolutionContext(
            exact_matches={},
            alias_matches={},
            candidates={"first svc": (first,), "second svc": (second,)},
        )
    )
    client = _sequenced_client(
        [
            [EntityBatchValidationDecision(mention="first svc", matched_id=1, confidence=0.99)],
            [
                EntityBatchValidationDecision(mention="first svc", matched_id=1, confidence=0.99),
                EntityBatchValidationDecision(mention="second svc", matched_id=2, confidence=0.99),
            ],
        ]
    )
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    result = await resolver.resolve_many(_mentions("first svc", "second svc"), scope=_SCOPE)

    assert client.calls == 2
    assert "<correction>" in client.prompts[1]
    assert result.metrics.structured_llm_calls == 2
    assert result.metrics.validation_retries == 1
    assert result.entity_id("first svc") == 1
    assert result.entity_id("second svc") == 2


@pytest.mark.asyncio
async def test_resolve_many_bounds_context_batches_and_packs_adjudication_by_capacity(monkeypatch):
    mentions = tuple(f"service {index}" for index in range(5))
    candidates = {
        mention: (Entity(id=index + 1, canonical_name=f"canonical {index}", display_name=f"Canonical {index}"),)
        for index, mention in enumerate(mentions)
    }

    class ChunkedStore(FakeEntityStore):
        async def load_entity_resolution_context(self, canonical_names, *, candidate_limit, scope):
            self.context_calls += 1
            return EntityResolutionContext(
                exact_matches={},
                alias_matches={},
                candidates={name: candidates[name] for name in canonical_names},
            )

    store = ChunkedStore(EntityResolutionContext({}, {}, {}))
    client = EntityClient(respond=_match_first_candidate, max_cases=2, max_concurrent=2)
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
        policy=EntityResolutionPolicy(context_batch_size=2),
    )

    result = await resolver.resolve_many(_mentions(*mentions), scope=_SCOPE)

    assert store.context_calls == 3
    assert [[case["mention"] for case in _cases(prompt)] for prompt in client.prompts] == [
        list(mentions[0:2]), list(mentions[2:4]), list(mentions[4:]),
    ]
    assert tuple(result.entity_id(mention) for mention in mentions) == (1, 2, 3, 4, 5)
    assert result.metrics.structured_llm_calls == 3
    assert result.metrics.new_entities == 0
    assert client.peak_in_flight == 2


@pytest.mark.asyncio
async def test_resolve_many_fails_before_entity_writes_when_one_mention_exceeds_capacity(monkeypatch):
    candidate = Entity(id=1, canonical_name="payroll service", display_name="Payroll Service")
    store = FakeEntityStore(
        EntityResolutionContext(exact_matches={}, alias_matches={}, candidates={"pay service": (candidate,)})
    )
    client = EntityClient(respond=_match_first_candidate, max_cases=0)
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
    )

    with pytest.raises(RuntimeError, match="input_capacity_exceeded"):
        await resolver.resolve_many(_mentions("pay service"), scope=_SCOPE)

    assert client.prompts == []
    assert store.created == []


@pytest.mark.asyncio
async def test_resolve_many_keeps_provider_error_type(monkeypatch):
    candidate = Entity(id=1, canonical_name="payroll service", display_name="Payroll Service")
    store = FakeEntityStore(
        EntityResolutionContext(exact_matches={}, alias_matches={}, candidates={"pay service": (candidate,)})
    )

    def respond(_prompt: str) -> EntityBatchValidationResponse:
        raise StructuredLlmError("provider down", terminal_category="provider_error", error_code="provider_error")

    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=EntityClient(respond=respond),
    )

    with pytest.raises(StructuredLlmError, match="provider down"):
        await resolver.resolve_many(_mentions("pay service"), scope=_SCOPE)


@pytest.mark.asyncio
async def test_sqlite_entity_resolution_context_is_bounded_and_bulk(db):
    exact_id = await db.upsert_entity("memforge", "MemForge")
    candidate_id = await db.upsert_entity("payroll service", "Payroll Service")
    await db.insert_alias(
        alias="MF",
        alias_normalized="mf",
        canonical_id=exact_id,
        source="admin_manual",
    )

    context = await db.load_entity_resolution_context(
        ("memforge", "mf", "pay service", "missing"),
        candidate_limit=2,
        scope=_SCOPE,
    )

    assert context.exact_matches["memforge"].id == exact_id
    assert context.alias_matches["mf"].canonical_id == exact_id
    assert [item.id for item in context.candidates["pay service"]] == [candidate_id]
    assert context.candidates["missing"] == ()

    created = await db.upsert_entities((EntityUpsert("new component", "New Component"),))
    await db.insert_aliases(
        (
            EntityAlias(
                alias="NC",
                alias_normalized="nc",
                canonical_id=created["new component"],
                source="resolver_confirmed",
                access_context_hash=_SCOPE.access_context_hash,
            ),
        )
    )
    current_scope = await db.load_entity_resolution_context(
        ("nc",),
        candidate_limit=2,
        scope=_SCOPE,
    )
    assert current_scope.alias_matches["nc"].canonical_id == created["new component"]
    assert await db.get_entity_by_alias("nc") is None

    await db.insert_aliases(
        (
            EntityAlias(
                alias="NC",
                alias_normalized="nc",
                canonical_id=created["new component"],
                source="resolver_confirmed",
                access_context_hash="access-b",
            ),
        )
    )
    alias_rows = await db.db.execute_fetchall(
        "SELECT access_context_hash FROM entity_aliases WHERE alias_normalized = ? ORDER BY access_context_hash",
        ("nc",),
    )
    assert [row["access_context_hash"] for row in alias_rows] == ["access-a", "access-b"]

    other_scope = await db.load_entity_resolution_context(
        ("nc",),
        candidate_limit=2,
        scope=EntityResolutionScope(access_context_hash="access-b"),
    )
    assert other_scope.alias_matches["nc"].canonical_id == created["new component"]


@pytest.mark.asyncio
async def test_sqlite_entity_context_preserves_tail_mentions_beyond_global_term_cap(db):
    mentions = tuple(f"service{index:02d} request" for index in range(12))
    for index in range(12):
        await db.upsert_entity(f"service{index:02d} canonical", f"Service {index:02d}")

    context = await db.load_entity_resolution_context(
        mentions,
        candidate_limit=2,
        scope=_SCOPE,
    )

    assert [entity.canonical_name for entity in context.candidates[mentions[-1]]] == [
        "service11 canonical"
    ]
