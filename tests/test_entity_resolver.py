"""Tests for the entity resolution pipeline."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from memforge.memory.entity_resolver import validate_alias, EntityResolver
from memforge.models import Entity, canonicalize_entity_name


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
# Entity model (tags)
# ---------------------------------------------------------------------------


class TestEntityTags:
    def test_entity_tags_default(self):
        """Entity tags default to empty list."""
        e = Entity(id=1, canonical_name="test")
        assert e.tags == []

    def test_entity_tags_multi(self):
        """Entity can have multiple tags."""
        e = Entity(id=1, canonical_name="auth-service", tags=["service", "api"])
        assert e.tags == ["service", "api"]

    def test_entity_type_backward_compat(self):
        """Deprecated entity_type property returns first tag."""
        e = Entity(id=1, canonical_name="postgresql", tags=["technology"])
        assert e.entity_type == "technology"

    def test_entity_type_empty_tags(self):
        """entity_type returns 'unknown' when no tags."""
        e = Entity(id=1, canonical_name="test")
        assert e.entity_type == "unknown"

    def test_entity_resolver_class_exists(self):
        """EntityResolver class is importable and has resolve method."""
        assert hasattr(EntityResolver, "resolve")
        assert hasattr(EntityResolver, "invalidate_cache")

    @pytest.mark.asyncio
    async def test_embedding_cache_load_is_single_flight_under_concurrent_resolve(self, monkeypatch):
        class FakeDb:
            def __init__(self) -> None:
                self.entities = [
                    Entity(id=1, canonical_name="pay api", display_name="pay-api"),
                    Entity(id=2, canonical_name="auth service", display_name="auth-service"),
                ]
                self.full_embedding_batches = 0

            async def get_entity_by_canonical(self, canonical_name: str):
                del canonical_name
                return None

            async def get_entity_by_alias(self, alias_normalized: str):
                del alias_normalized
                return None

            async def get_all_entities(self):
                await asyncio.sleep(0.01)
                return self.entities

            async def insert_alias(self, **_kwargs):
                return None

            async def upsert_entity(self, canonical_name: str, display_name: str, tags=None):
                raise AssertionError(f"unexpected new entity: {canonical_name}/{display_name}/{tags}")

        db = FakeDb()

        def fake_embed_texts(names, base_url, api_key, model):
            del base_url, api_key, model
            if len(names) == len(db.entities):
                db.full_embedding_batches += 1
            return [[1.0, 0.0] for _ in names]

        monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

        resolver = EntityResolver(
            db=db,  # type: ignore[arg-type]
            embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        )

        results = await asyncio.gather(*(resolver.resolve("pay-api") for _ in range(5)))

        assert results == [1, 1, 1, 1, 1]
        assert db.full_embedding_batches == 1

    @pytest.mark.asyncio
    async def test_embedding_cache_full_load_is_single_flight_across_resolvers(self, monkeypatch):
        class FakeDb:
            def __init__(self) -> None:
                self.entities = [
                    Entity(id=1, canonical_name="pay api", display_name="pay-api"),
                    Entity(id=2, canonical_name="auth service", display_name="auth-service"),
                ]

            async def get_entity_by_canonical(self, canonical_name: str):
                del canonical_name
                return None

            async def get_entity_by_alias(self, alias_normalized: str):
                del alias_normalized
                return None

            async def get_all_entities(self):
                await asyncio.sleep(0.01)
                return self.entities

            async def insert_alias(self, **_kwargs):
                return None

            async def upsert_entity(self, canonical_name: str, display_name: str, tags=None):
                raise AssertionError(f"unexpected new entity: {canonical_name}/{display_name}/{tags}")

        active_full_loads = 0
        max_active_full_loads = 0
        counter_lock = threading.Lock()
        db = FakeDb()

        def fake_embed_texts(names, base_url, api_key, model):
            nonlocal active_full_loads, max_active_full_loads
            del base_url, api_key, model
            if len(names) == len(db.entities):
                with counter_lock:
                    active_full_loads += 1
                    max_active_full_loads = max(max_active_full_loads, active_full_loads)
                try:
                    time.sleep(0.05)
                finally:
                    with counter_lock:
                        active_full_loads -= 1
            return [[1.0, 0.0] for _ in names]

        monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

        resolvers = [
            EntityResolver(
                db=db,  # type: ignore[arg-type]
                embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
            )
            for _ in range(3)
        ]

        results = await asyncio.gather(*(resolver.resolve("pay-api") for resolver in resolvers))

        assert results == [1, 1, 1]
        assert max_active_full_loads == 1
