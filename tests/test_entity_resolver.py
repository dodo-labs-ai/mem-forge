"""Tests for the entity resolution pipeline."""

from __future__ import annotations

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
