from __future__ import annotations

from types import SimpleNamespace

import pytest

from memforge.llm.relation_catalog import RelationCoverage, RequestCatalog, request_ref


def test_prefixes_overflow_and_conflicting_snapshots():
    for prefix in ("NEW", "MEM", "PRM", "REQ", "WRK"):
        assert request_ref(prefix, 1) == prefix + "-0001"
        assert request_ref(prefix, 9999) == prefix + "-9999"
    for prefix, ordinal in (("N", 1), ("new", 1), ("NEW", 0), ("NEW", 10000)):
        with pytest.raises(ValueError):
            request_ref(prefix, ordinal)
    catalog = RequestCatalog("MEM")
    assert catalog.add("id", {"text": "A"}) == catalog.add("id", {"text": "A"})
    with pytest.raises(ValueError, match="conflicting"):
        catalog.add("id", {"text": "B"})


def test_uncertain_refs_share_the_same_duplicate_and_allowed_validation():
    coverage = RelationCoverage({"NEW-0001": frozenset({"MEM-0001"})})
    assert coverage.row_error(
        SimpleNamespace(candidate_id="NEW-0001", relations=[], uncertain_existing_ids=["MEM-0001"])
    ) is None
    assert coverage.row_error(
        SimpleNamespace(
            candidate_id="NEW-0001",
            relations=[SimpleNamespace(existing_id="MEM-0001")],
            uncertain_existing_ids=["MEM-0001"],
        )
    ) == "NEW-0001 names MEM-0001 more than once"
    assert coverage.row_error(
        SimpleNamespace(candidate_id="NEW-0001", relations=[], uncertain_existing_ids=["MEM-0002"])
    ) == "NEW-0001 names MEM-0002, which is not in its allowed list"
