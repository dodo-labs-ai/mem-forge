import pytest

from memforge.local_agent.device_manifest import ExcludedScopeListing, validate_excluded_listings_for_job


def test_a_manifest_lists_only_the_excluded_scopes_its_job_carried() -> None:
    listing = ExcludedScopeListing("scope-old", frozenset({"jira-PAY-1"}))

    validate_excluded_listings_for_job([listing], {"scope-old", "scope-superseded"})
    validate_excluded_listings_for_job([], {"scope-old"})
    with pytest.raises(ValueError, match="did not carry: scope-old"):
        validate_excluded_listings_for_job([listing], set())
    with pytest.raises(ValueError, match="duplicate scope_key"):
        validate_excluded_listings_for_job([listing, ExcludedScopeListing("scope-old")], {"scope-old"})
