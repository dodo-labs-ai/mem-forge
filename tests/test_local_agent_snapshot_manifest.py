from dataclasses import dataclass

import pytest

from memforge.local_agent.snapshot_manifest import (
    ExcludedScopeListing,
    SnapshotManifestItem,
    plan_snapshot_manifest,
    snapshot_manifest_sha256,
    validate_excluded_listings_for_job,
    validate_snapshot_manifest_values,
)


@dataclass(frozen=True)
class _RetainedInput:
    input_id: str
    input_generation: int
    metadata: dict


def _retained(
    doc_id: str,
    revision: str,
    *,
    input_id: str,
    generation: int,
    attested: bool = True,
) -> _RetainedInput:
    package_hash = f"package-{input_id}" if attested else ""
    return _RetainedInput(
        input_id=input_id,
        input_generation=generation,
        metadata={
            "package_sha256": package_hash,
            "manifest_entry": {
                "doc_id": doc_id,
                "version": revision,
                "package_sha256": package_hash,
            },
        },
    )


def test_manifest_plan_reuses_exact_attested_revision_and_requests_only_changes() -> None:
    plan = plan_snapshot_manifest(
        [
            SnapshotManifestItem(doc_id="doc-a", revision="v1"),
            SnapshotManifestItem(doc_id="doc-b", revision="v2"),
            SnapshotManifestItem(doc_id="doc-c", revision="v1"),
        ],
        [
            _retained("doc-a", "v1", input_id="input-a", generation=1),
            _retained("doc-b", "v1", input_id="input-b", generation=2),
            _retained("doc-c", "v1", input_id="legacy-c", generation=3, attested=False),
        ],
    )

    assert plan.reused_memberships == (("doc-a", "input-a"),)
    assert plan.required_doc_ids == ("doc-b", "doc-c")


def test_manifest_plan_rejects_duplicate_or_incomplete_identity() -> None:
    with pytest.raises(ValueError, match="duplicate doc_id"):
        plan_snapshot_manifest(
            [
                SnapshotManifestItem(doc_id="doc-a", revision="v1"),
                SnapshotManifestItem(doc_id="doc-a", revision="v2"),
            ],
            [],
        )
    with pytest.raises(ValueError, match="require doc_id and revision"):
        plan_snapshot_manifest([SnapshotManifestItem(doc_id="doc-a", revision="")], [])


def test_manifest_plan_prefers_provider_revision_over_semantic_package_version() -> None:
    retained = _retained("doc-a", "semantic-hash", input_id="input-a", generation=1)
    retained.metadata["manifest_entry"]["provider_revision"] = "provider-v7"

    plan = plan_snapshot_manifest(
        [SnapshotManifestItem(doc_id="doc-a", revision="provider-v7")],
        [retained],
    )

    assert plan.reused_memberships == (("doc-a", "input-a"),)
    assert plan.required_doc_ids == ()


def test_manifest_plan_does_not_reuse_a_stored_tombstone() -> None:
    # A manifest stored before every item was an upsert may hold a tombstone.
    retained = _retained("doc-1", "revision-1", input_id="input-1", generation=1)
    retained.metadata["manifest_entry"]["change_kind"] = "tombstone"

    plan = plan_snapshot_manifest([SnapshotManifestItem(doc_id="doc-1", revision="revision-1")], [retained])

    assert plan.required_doc_ids == ("doc-1",)
    assert plan.reused_memberships == ()


def test_manifest_digest_is_the_digest_agents_and_stored_manifests_already_carry() -> None:
    # Coverage and change kind stay in the digest as constants, so every
    # stored manifest digest and every replay of it still match.
    items = [
        SnapshotManifestItem(doc_id="jira-PAY-9", revision="2026-10-01T10:00:00.000+0000"),
        SnapshotManifestItem(doc_id="jira-PAY-10", revision="2026-10-02T11:30:00.000+0000"),
    ]

    assert snapshot_manifest_sha256(items) == "785035e8f86e446f38ba663f6a87d0a83ae1f7bb2ee0211be2491e7aba380169"
    assert snapshot_manifest_sha256(reversed(items)) == snapshot_manifest_sha256(items)
    assert snapshot_manifest_sha256([]) == "0867a2ae856beb4752421ad5c5696240ab177536abda256af940581d4da120b1"


@pytest.mark.parametrize(
    ("coverage", "change_kind", "message"),
    [
        ("bounded_delta", "upsert", "coverage is invalid"),
        ("partial", "upsert", "coverage is invalid"),
        ("complete_snapshot", "tombstone", "change_kind is invalid"),
    ],
)
def test_a_manifest_declares_complete_coverage_of_upserts_only(coverage: str, change_kind: str, message: str) -> None:
    validate_snapshot_manifest_values("complete_snapshot", [("doc-a", "v1", "upsert")])
    with pytest.raises(ValueError, match=message):
        validate_snapshot_manifest_values(coverage, [("doc-a", "v1", change_kind)])


def test_excluded_listings_enter_the_digest_only_when_a_manifest_carries_them() -> None:
    items = [
        SnapshotManifestItem(doc_id="jira-PAY-9", revision="2026-10-01T10:00:00.000+0000"),
        SnapshotManifestItem(doc_id="jira-PAY-10", revision="2026-10-02T11:30:00.000+0000"),
    ]
    listings = (
        ExcludedScopeListing("scope-b", frozenset({"jira-PAY-3", "jira-PAY-1"})),
        ExcludedScopeListing("scope-a", rejection="JiraQueryRejected: no"),
    )

    # Without listings, the digest every stored manifest already carries.
    assert snapshot_manifest_sha256(items, ()) == "785035e8f86e446f38ba663f6a87d0a83ae1f7bb2ee0211be2491e7aba380169"
    assert snapshot_manifest_sha256(items, listings) == (
        "c37e3c9a0ca799b2340ee3c89094aa598152b79518aac9fcc46ab843a2c71eb6"
    )
    assert snapshot_manifest_sha256(items, tuple(reversed(listings))) == snapshot_manifest_sha256(items, listings)
    assert snapshot_manifest_sha256(items, listings[:1]) != snapshot_manifest_sha256(items, listings)


def test_a_manifest_lists_only_the_excluded_scopes_its_job_carried() -> None:
    listing = ExcludedScopeListing("scope-old", frozenset({"jira-PAY-1"}))

    validate_excluded_listings_for_job([listing], {"scope-old", "scope-superseded"})
    validate_excluded_listings_for_job([], {"scope-old"})
    with pytest.raises(ValueError, match="did not carry: scope-old"):
        validate_excluded_listings_for_job([listing], set())
    with pytest.raises(ValueError, match="duplicate scope_key"):
        validate_excluded_listings_for_job([listing, ExcludedScopeListing("scope-old")], {"scope-old"})
