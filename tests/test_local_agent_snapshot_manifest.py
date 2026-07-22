from dataclasses import dataclass

import pytest

from memforge.local_agent.snapshot_manifest import (
    CollectionChangeKind,
    CollectionCoverage,
    SnapshotManifestItem,
    plan_snapshot_manifest,
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
            SnapshotManifestItem(doc_id="doc-a", revision="v1", change_kind=CollectionChangeKind.UPSERT),
            SnapshotManifestItem(doc_id="doc-b", revision="v2", change_kind=CollectionChangeKind.UPSERT),
            SnapshotManifestItem(doc_id="doc-c", revision="v1", change_kind=CollectionChangeKind.UPSERT),
        ],
        [
            _retained("doc-a", "v1", input_id="input-a", generation=1),
            _retained("doc-b", "v1", input_id="input-b", generation=2),
            _retained("doc-c", "v1", input_id="legacy-c", generation=3, attested=False),
        ],
        coverage=CollectionCoverage.COMPLETE_SNAPSHOT,
    )

    assert plan.reused_memberships == (("doc-a", "input-a"),)
    assert plan.required_doc_ids == ("doc-b", "doc-c")


def test_manifest_plan_rejects_duplicate_or_incomplete_identity() -> None:
    with pytest.raises(ValueError, match="duplicate doc_id"):
        plan_snapshot_manifest(
            [
                SnapshotManifestItem(
                    doc_id="doc-a",
                    revision="v1",
                    change_kind=CollectionChangeKind.UPSERT,
                ),
                SnapshotManifestItem(
                    doc_id="doc-a",
                    revision="v2",
                    change_kind=CollectionChangeKind.UPSERT,
                ),
            ],
            [],
            coverage=CollectionCoverage.COMPLETE_SNAPSHOT,
        )
    with pytest.raises(ValueError, match="require doc_id and revision"):
        plan_snapshot_manifest(
            [
                SnapshotManifestItem(
                    doc_id="doc-a",
                    revision="",
                    change_kind=CollectionChangeKind.UPSERT,
                )
            ],
            [],
            coverage=CollectionCoverage.COMPLETE_SNAPSHOT,
        )


def test_manifest_plan_rejects_partial_coverage_before_reuse_or_deletion() -> None:
    with pytest.raises(ValueError, match="partial collection cannot be planned"):
        plan_snapshot_manifest(
            [
                SnapshotManifestItem(
                    doc_id="doc-a",
                    revision="v1",
                    change_kind=CollectionChangeKind.UPSERT,
                )
            ],
            [_retained("doc-a", "v1", input_id="input-a", generation=1)],
            coverage=CollectionCoverage.PARTIAL,
        )


def test_manifest_plan_keeps_explicit_tombstone_in_a_bounded_delta() -> None:
    plan = plan_snapshot_manifest(
        [
            SnapshotManifestItem(
                doc_id="doc-a",
                revision="deleted-v2",
                change_kind=CollectionChangeKind.TOMBSTONE,
            )
        ],
        [],
        coverage=CollectionCoverage.BOUNDED_DELTA,
    )

    assert plan.required_doc_ids == ("doc-a",)
    assert plan.reused_memberships == ()


def test_manifest_plan_prefers_provider_revision_over_semantic_package_version() -> None:
    retained = _retained("doc-a", "semantic-hash", input_id="input-a", generation=1)
    retained.metadata["manifest_entry"]["provider_revision"] = "provider-v7"

    plan = plan_snapshot_manifest(
        [
            SnapshotManifestItem(
                doc_id="doc-a",
                revision="provider-v7",
                change_kind=CollectionChangeKind.UPSERT,
            )
        ],
        [retained],
        coverage=CollectionCoverage.COMPLETE_SNAPSHOT,
    )

    assert plan.reused_memberships == (("doc-a", "input-a"),)
    assert plan.required_doc_ids == ()


def test_manifest_plan_does_not_reuse_tombstone_as_upsert() -> None:
    retained = _retained("doc-1", "revision-1", input_id="input-1", generation=1)
    retained.metadata["manifest_entry"]["change_kind"] = "tombstone"

    plan = plan_snapshot_manifest(
        [
            SnapshotManifestItem(
                doc_id="doc-1",
                revision="revision-1",
                change_kind=CollectionChangeKind.UPSERT,
            )
        ],
        [retained],
        coverage=CollectionCoverage.BOUNDED_DELTA,
    )

    assert plan.required_doc_ids == ("doc-1",)
    assert plan.reused_memberships == ()
