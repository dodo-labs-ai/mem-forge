from __future__ import annotations

from memforge.plugin_mcp_proxy import _compact_memory_response, _compact_search_result


def _conflict_context() -> dict[str, object]:
    return {
        "review_id": "review-conflict-1",
        "counterpart_memory_id": "mem-counterpart",
        "counterpart_summary": "Payroll closes on the 25th.",
        "review_status": "approved",
        "disposition": "confirmed",
        "reason": "Same scope, incompatible deadline.",
        "review_note": "Confirmed after source comparison.",
        "reviewer": "reviewer-1",
        "resolved_at": "2026-08-09T01:00:00+00:00",
    }


def test_search_compaction_preserves_conflict_warning_and_structured_context() -> None:
    result = _compact_search_result(
        {
            "memory_id": "mem-primary",
            "summary": "Payroll closes on the 20th.",
            "contradiction_warning": "This memory has 1 reviewed cross-source conflict(s).",
            "conflict_contexts": [_conflict_context()],
            "retrieval_evidence": {"large": "internal payload"},
        }
    )

    assert result["contradiction_warning"].startswith("This memory has 1 reviewed")
    assert result["conflict_contexts"] == [_conflict_context()]
    assert "retrieval_evidence" not in result


def test_memory_detail_compaction_preserves_structured_conflict_context() -> None:
    result = _compact_memory_response(
        {
            "id": "mem-primary",
            "content": "Payroll closes on the 20th.",
            "conflict_contexts": [_conflict_context()],
            "owner_user_id": "internal-owner",
        }
    )

    assert result["conflict_contexts"] == [_conflict_context()]
    assert "owner_user_id" not in result


def test_memory_detail_compaction_preserves_grouped_role_aware_evidence() -> None:
    result = _compact_memory_response(
        {
            "id": "mem-primary",
            "content": "Deployment requires approval.",
            "evidence": [
                {
                    "evidence_unit_id": "eu-v2-1",
                    "support_scope_version": "evidence-unit-set-v2",
                    "source_type": "github_repo",
                    "doc_id": "doc-1",
                    "current": True,
                    "legacy_limited": False,
                    "items": [
                        {
                            "role": "primary",
                            "kind": "text",
                            "support_contribution": True,
                            "excerpt": "Deployment requires approval.",
                            "current": True,
                            "raw_content_sha256": "internal",
                        },
                        {
                            "role": "context",
                            "kind": "artifact",
                            "support_contribution": False,
                            "current": True,
                            "artifact": {
                                "summary": "Approval flow diagram",
                                "evidence_role": "context",
                                "filename": "flow.png",
                                "content_type": "image/png",
                                "size_bytes": 42,
                                "url": "/api/v1/source-artifacts/rev-1",
                                "sha256": "internal",
                            },
                        },
                    ],
                }
            ],
        }
    )

    [unit] = result["evidence"]
    assert unit["evidence_unit_id"] == "eu-v2-1"
    assert [item["role"] for item in unit["items"]] == ["primary", "context"]
    assert unit["items"][1]["artifact"]["url"].endswith("/rev-1")
    assert "raw_content_sha256" not in unit["items"][0]
    assert "sha256" not in unit["items"][1]["artifact"]
