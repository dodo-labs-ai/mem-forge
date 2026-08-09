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
