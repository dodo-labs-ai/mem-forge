from __future__ import annotations

import ast
from pathlib import Path


def _method_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_provider_backed_memory_writes_expose_only_projected_lifecycle_entrypoint() -> None:
    root = Path(__file__).resolve().parents[1]

    engine_methods = _method_names(root / "src/memforge/memory/engine.py")
    assert "apply_projected_lifecycle" in engine_methods
    assert "apply_projected_tombstone" in engine_methods
    assert "process_memories" not in engine_methods
    assert not {
        "_document_evidence_unit",
        "_evidence_unit_has_materialized_memory",
        "_record_document_relation_outcome",
        "_document_relation_outcome_bundle",
    } & engine_methods
    assert "_retire_orphaned_memories" not in _method_names(
        root / "src/memforge/pipeline/sync.py"
    )


def test_memory_engine_lifecycle_writes_include_relation_outcome() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src/memforge/memory/engine.py"
    tree = ast.parse(source_path.read_text())
    guarded_methods = {
        "insert_memory",
        "insert_memory_with_source_and_relation",
        "mark_pending_review",
        "supersede_memory",
        "supersede_memory_and_upsert_agent_claim",
        "update_memory_status_with_relation_outcome",
    }
    forbidden_methods = {"update_memory_status"}
    missing: list[tuple[str, int]] = []
    forbidden: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Attribute) or node.func.value.attr != "memory_store":
            continue
        if node.func.attr in forbidden_methods:
            forbidden.append((node.func.attr, node.lineno))
            continue
        if node.func.attr not in guarded_methods:
            continue
        if not any(keyword.arg == "relation_outcome" for keyword in node.keywords):
            missing.append((node.func.attr, node.lineno))

    assert missing == []
    assert forbidden == []
