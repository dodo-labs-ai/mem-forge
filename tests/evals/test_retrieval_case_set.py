from __future__ import annotations

from dataclasses import fields
from importlib import resources
import subprocess
import sys

import pytest

from memforge.storage.adapters.context import AccessScope


def test_retrieval_case_resources_are_importable_from_package() -> None:
    case_dir = resources.files("memforge.evals.retrieval.cases")

    assert case_dir.joinpath("manifest.yaml").is_file()
    assert case_dir.joinpath("metadata_lexical.yaml").is_file()


def test_load_case_set_validates_manifest_cases_and_scope_shape() -> None:
    from memforge.evals.retrieval import load_case_set

    case_set = load_case_set("retrieval-core-v1")

    assert case_set.manifest.case_set_id == "retrieval-core-v1"
    assert case_set.manifest.case_schema_version == 1
    assert case_set.case_ids == (
        "exact_external_id_lookup",
        "metadata_title_exact",
        "compact_trigram_metadata_recall",
        "queryless_source_listing",
    )

    title_case = case_set.get_case("metadata_title_exact")
    assert title_case.expected.required_channels["mem-access-review"] == (
        "bm25_metadata_tokens",
    )
    assert set(title_case.scope.raw) == {field.name for field in fields(AccessScope)}

    access_scope = title_case.scope.to_access_scope()
    assert access_scope == AccessScope(
        user_id="eval-user",
        include_private=False,
        allowed_statuses=("active",),
        active_project="PAY",
        scope_mode="project-first",
        active_repo_identifier=None,
    )


def test_case_set_hash_uses_canonical_content_and_excludes_hash_field() -> None:
    from memforge.evals.retrieval import compute_case_set_sha, load_case_set, validate_case_set_sha

    case_set = load_case_set("retrieval-core-v1")

    assert validate_case_set_sha(case_set)
    assert compute_case_set_sha(case_set) == case_set.manifest.case_set_sha

    mutated = case_set.with_manifest_sha("sha256:not-the-real-value")
    assert compute_case_set_sha(mutated) == case_set.manifest.case_set_sha

    changed_query = case_set.replace_case(
        "metadata_title_exact",
        query="Create Access Review in Annual Payroll",
    )
    assert compute_case_set_sha(changed_query) != case_set.manifest.case_set_sha


def test_case_set_validation_rejects_missing_fixture_references() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest = {
        "case_schema_version": 1,
        "case_set_id": "broken",
        "case_set_sha": "sha256:placeholder",
        "case_files": ["cases.yaml"],
        "fixtures": {
            "default": {
                "workspace_id": "ws-eval",
                "tenant_id": "tenant-eval",
                "users": ["eval-user"],
                "sources": [{"id": "src-payroll", "status": "active"}],
                "source_subscriptions": [
                    {
                        "user_id": "eval-user",
                        "source_id": "src-payroll",
                        "enabled": True,
                    }
                ],
                "documents": [],
                "memories": [{"id": "mem-known"}],
            }
        },
    }
    cases = {
        "cases.yaml": [
            {
                "id": "missing-memory",
                "family": "exact_title_lookup",
                "description": "references a memory absent from the fixture",
                "query": "known title",
                "top_k": 10,
                "entities": [],
                "fixture_variant": "default",
                "scope": {
                    "user_id": "eval-user",
                    "include_private": False,
                    "allowed_statuses": ["active"],
                    "active_project": "PAY",
                    "scope_mode": "project-first",
                    "active_repo_identifier": None,
                },
                "expected": {
                    "relevant": {
                        "mem-missing": 3,
                    },
                },
            }
        ]
    }

    with pytest.raises(CaseSetValidationError, match="mem-missing"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_unknown_case_fields() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["required_channel"] = ["bm25_metadata_tokens"]

    with pytest.raises(CaseSetValidationError, match="Unknown fields"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_wrong_scalar_types() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["top_k"] = "10"

    with pytest.raises(CaseSetValidationError, match="top_k"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_non_positive_schema_version() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    manifest["case_schema_version"] = 0

    with pytest.raises(CaseSetValidationError, match="case_schema_version"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_missing_core_case_fields() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    for field_name in ("query", "top_k", "fixture_variant", "entities"):
        manifest, cases = _minimal_valid_case_set_data()
        del cases["cases.yaml"][0][field_name]

        with pytest.raises(CaseSetValidationError, match=field_name):
            load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_path_unsafe_case_id() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["id"] = "../unsafe"

    with pytest.raises(CaseSetValidationError, match="Case id"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_boolean_top_k() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["top_k"] = False

    with pytest.raises(CaseSetValidationError, match="top_k"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_non_positive_top_k() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["top_k"] = 0

    with pytest.raises(CaseSetValidationError, match="top_k"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_negative_offset() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["offset"] = -1

    with pytest.raises(CaseSetValidationError, match="offset"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_null_query() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["query"] = None

    with pytest.raises(CaseSetValidationError, match="query"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_unknown_source_filter_and_time_range_fields() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["source_filter"] = {"source_id": "src-payroll"}

    with pytest.raises(CaseSetValidationError, match="source_filter"):
        load_case_set_from_data(manifest, cases)

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["time_range"] = {"after_date": "2026-01-01T00:00:00+00:00"}

    with pytest.raises(CaseSetValidationError, match="time_range"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_string_booleans_in_scope() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["scope"]["include_private"] = "false"

    with pytest.raises(CaseSetValidationError, match="include_private"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_required_channels_scalar() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["expected"]["required_channels"] = {
        "mem-known": "bm25_metadata_tokens",
    }

    with pytest.raises(CaseSetValidationError, match="required_channels"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_expected_relevant_sequence() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    cases["cases.yaml"][0]["expected"]["relevant"] = []

    with pytest.raises(CaseSetValidationError, match="expected.relevant"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_missing_source_subscription_rows() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest = {
        "case_schema_version": 1,
        "case_set_id": "broken",
        "case_set_sha": "sha256:placeholder",
        "case_files": ["cases.yaml"],
        "fixtures": {
            "default": {
                "workspace_id": "ws-eval",
                "tenant_id": "tenant-eval",
                "users": ["eval-user"],
                "sources": [
                    {"id": "src-payroll", "status": "active"},
                    {"id": "src-muted", "status": "active"},
                ],
                "source_subscriptions": [
                    {
                        "user_id": "eval-user",
                        "source_id": "src-payroll",
                        "enabled": True,
                    }
                ],
                "documents": [],
                "memories": [{"id": "mem-known"}],
            }
        },
    }
    cases = {
        "cases.yaml": [
            {
                "id": "known",
                "family": "exact_title_lookup",
                "description": "valid case except incomplete subscription matrix",
                "query": "known title",
                "top_k": 10,
                "entities": [],
                "fixture_variant": "default",
                "scope": {
                    "user_id": "eval-user",
                    "include_private": False,
                    "allowed_statuses": ["active"],
                    "active_project": "PAY",
                    "scope_mode": "project-first",
                    "active_repo_identifier": None,
                },
                "expected": {
                    "relevant": {
                        "mem-known": 3,
                    },
                },
            }
        ]
    }

    with pytest.raises(CaseSetValidationError, match="src-muted"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_document_source_not_in_fixture() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    manifest["fixtures"]["default"]["documents"] = [
        {
            "doc_id": "doc-1",
            "source_id": "src-missing",
        }
    ]

    with pytest.raises(CaseSetValidationError, match="src-missing"):
        load_case_set_from_data(manifest, cases)


def test_case_set_validation_rejects_missing_document_source_id_with_doc_id() -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set_from_data

    manifest, cases = _minimal_valid_case_set_data()
    manifest["fixtures"]["default"]["documents"] = [
        {
            "doc_id": "doc-without-source",
        }
    ]

    with pytest.raises(CaseSetValidationError, match="doc-without-source"):
        load_case_set_from_data(manifest, cases)


def test_case_set_yaml_rejects_duplicate_keys(tmp_path, monkeypatch) -> None:
    from memforge.evals.retrieval import CaseSetValidationError, load_case_set
    from memforge.evals.retrieval import schema

    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "manifest.yaml").write_text(
        """
case_schema_version: 1
case_set_id: retrieval-core-v1
case_set_sha: sha256:placeholder
case_files:
  - metadata_lexical.yaml
fixtures:
  default:
    workspace_id: ws-eval
    tenant_id: tenant-eval
    users: [eval-user]
    sources: [{id: src-payroll, status: active}]
    source_subscriptions:
      - {user_id: eval-user, source_id: src-payroll, enabled: true}
    documents: []
    memories: [{id: mem-known}]
""",
        encoding="utf-8",
    )
    (case_dir / "metadata_lexical.yaml").write_text(
        """
- id: duplicate-key-case
  family: exact_title_lookup
  description: duplicate query should be rejected
  query: first
  query: second
  top_k: 10
  fixture_variant: default
  scope:
    user_id: eval-user
    include_private: false
    allowed_statuses: [active]
    active_project: PAY
    scope_mode: project-first
    active_repo_identifier: null
  expected:
    relevant:
      mem-known: 3
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(schema, "_case_resource_root", lambda: case_dir)

    with pytest.raises(CaseSetValidationError, match="Duplicate YAML key"):
        load_case_set("retrieval-core-v1")


def test_hash_case_set_module_checks_packaged_manifest() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "memforge.evals.retrieval.hash_case_set",
            "--case-set",
            "retrieval-core-v1",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "retrieval-core-v1" in completed.stdout
    assert "OK" in completed.stdout


def test_hash_case_set_module_reports_stale_manifest_on_check(tmp_path) -> None:
    source_case_dir = resources.files("memforge.evals.retrieval.cases")
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for name in ("manifest.yaml", "metadata_lexical.yaml"):
        case_dir.joinpath(name).write_text(
            source_case_dir.joinpath(name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    manifest_path = case_dir / "manifest.yaml"
    current_sha = _manifest_sha(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            current_sha,
            "sha256:stale",
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "memforge.evals.retrieval.hash_case_set",
            "--case-set",
            "retrieval-core-v1",
            "--check",
            "--case-root",
            str(case_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "mismatch expected=sha256:stale" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_hash_case_set_module_writes_manifest_sha(tmp_path, monkeypatch) -> None:
    from memforge.evals.retrieval import compute_case_set_sha, load_case_set
    from memforge.evals.retrieval import schema

    source_case_dir = resources.files("memforge.evals.retrieval.cases")
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for name in ("manifest.yaml", "metadata_lexical.yaml"):
        case_dir.joinpath(name).write_text(
            source_case_dir.joinpath(name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    manifest_path = case_dir / "manifest.yaml"
    current_sha = _manifest_sha(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            current_sha,
            "sha256:stale",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(schema, "_case_resource_root", lambda: case_dir)

    with pytest.raises(ValueError, match="case_set_sha mismatch"):
        load_case_set("retrieval-core-v1")

    stale = load_case_set("retrieval-core-v1", verify_sha=False)
    assert compute_case_set_sha(stale) != stale.manifest.case_set_sha

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "memforge.evals.retrieval.hash_case_set",
            "--case-set",
            "retrieval-core-v1",
            "--write",
            "--case-root",
            str(case_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "wrote" in completed.stdout
    assert "case_files:\n  - metadata_lexical.yaml" in manifest_path.read_text(encoding="utf-8")
    fixed = load_case_set("retrieval-core-v1")
    assert compute_case_set_sha(fixed) == fixed.manifest.case_set_sha


def _manifest_sha(manifest_text: str) -> str:
    for line in manifest_text.splitlines():
        if line.startswith("case_set_sha: "):
            return line.removeprefix("case_set_sha: ")
    raise AssertionError("case_set_sha missing")


def _minimal_valid_case_set_data() -> tuple[dict, dict]:
    manifest = {
        "case_schema_version": 1,
        "case_set_id": "valid",
        "case_set_sha": "sha256:placeholder",
        "case_files": ["cases.yaml"],
        "fixtures": {
            "default": {
                "workspace_id": "ws-eval",
                "tenant_id": "tenant-eval",
                "users": ["eval-user"],
                "sources": [{"id": "src-payroll", "status": "active"}],
                "source_subscriptions": [
                    {
                        "user_id": "eval-user",
                        "source_id": "src-payroll",
                        "enabled": True,
                    }
                ],
                "documents": [],
                "memories": [{"id": "mem-known"}],
            }
        },
    }
    cases = {
        "cases.yaml": [
            {
                "id": "known",
                "family": "exact_title_lookup",
                "description": "valid minimal case",
                "query": "known title",
                "top_k": 10,
                "entities": [],
                "fixture_variant": "default",
                "scope": {
                    "user_id": "eval-user",
                    "include_private": False,
                    "allowed_statuses": ["active"],
                    "active_project": "PAY",
                    "scope_mode": "project-first",
                    "active_repo_identifier": None,
                },
                "expected": {
                    "relevant": {
                        "mem-known": 3,
                    },
                },
            }
        ]
    }
    return manifest, cases
