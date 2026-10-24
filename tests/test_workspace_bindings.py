from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from memforge.workspace_bindings import (
    WorkspaceBindingError,
    load_workspace_bindings,
    resolve_workspace_binding,
)


ORIGIN = "https://memforge.example.com"
SETUP_HELPER = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "codex"
    / "memforge-memory"
    / "skills"
    / "memforge-setup"
    / "scripts"
    / "workspace_config.py"
)


def _load_setup_helper():
    spec = importlib.util.spec_from_file_location("memforge_workspace_setup_test", SETUP_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, target: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"version": 1, "targets": {ORIGIN: target}}),
        encoding="utf-8",
    )


def _init_git_repo(path: Path, remote: str) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", remote],
        check=True,
    )


def test_missing_file_resolves_no_local_workspace(tmp_path: Path) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()

    result = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(working_directory),
        config_path=tmp_path / "missing.json",
    )

    assert result.workspace_id is None
    assert result.source == "none"
    assert result.repo_identifier is None


def test_repository_binding_uses_normalized_git_origin(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _init_git_repo(repository, "git@github.com:Acme/Payroll-Agent.git")
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {
            "repository_bindings": {
                "github.com/acme/payroll-agent": "payroll_agent",
            }
        },
    )

    result = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(repository),
        config_path=config,
    )

    assert result.workspace_id == "payroll_agent"
    assert result.source == "repository"
    assert result.repo_identifier == "github.com/acme/payroll-agent"


def test_directory_binding_covers_descendants_and_most_specific_wins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    specific = root / "private"
    child = specific / "drafts"
    child.mkdir(parents=True)
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {
            "directory_bindings": {
                str(root): "mount_tai",
                str(specific): "payroll_agent",
            }
        },
    )

    result = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(child),
        config_path=config,
    )

    assert result.workspace_id == "payroll_agent"
    assert result.source == "directory"
    assert result.repo_identifier is None


def test_specific_directory_binding_overrides_repository_binding(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _init_git_repo(repository, "https://github.com/acme/payroll-agent.git")
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {
            "repository_bindings": {
                "github.com/acme/payroll-agent": "repository_workspace",
            },
            "directory_bindings": {
                str(repository): "directory_workspace",
            },
        },
    )

    result = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(repository),
        config_path=config,
    )

    assert result.workspace_id == "directory_workspace"
    assert result.source == "directory"
    assert result.repo_identifier == "github.com/acme/payroll-agent"


@pytest.mark.parametrize("value", ["capture_workspace", "", None, 7])
def test_obsolete_hook_workspace_key_parses_and_selects_no_workspace(
    tmp_path: Path,
    value: object,
) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {
            "hook_workspace_id": value,
            "repository_bindings": {"github.com/acme/payroll": "payroll_workspace"},
        },
    )

    bindings = load_workspace_bindings(config)
    result = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(working_directory),
        config_path=config,
    )

    assert bindings.targets[ORIGIN].repository_bindings == {"github.com/acme/payroll": "payroll_workspace"}
    assert result.workspace_id is None
    assert result.source == "none"


def test_bindings_are_scoped_to_the_exact_memforge_origin(tmp_path: Path) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {"directory_bindings": {str(working_directory): "dev_workspace"}},
    )

    result = resolve_workspace_binding(
        origin="https://another.example.com",
        working_directory=str(working_directory),
        config_path=config,
    )

    assert result.workspace_id is None
    assert result.source == "none"


def test_multiple_roots_with_different_bindings_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    config = tmp_path / "bindings.json"
    _write_config(
        config,
        {
            "directory_bindings": {
                str(first): "workspace_a",
                str(second): "workspace_b",
            }
        },
    )

    with pytest.raises(WorkspaceBindingError, match="workspace_binding_ambiguous"):
        resolve_workspace_binding(
            origin=ORIGIN,
            root_paths=(str(first), str(second)),
            config_path=config,
        )


@pytest.mark.parametrize(
    "document",
    [
        {"version": 2, "targets": {}},
        {"version": 1, "targets": []},
        {
            "version": 1,
            "targets": {ORIGIN: {"directory_bindings": {"relative/path": "workspace"}}},
        },
        {
            "version": 1,
            "targets": {ORIGIN: {"repository_bindings": {"github.com/acme/repo": ""}}},
        },
        {
            "version": 1,
            "targets": {ORIGIN: {"capture_workspace_id": "workspace"}},
        },
    ],
)
def test_invalid_binding_documents_fail_closed(
    tmp_path: Path,
    document: dict[str, object],
) -> None:
    config = tmp_path / "bindings.json"
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(WorkspaceBindingError):
        load_workspace_bindings(config)


def test_setup_helper_previews_and_applies_ordinary_directory_binding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    helper = _load_setup_helper()
    bindings = tmp_path / "workspace-bindings.json"
    ordinary = tmp_path / "notes"
    ordinary.mkdir()
    target = SimpleNamespace(origin="https://cloud.example")
    monkeypatch.setattr(helper, "workspace_bindings_path", lambda: bindings)
    monkeypatch.setattr(helper, "configured_target", lambda: target)
    monkeypatch.setattr(helper, "_validate_workspace", lambda _workspace: target)
    args = SimpleNamespace(
        operation="bind",
        client="codex",
        path=str(ordinary),
        workspace="notes_workspace",
        apply=False,
        expect_digest=None,
    )

    preview = helper._mutate(args)

    assert preview["binding_kind"] == "directory"
    assert preview["planned_document"]["targets"][target.origin]["directory_bindings"] == {
        str(ordinary): "notes_workspace"
    }
    assert not bindings.exists()

    args.apply = True
    args.expect_digest = preview["document_digest"]
    helper._mutate(args)

    assert load_workspace_bindings(bindings).targets[target.origin].directory_bindings == {ordinary: "notes_workspace"}
    assert stat.S_IMODE(bindings.stat().st_mode) == 0o600


def test_setup_helper_uses_normalized_origin_for_git_repository(
    monkeypatch,
    tmp_path: Path,
) -> None:
    helper = _load_setup_helper()
    bindings = tmp_path / "workspace-bindings.json"
    repository = tmp_path / "repository"
    repository.mkdir()
    _init_git_repo(repository, "git@github.com:acme/payroll.git")
    target = SimpleNamespace(origin="https://cloud.example")
    monkeypatch.setattr(helper, "workspace_bindings_path", lambda: bindings)
    monkeypatch.setattr(helper, "configured_target", lambda: target)
    monkeypatch.setattr(helper, "_validate_workspace", lambda _workspace: target)
    args = SimpleNamespace(
        operation="bind",
        client="codex",
        path=str(repository),
        workspace="payroll_workspace",
        apply=False,
        expect_digest=None,
    )

    preview = helper._mutate(args)

    assert preview["binding_kind"] == "repository"
    assert preview["binding_key"] == "github.com/acme/payroll"


def test_setup_helper_inspect_reports_obsolete_keys(monkeypatch, tmp_path: Path) -> None:
    helper = _load_setup_helper()
    bindings = tmp_path / "workspace-bindings.json"
    ordinary = tmp_path / "notes"
    ordinary.mkdir()
    bindings.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    "https://cloud.example": {
                        "hook_workspace_id": "capture_workspace",
                        "directory_bindings": {str(ordinary): "notes_workspace"},
                    },
                    "http://127.0.0.1:8765": {"hook_workspace_id": "default"},
                },
            }
        ),
        encoding="utf-8",
    )
    target = SimpleNamespace(origin="https://cloud.example")
    monkeypatch.setattr(helper, "workspace_bindings_path", lambda: bindings)
    monkeypatch.setattr(helper, "configured_target", lambda: target)
    monkeypatch.setattr(
        helper,
        "resolve_workspace_binding",
        lambda **kwargs: resolve_workspace_binding(config_path=bindings, **kwargs),
    )

    report = helper._inspect(SimpleNamespace(operation="inspect", client="codex", path=str(ordinary)))

    assert report["effective_workspace_id"] == "notes_workspace"
    assert report["effective_source"] == "directory"
    assert report["obsolete_keys"] == [
        {"origin": "https://cloud.example", "key": "hook_workspace_id", "value": "capture_workspace"},
        {"origin": "http://127.0.0.1:8765", "key": "hook_workspace_id", "value": "default"},
    ]


def test_setup_helper_removes_obsolete_keys_only_after_confirmed_preview(monkeypatch, tmp_path: Path) -> None:
    helper = _load_setup_helper()
    bindings = tmp_path / "workspace-bindings.json"
    ordinary = tmp_path / "notes"
    ordinary.mkdir()
    original = {
        "version": 1,
        "targets": {
            "https://cloud.example": {
                "hook_workspace_id": "capture_workspace",
                "directory_bindings": {str(ordinary): "notes_workspace"},
            },
            "http://127.0.0.1:8765": {"hook_workspace_id": "default"},
        },
    }
    bindings.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(helper, "workspace_bindings_path", lambda: bindings)
    monkeypatch.setattr(helper, "configured_target", lambda: SimpleNamespace(origin="https://cloud.example"))
    args = SimpleNamespace(operation="remove-obsolete-keys", client="codex", apply=False, expect_digest=None)

    preview = helper._mutate(args)

    assert [entry["origin"] for entry in preview["removed_obsolete_keys"]] == [
        "https://cloud.example",
        "http://127.0.0.1:8765",
    ]
    assert preview["planned_document"] == {
        "version": 1,
        "targets": {"https://cloud.example": {"directory_bindings": {str(ordinary): "notes_workspace"}}},
    }
    assert json.loads(bindings.read_text(encoding="utf-8")) == original

    args.apply = True
    args.expect_digest = "stale-digest"
    with pytest.raises(helper.SetupError, match="changed after preview"):
        helper._mutate(args)
    assert json.loads(bindings.read_text(encoding="utf-8")) == original

    args.expect_digest = preview["document_digest"]
    helper._mutate(args)

    assert json.loads(bindings.read_text(encoding="utf-8")) == preview["planned_document"]
    args.apply = False
    with pytest.raises(helper.SetupError, match="no obsolete keys"):
        helper._mutate(args)


def test_setup_helper_no_longer_offers_a_hook_workspace() -> None:
    helper = _load_setup_helper()

    with pytest.raises(SystemExit):
        helper._parser().parse_args(["--client", "codex", "set-hook-workspace", "--workspace", "w"])
