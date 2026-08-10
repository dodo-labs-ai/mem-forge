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


def test_hook_workspace_is_never_used_by_interactive_resolution(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "ordinary"
    working_directory.mkdir()
    config = tmp_path / "bindings.json"
    _write_config(config, {"hook_workspace_id": "capture_workspace"})

    interactive = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(working_directory),
        allow_hook_default=False,
        config_path=config,
    )
    hook = resolve_workspace_binding(
        origin=ORIGIN,
        working_directory=str(working_directory),
        allow_hook_default=True,
        config_path=config,
    )

    assert interactive.workspace_id is None
    assert interactive.source == "none"
    assert hook.workspace_id == "capture_workspace"
    assert hook.source == "hook_default"


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
