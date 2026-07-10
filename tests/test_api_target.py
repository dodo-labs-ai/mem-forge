from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from memforge.api_target import Edition, MemForgeTarget, TargetConfigurationError, build_target
from memforge.plugin_config import configured_target


@pytest.mark.parametrize(
    ("edition", "origin", "workspace_id", "resource_base"),
    [
        ("oss", "https://self.example", None, "https://self.example/api"),
        (
            "cloud",
            "https://cloud.example",
            "mount_tai",
            "https://cloud.example/api/workspaces/mount_tai/api",
        ),
    ],
)
def test_build_target_returns_one_resource_base(edition, origin, workspace_id, resource_base):
    target = build_target(edition=edition, origin=origin, workspace_id=workspace_id)

    assert target.workspace_api_base == resource_base


@pytest.mark.parametrize(
    ("edition", "origin", "workspace_id", "code"),
    [
        (None, "https://self.example", None, "memforge_edition_required"),
        ("cloud", None, "mount_tai", "cloud_api_url_required"),
        ("cloud", "https://cloud.example", None, "cloud_workspace_required"),
        ("oss", "https://self.example", "mount_tai", "workspace_not_supported_for_oss"),
        ("oss", "https://self.example/api", None, "memforge_origin_required"),
        ("invalid", "https://self.example", None, "invalid_memforge_edition"),
        ("oss", None, None, "memforge_origin_required"),
    ],
)
def test_build_target_rejects_invalid_tagged_union(edition, origin, workspace_id, code):
    with pytest.raises(TargetConfigurationError, match=code) as exc_info:
        build_target(edition=edition, origin=origin, workspace_id=workspace_id)

    assert exc_info.value.code == code


def test_build_target_defaults_only_when_all_configuration_is_absent():
    target = build_target(edition=None, origin=None, workspace_id=None)

    assert target == MemForgeTarget(
        edition=Edition.OSS,
        origin="http://127.0.0.1:8765",
        workspace_id=None,
    )
    assert target.workspace_api_base == "http://127.0.0.1:8765/api"


def test_build_target_normalizes_origin_and_quotes_workspace():
    target = build_target(
        edition=" cloud ",
        origin=" https://cloud.example/ ",
        workspace_id=" mount tai/blue ",
    )

    assert target.origin == "https://cloud.example"
    assert target.workspace_id == "mount tai/blue"
    assert target.workspace_api_base == (
        "https://cloud.example/api/workspaces/mount%20tai%2Fblue/api"
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://self.example/custom",
        "https://self.example/api/workspaces/mount_tai/api",
        "https://self.example?api=v1",
        "https://self.example#api",
    ],
)
def test_build_target_rejects_non_origin_api_urls(origin):
    with pytest.raises(TargetConfigurationError, match="memforge_origin_required"):
        build_target(edition="oss", origin=origin, workspace_id=None)


def test_memforge_target_is_immutable():
    target = build_target(edition="oss", origin="https://self.example", workspace_id=None)

    with pytest.raises(FrozenInstanceError):
        target.origin = "https://other.example"


def test_resource_url_resolves_relative_resource_path():
    target = build_target(
        edition="cloud",
        origin="https://cloud.example",
        workspace_id="mount_tai",
    )

    assert target.resource_url("/sources") == (
        "https://cloud.example/api/workspaces/mount_tai/api/sources"
    )


@pytest.mark.parametrize("relative_path", ["sources", "/api/sources"])
def test_resource_url_rejects_paths_outside_api_base(relative_path):
    target = build_target(edition="oss", origin="https://self.example", workspace_id=None)

    with pytest.raises(ValueError, match="resource_path_must_be_relative_to_api_base"):
        target.resource_url(relative_path)


def test_configured_target_reads_explicit_cloud_target(monkeypatch, tmp_path):
    from memforge import plugin_config

    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example/")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    assert configured_target() == MemForgeTarget(
        edition=Edition.CLOUD,
        origin="https://cloud.example",
        workspace_id="mount_tai",
    )


def test_configured_target_preserves_zero_configuration_default(monkeypatch, tmp_path):
    from memforge import plugin_config

    for name in ("MEMFORGE_EDITION", "MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    assert configured_target() == MemForgeTarget(
        edition=Edition.OSS,
        origin="http://127.0.0.1:8765",
        workspace_id=None,
    )
