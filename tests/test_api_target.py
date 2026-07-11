from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from memforge.api_target import (
    Edition,
    MemForgeTarget,
    TargetConfigurationError,
    build_host_target,
    build_target,
)
from memforge.plugin_config import configured_target


@pytest.mark.parametrize(
    ("origin", "workspace_id", "edition", "resource_base"),
    [
        ("https://self.example", None, Edition.OSS, "https://self.example/api"),
        (
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "mount_tai",
            Edition.CLOUD,
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com/api/workspaces/mount_tai/api",
        ),
    ],
)
def test_build_target_derives_edition_and_returns_one_resource_base(origin, workspace_id, edition, resource_base):
    target = build_target(origin=origin, workspace_id=workspace_id)

    assert target.edition is edition
    assert target.workspace_api_base == resource_base


@pytest.mark.parametrize(
    ("origin", "workspace_id", "code"),
    [
        (None, "mount_tai", "memforge_origin_required"),
        ("https://memforge-dev.cfapps.eu12.hana.ondemand.com", None, "cloud_workspace_required"),
        ("https://self.example", "mount_tai", "workspace_not_supported_for_oss"),
        ("https://self.example/api", None, "memforge_origin_required"),
    ],
)
def test_build_target_rejects_invalid_derived_target(origin, workspace_id, code):
    with pytest.raises(TargetConfigurationError, match=code) as exc_info:
        build_target(origin=origin, workspace_id=workspace_id)

    assert exc_info.value.code == code


def test_build_target_defaults_only_when_all_configuration_is_absent():
    target = build_target(origin=None, workspace_id=None)

    assert target == MemForgeTarget(
        edition=Edition.OSS,
        origin="http://127.0.0.1:8765",
        workspace_id=None,
    )
    assert target.workspace_api_base == "http://127.0.0.1:8765/api"


def test_build_host_target_allows_cloud_control_plane_without_workspace():
    target = build_host_target(
        origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com/",
    )

    assert target == MemForgeTarget(
        edition=Edition.CLOUD,
        origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
        workspace_id=None,
    )


def test_build_target_normalizes_origin_and_quotes_workspace():
    target = build_target(
        origin=" https://MEMFORGE-DEV.CFAPPS.EU12.HANA.ONDEMAND.COM/ ",
        workspace_id=" mount tai/blue ",
    )

    assert target.edition is Edition.CLOUD
    assert target.origin == "https://MEMFORGE-DEV.CFAPPS.EU12.HANA.ONDEMAND.COM"
    assert target.workspace_id == "mount tai/blue"
    assert target.workspace_api_base == (
        "https://MEMFORGE-DEV.CFAPPS.EU12.HANA.ONDEMAND.COM/api/workspaces/mount%20tai%2Fblue/api"
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://user:pass@self.example",
        "https://:443",
        "https://self.example:notaport",
        "https://self.example/custom",
        "https://self.example/api/workspaces/mount_tai/api",
        "https://self.example?api=v1",
        "https://self.example#api",
    ],
)
def test_build_target_rejects_non_origin_api_urls(origin):
    with pytest.raises(TargetConfigurationError, match="memforge_origin_required"):
        build_target(origin=origin, workspace_id=None)


@pytest.mark.parametrize("origin", ["https://self.example:8443", "https://[::1]:8443"])
def test_build_target_accepts_normal_host_and_ipv6_origins(origin):
    target = build_target(origin=origin, workspace_id=None)

    assert target.origin == origin


@pytest.mark.parametrize(
    "origin",
    [
        "https://hana.ondemand.com",
        "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
        "https://MEMFORGE-DEV.CFAPPS.EU12.HANA.ONDEMAND.COM.",
    ],
)
def test_build_target_classifies_only_hana_ondemand_hosts_as_cloud(origin):
    target = build_target(origin=origin, workspace_id="mount_tai")

    assert target.edition is Edition.CLOUD


@pytest.mark.parametrize(
    "origin",
    [
        "https://hana.ondemand.com.evil.example",
        "https://hana-ondemand.com",
        "https://self.example",
    ],
)
def test_build_target_does_not_use_substring_cloud_detection(origin):
    target = build_target(origin=origin, workspace_id=None)

    assert target.edition is Edition.OSS


def test_memforge_target_is_immutable():
    target = build_target(origin="https://self.example", workspace_id=None)

    with pytest.raises(FrozenInstanceError):
        target.origin = "https://other.example"


def test_resource_url_resolves_relative_resource_path():
    target = build_target(
        origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
        workspace_id="mount_tai",
    )

    assert target.resource_url("/sources") == (
        "https://memforge-dev.cfapps.eu12.hana.ondemand.com/api/workspaces/mount_tai/api/sources"
    )


@pytest.mark.parametrize("relative_path", ["sources", "/api/sources"])
def test_resource_url_rejects_paths_outside_api_base(relative_path):
    target = build_target(origin="https://self.example", workspace_id=None)

    with pytest.raises(ValueError, match="resource_path_must_be_relative_to_api_base"):
        target.resource_url(relative_path)


def test_configured_target_reads_explicit_cloud_target(monkeypatch, tmp_path):
    from memforge import plugin_config

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge-dev.cfapps.eu12.hana.ondemand.com/")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    assert configured_target() == MemForgeTarget(
        edition=Edition.CLOUD,
        origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
        workspace_id="mount_tai",
    )


def test_configured_target_preserves_zero_configuration_default(monkeypatch, tmp_path):
    from memforge import plugin_config

    for name in ("MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    assert configured_target() == MemForgeTarget(
        edition=Edition.OSS,
        origin="http://127.0.0.1:8765",
        workspace_id=None,
    )


def test_plugin_config_package_import_does_not_hide_api_target_import_error(tmp_path):
    package = tmp_path / "broken_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api_target.py").write_text(
        'raise ImportError("internal_api_target_defect")\n',
        encoding="utf-8",
    )
    source = Path(__file__).parents[1] / "src" / "memforge" / "plugin_config.py"
    (package / "plugin_config.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, "-c", "import broken_plugin.plugin_config"],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "internal_api_target_defect" in result.stderr
