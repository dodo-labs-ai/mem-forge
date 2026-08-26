from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from memforge.api_target import (
    Edition,
    TargetConfigurationError,
    build_target,
    capability_document,
    target_from_capability_document,
)
from memforge.capability_discovery import discover_target
from memforge.plugin_config import configured_api_token, configured_target


@pytest.fixture(autouse=True)
def _clear_configured_target_cache():
    from memforge import plugin_config

    plugin_config._target_for_configuration.cache_clear()
    yield
    plugin_config._target_for_configuration.cache_clear()


def test_unified_v1_target_does_not_require_a_configured_workspace():
    target = build_target(
        origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
        edition=Edition.CLOUD,
    )

    assert target.api_base == (
        "https://memforge-dev.cfapps.eu12.hana.ondemand.com/api/v1"
    )
    assert target.resource_url("/memories/search") == (
        "https://memforge-dev.cfapps.eu12.hana.ondemand.com/api/v1/memories/search"
    )


def test_unified_v1_target_adds_explicit_workspace_as_query_parameter():
    target = build_target(origin="https://self.example", edition=Edition.OSS)

    assert target.resource_url(
        "/memories/search?include_private=true",
        workspace_id="mount tai/blue",
    ) == (
        "https://self.example/api/v1/memories/search"
        "?include_private=true&workspace_id=mount+tai%2Fblue"
    )


def test_resource_url_rejects_an_embedded_workspace_selector():
    target = build_target(origin="https://self.example", edition=Edition.OSS)

    with pytest.raises(ValueError, match="workspace_id_must_use_explicit_parameter"):
        target.resource_url("/memories?workspace_id=ws-a")


@pytest.mark.parametrize(
    ("origin", "edition"),
    [
        ("https://self.example", Edition.OSS),
        (
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            Edition.CLOUD,
        ),
    ],
)
def test_build_target_uses_explicit_edition_without_changing_the_v1_contract(
    origin: str,
    edition: Edition,
):
    target = build_target(origin=origin, edition=edition)

    assert target.edition is edition
    assert target.api_base == f"{origin}/api/v1"


def test_build_target_defaults_to_local_oss():
    target = build_target(origin=None)

    assert target.edition is Edition.OSS
    assert target.origin == "http://127.0.0.1:8765"
    assert target.health_path == "/api/v1/health"
    assert target.authentication_required is False


def test_remote_target_requires_explicit_or_discovered_edition():
    with pytest.raises(TargetConfigurationError, match="memforge_edition_required"):
        build_target(origin="https://memforge.example")


def test_capability_document_defines_edition_auth_and_health_contract():
    assert capability_document(Edition.OSS) == {
        "protocol": "memforge",
        "protocol_version": 1,
        "edition": "oss",
        "api_base": "/api/v1",
        "health_path": "/api/v1/health",
        "authentication": {"required": False, "scheme": "none"},
    }
    assert capability_document(Edition.CLOUD) == {
        "protocol": "memforge",
        "protocol_version": 1,
        "edition": "cloud",
        "api_base": "/api/v1",
        "health_path": "/healthz",
        "authentication": {"required": True, "scheme": "bearer"},
    }


def test_discovery_uses_service_capability_instead_of_dns_suffix():
    cloud_custom_domain = discover_target(
        "https://memory.example.com",
        fetcher=lambda url: capability_document(Edition.CLOUD),
    )
    oss_on_hana_domain = discover_target(
        "https://self-hosted.hana.ondemand.com",
        fetcher=lambda url: capability_document(Edition.OSS),
    )

    assert cloud_custom_domain.edition is Edition.CLOUD
    assert cloud_custom_domain.authentication_required is True
    assert oss_on_hana_domain.edition is Edition.OSS
    assert oss_on_hana_domain.authentication_required is False


def test_capability_parser_rejects_unsupported_or_drifted_contract():
    unsupported = capability_document(Edition.CLOUD)
    unsupported["protocol_version"] = 2
    with pytest.raises(TargetConfigurationError, match="memforge_capability_version_unsupported"):
        target_from_capability_document(origin="https://memory.example", document=unsupported)

    drifted = capability_document(Edition.CLOUD)
    drifted["health_path"] = "/api/v1/health"
    with pytest.raises(TargetConfigurationError, match="memforge_capability_contract_invalid"):
        target_from_capability_document(origin="https://memory.example", document=drifted)


@pytest.mark.parametrize(
    "origin",
    [
        "https://user:pass@self.example",
        "https://:443",
        "https://self.example:notaport",
        "https://self.example/custom",
        "https://self.example/api/v1",
        "https://self.example?api=v1",
        "https://self.example#api",
    ],
)
def test_build_target_rejects_non_origin_api_urls(origin: str):
    with pytest.raises(TargetConfigurationError, match="memforge_origin_required"):
        build_target(origin=origin, edition=Edition.OSS)


@pytest.mark.parametrize("origin", ["https://self.example:8443", "https://[::1]:8443"])
def test_build_target_accepts_normal_host_and_ipv6_origins(origin: str):
    assert build_target(origin=origin, edition=Edition.OSS).origin == origin


def test_memforge_target_is_immutable():
    target = build_target(origin="https://self.example", edition=Edition.OSS)

    with pytest.raises(FrozenInstanceError):
        target.origin = "https://other.example"


@pytest.mark.parametrize("relative_path", ["sources", "/api/v1/sources"])
def test_resource_url_rejects_paths_outside_api_base(relative_path: str):
    target = build_target(origin="https://self.example", edition=Edition.OSS)

    with pytest.raises(ValueError, match="resource_path_must_be_relative_to_api_base"):
        target.resource_url(relative_path)


def test_configured_target_reads_origin_but_ignores_obsolete_workspace_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    repository = tmp_path / "repository"
    (repository / ".memforge").mkdir(parents=True)
    (repository / ".memforge" / "config.toml").write_text(
        '[memforge]\nworkspace_id = "repository_workspace"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MEMFORGE_API_URL",
        "https://memforge-dev.cfapps.eu12.hana.ondemand.com/",
    )
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "obsolete-process-workspace")
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.chdir(repository)
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    target = configured_target()
    assert target.edition is Edition.CLOUD
    assert target.origin == "https://memforge-dev.cfapps.eu12.hana.ondemand.com"


def test_configured_target_preserves_zero_configuration_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    for name in ("MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID", "MEMFORGE_EDITION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    target = configured_target()
    assert target.edition is Edition.OSS
    assert target.origin == "http://127.0.0.1:8765"


def test_configured_target_discovers_custom_domain_without_dns_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memory.example.com")
    monkeypatch.delenv("MEMFORGE_EDITION", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(
        plugin_config,
        "discover_target",
        lambda origin: build_target(origin=origin, edition=Edition.CLOUD),
    )

    target = configured_target()

    assert target.edition is Edition.CLOUD
    assert target.origin == "https://memory.example.com"


def test_environment_url_does_not_inherit_edition_from_codex_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    user_config = tmp_path / "codex-config.toml"
    user_config.write_text(
        """
[memforge]
MEMFORGE_API_URL = "https://configured.example"
MEMFORGE_EDITION = "cloud"
""",
        encoding="utf-8",
    )
    discovered: list[str] = []
    monkeypatch.setenv("MEMFORGE_API_URL", "https://environment.example")
    monkeypatch.delenv("MEMFORGE_EDITION", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(user_config))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(
        plugin_config,
        "discover_target",
        lambda origin: (
            discovered.append(origin)
            or build_target(origin=origin, edition=Edition.OSS)
        ),
    )

    target = configured_target()

    assert target.edition is Edition.OSS
    assert discovered == ["https://environment.example"]


def test_configured_target_caches_discovery_for_the_exact_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    calls: list[str] = []
    monkeypatch.setenv("MEMFORGE_API_URL", "https://memory.example.com")
    monkeypatch.delenv("MEMFORGE_EDITION", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(
        plugin_config,
        "discover_target",
        lambda origin: (
            calls.append(origin)
            or build_target(origin=origin, edition=Edition.CLOUD)
        ),
    )

    first = configured_target()
    second = configured_target()

    assert first is second
    assert calls == ["https://memory.example.com"]


def test_repository_config_does_not_override_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from memforge import plugin_config

    user_config = tmp_path / "codex-config.toml"
    user_config.write_text(
        '[memforge]\nMEMFORGE_API_TOKEN = "user-token"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(user_config))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    assert configured_api_token() == "user-token"


def test_plugin_config_package_import_does_not_hide_api_target_import_error(
    tmp_path: Path,
):
    package = tmp_path / "broken_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api_target.py").write_text(
        'raise ImportError("internal_api_target_defect")\n',
        encoding="utf-8",
    )
    source = Path(__file__).parents[1] / "src" / "memforge" / "plugin_config.py"
    (package / "plugin_config.py").write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), env.get("PYTHONPATH")))
    )

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
