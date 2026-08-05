from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPOSITORY_ROOT / "compose.yaml"
NETWORK_BIND_VARIABLES = (
    "MEMFORGE_API_BIND_ADDR",
    "MEMFORGE_ADMIN_UI_BIND_ADDR",
)


def _render_compose(extra_environment: dict[str, str] | None = None) -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render the self-hosted deployment contract")

    compose_version = subprocess.run(
        [docker, "compose", "version"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if compose_version.returncode != 0:
        pytest.skip("Docker Compose v2 is required to render the self-hosted deployment contract")

    environment = os.environ.copy()
    for variable in NETWORK_BIND_VARIABLES:
        environment.pop(variable, None)
    environment.update(extra_environment or {})

    rendered = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            os.devnull,
            "--file",
            str(COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(rendered.stdout)


def _published_host_ip(rendered: dict, service_name: str) -> str | None:
    ports = rendered["services"][service_name]["ports"]
    assert len(ports) == 1
    return ports[0].get("host_ip")


def test_default_compose_publishes_api_and_admin_ui_on_loopback():
    rendered = _render_compose()

    assert _published_host_ip(rendered, "api") == "127.0.0.1"
    assert _published_host_ip(rendered, "admin-ui") == "127.0.0.1"


def test_compose_does_not_allow_bind_address_environment_overrides():
    rendered = _render_compose(
        {
            "MEMFORGE_API_BIND_ADDR": "0.0.0.0",
            "MEMFORGE_ADMIN_UI_BIND_ADDR": "0.0.0.0",
        }
    )

    assert _published_host_ip(rendered, "api") == "127.0.0.1"
    assert _published_host_ip(rendered, "admin-ui") == "127.0.0.1"
