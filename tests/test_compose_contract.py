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
IMAGE_VARIABLES = (
    "MEMFORGE_IMAGE_REGISTRY",
    "MEMFORGE_IMAGE_TAG",
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
    for variable in (*NETWORK_BIND_VARIABLES, *IMAGE_VARIABLES):
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


def _image_name_and_tag(rendered: dict, service_name: str) -> tuple[str, str]:
    name, tag = rendered["services"][service_name]["image"].rsplit(":", 1)
    return name, tag


def test_each_service_names_its_published_image_and_keeps_its_build():
    rendered = _render_compose()

    assert _image_name_and_tag(rendered, "api")[0] == "ghcr.io/dodo-labs-ai/memforge-api"
    assert _image_name_and_tag(rendered, "admin-ui")[0] == "ghcr.io/dodo-labs-ai/memforge-admin-ui"
    assert rendered["services"]["api"]["build"]["dockerfile"] == "Dockerfile"
    assert rendered["services"]["admin-ui"]["build"]["dockerfile"] == "admin/Dockerfile"


def test_the_build_keeps_its_mirror_arguments_beside_the_image():
    rendered = _render_compose({"MEMFORGE_DOCKERHUB_PREFIX": "mirror.example.com/library/"})

    for service_name in ("api", "admin-ui"):
        assert rendered["services"][service_name]["build"]["args"]["DOCKERHUB_PREFIX"] == "mirror.example.com/library/"


def test_the_registry_setting_moves_only_the_registry():
    default = _render_compose()
    rendered = _render_compose({"MEMFORGE_IMAGE_REGISTRY": "mirror.example.com/acme"})

    for service_name, image in (("api", "memforge-api"), ("admin-ui", "memforge-admin-ui")):
        name, tag = _image_name_and_tag(rendered, service_name)
        assert name == f"mirror.example.com/acme/{image}"
        assert tag == _image_name_and_tag(default, service_name)[1]


def test_an_empty_registry_setting_keeps_the_default_registry():
    rendered = _render_compose({"MEMFORGE_IMAGE_REGISTRY": ""})

    assert _image_name_and_tag(rendered, "api")[0] == "ghcr.io/dodo-labs-ai/memforge-api"
    assert _image_name_and_tag(rendered, "admin-ui")[0] == "ghcr.io/dodo-labs-ai/memforge-admin-ui"


def test_the_environment_cannot_detach_a_checkout_from_its_image_tag():
    default = _render_compose()
    rendered = _render_compose({"MEMFORGE_IMAGE_TAG": "9.9.9"})

    for service_name in ("api", "admin-ui"):
        assert _image_name_and_tag(rendered, service_name)[1] == _image_name_and_tag(default, service_name)[1]
