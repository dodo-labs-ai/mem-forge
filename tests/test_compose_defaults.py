from pathlib import Path

import yaml


def test_admin_ui_is_published_on_loopback_by_default():
    compose_path = Path(__file__).resolve().parents[1] / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert compose["services"]["admin-ui"]["ports"] == [
        "127.0.0.1:${MEMFORGE_ADMIN_UI_HOST_PORT:-5174}:8080"
    ]
