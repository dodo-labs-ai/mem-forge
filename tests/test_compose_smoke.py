"""The compose smoke helper's offline checks: port bindings, waiting, and the MCP tool set it expects."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compose_smoke.py"
PLUGIN = ROOT / "integrations" / "claude-code" / "memforge-memory"


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("compose_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def _container(service: str, *bindings: tuple[str, str]) -> dict:
    return {
        "Config": {"Labels": {"com.docker.compose.service": service}},
        "NetworkSettings": {
            "Ports": {"8765/tcp": [{"HostIp": host_ip, "HostPort": port} for host_ip, port in bindings]}
        },
    }


def test_loopback_bindings_pass() -> None:
    inspected = [_container("api", ("127.0.0.1", "18765")), _container("admin-ui", ("127.0.0.1", "15174"))]

    detail = smoke.check_bindings(inspected, {"api": 18765, "admin-ui": 15174})

    assert detail == "admin-ui 127.0.0.1:15174; api 127.0.0.1:18765"


@pytest.mark.parametrize("host_ip", ["0.0.0.0", "", "::"])
def test_a_port_published_beyond_loopback_fails(host_ip: str) -> None:
    inspected = [_container("api", ("127.0.0.1", "18765"), (host_ip, "18765"))]

    with pytest.raises(smoke.CheckFailed, match="published beyond loopback"):
        smoke.check_bindings(inspected, {"api": 18765})


def test_a_service_without_its_port_fails() -> None:
    with pytest.raises(smoke.CheckFailed, match="not published: admin-ui 127.0.0.1:15174"):
        smoke.check_bindings([_container("api", ("127.0.0.1", "18765"))], {"api": 18765, "admin-ui": 15174})


def test_wait_until_gives_up_at_the_deadline() -> None:
    with pytest.raises(smoke.CheckFailed, match="the sync did not happen"):
        smoke.wait_until(lambda: None, timeout=0, interval=0, what="the sync")


def test_free_ports_are_distinct() -> None:
    ports = smoke.free_ports(2)

    assert len(set(ports)) == 2


def test_the_packaged_proxy_lists_the_expected_tools(tmp_path: Path) -> None:
    with smoke.McpSession(PLUGIN, "http://127.0.0.1:9", tmp_path / "home", timeout=30) as session:
        session.start()
        tools = {tool["name"] for tool in session.call("tools/list")["tools"]}

    assert smoke.EXPECTED_MCP_TOOLS <= tools
