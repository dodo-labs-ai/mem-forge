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


COHORT = ROOT / "evals" / "quality" / "quick" / "cohort.json"


def test_every_quality_case_names_a_document_the_smoke_syncs() -> None:
    import json
    import re

    titles = {
        match.group(1)
        for path in (ROOT / "evals" / "quality" / "quick" / "docs").glob("*.md")
        for match in [re.match(r"# (.+)", path.read_text(encoding="utf-8"))]
        if match
    }
    titles |= set(re.findall(r"^# (.+)$", (ROOT / "scripts" / "compose_smoke.sh").read_text(), re.MULTILINE))

    cases = json.loads(COHORT.read_text(encoding="utf-8"))["cases"]

    assert {case["title"] for case in cases} <= titles
    assert len({case["title"] for case in cases}) == len(cases)


def test_the_seed_request_names_each_document_within_the_smoke_source(tmp_path: Path) -> None:
    import json

    state = smoke.State(tmp_path / "state.json")
    state.set(source_id="src-smoke")

    spec = json.loads(smoke.quality_seed_spec(COHORT, state))

    assert (spec["selection_policy_version"], spec["role"]) == ("public-quick-v1", "sentinel")
    assert {case["source_id"] for case in spec["cases"]} == {"src-smoke"}
    with pytest.raises(smoke.CheckFailed, match="no source"):
        smoke.quality_seed_spec(COHORT, smoke.State(tmp_path / "empty.json"))


def test_the_run_spec_needs_every_case_pinned() -> None:
    import json

    spec = json.loads(smoke.quality_run_spec(COHORT, {"cohort_id": "aeo-1", "skipped": {"document_not_found": []}}))

    assert spec["cohort_id"] == "aeo-1"
    assert len(spec["candidate_manifest"]["prompt_hash"]) == 64
    assert "code_revision" not in spec["candidate_manifest"] and "model" not in spec["candidate_manifest"]
    with pytest.raises(smoke.CheckFailed, match="not pinned completely"):
        smoke.quality_run_spec(COHORT, {"cohort_id": "aeo-1", "skipped": {"document_not_found": ["src:Missing"]}})
