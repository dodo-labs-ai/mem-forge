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


QUICK = ROOT / "evals" / "quality" / "quick"


def _current_titles() -> dict[str, str]:
    """Each synced document's title after the smoke replaces it with its revision, by file name."""
    import re

    titles = {}
    for path in (QUICK / "docs").glob("*.md"):
        current = QUICK / "revisions" / path.name
        text = (current if current.exists() else path).read_text(encoding="utf-8")
        titles[path.name] = re.match(r"# (.+)", text).group(1)
    for title in re.findall(r"^# (.+)$", (ROOT / "scripts" / "compose_smoke.sh").read_text(), re.MULTILINE):
        titles[title] = title
    return titles


def test_every_quality_case_names_a_document_the_smoke_syncs() -> None:
    import json

    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    titles = set(_current_titles().values())

    for list_name, _command in smoke.QUALITY_CASE_KINDS.values():
        assert {case["title"] for case in cohort[list_name]} <= titles, list_name
    assert len({case["title"] for case in cohort["cases"]}) == len(cohort["cases"])


def test_every_revised_document_has_its_support_case() -> None:
    import json

    revised = {_current_titles()[path.name] for path in (QUICK / "revisions").glob("*.md")}
    support_titles = {case["title"] for case in json.loads(COHORT.read_text(encoding="utf-8"))["support_cases"]}

    assert revised == support_titles
    assert all((QUICK / "docs" / path.name).exists() for path in (QUICK / "revisions").glob("*.md"))


@pytest.mark.parametrize("kind", tuple(smoke.QUALITY_CASE_KINDS))
def test_the_seed_request_names_each_document_within_the_smoke_source(tmp_path: Path, kind: str) -> None:
    import json

    state = smoke.State(tmp_path / "state.json")
    state.set(source_id="src-smoke")

    spec = json.loads(smoke.quality_seed_spec(COHORT, state, kind))

    assert (spec["selection_policy_version"], spec["role"]) == (f"public-quick-v1-{kind}", "sentinel")
    assert spec["cases"] and {case["source_id"] for case in spec["cases"]} == {"src-smoke"}
    if kind == "reconciliation":
        assert all(set(case) == {"title", "source_id", "input", "rubric"} for case in spec["cases"])
    with pytest.raises(smoke.CheckFailed, match="no source"):
        smoke.quality_seed_spec(COHORT, smoke.State(tmp_path / "empty.json"), kind)


def test_a_revision_sync_waits_past_the_run_that_finished_before(tmp_path: Path, monkeypatch) -> None:
    runs = iter(["run-1", "run-1", "run-2"])
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(smoke, "_source", lambda _api, _source_id: {
        "sync": {"run_id": next(runs, "run-2"), "status": "success"}, "doc_count": 6, "memory_count": 11,
    })
    state = smoke.State(tmp_path / "state.json")
    state.set(source_id="src-smoke", sync_run_id="run-1")

    smoke.wait_for_sync("http://api", state, timeout=5, new_run=True)

    assert state.get("sync_run_id") == "run-2"
    with pytest.raises(smoke.CheckFailed, match="no earlier sync run"):
        smoke.wait_for_sync("http://api", smoke.State(tmp_path / "fresh.json"), timeout=5, new_run=True)


def test_the_run_spec_needs_every_case_pinned() -> None:
    import json

    spec = json.loads(smoke.quality_run_spec(COHORT, {"cohort_id": "aeo-1", "skipped": {"document_not_found": []}}))

    assert spec["cohort_id"] == "aeo-1"
    assert len(spec["candidate_manifest"]["prompt_hash"]) == 64
    assert "code_revision" not in spec["candidate_manifest"] and "model" not in spec["candidate_manifest"]
    with pytest.raises(smoke.CheckFailed, match="not pinned completely"):
        smoke.quality_run_spec(COHORT, {"cohort_id": "aeo-1", "skipped": {"document_not_found": ["src:Missing"]}})
