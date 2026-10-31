"""The compose smoke helper's offline checks: port bindings, waiting, the model plan and the MCP tools it expects."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from dataclasses import replace
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
REPLAYED_AT = "2026-10-30T12:00:00+00:00"


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
        assert {case["title"] for case in cohort[list_name] if "snapshot" not in case} <= titles, list_name
    assert len({(case.get("snapshot"), case["title"]) for case in cohort["cases"]}) == len(cohort["cases"])


def test_every_revised_document_has_its_support_case() -> None:
    import json

    revised = {_current_titles()[path.name] for path in (QUICK / "revisions").glob("*.md")}
    support_titles = {
        case["title"]
        for case in json.loads(COHORT.read_text(encoding="utf-8"))["support_cases"]
        if "snapshot" not in case
    }

    assert revised == support_titles
    assert all((QUICK / "docs" / path.name).exists() for path in (QUICK / "revisions").glob("*.md"))


def test_every_hand_written_situation_is_in_the_cohort() -> None:
    import json

    listed = {case["file"] for case in json.loads(COHORT.read_text(encoding="utf-8"))["reconciliation_cases"]}

    assert listed == {f"reconciliation/{path.name}" for path in (QUICK / "reconciliation").glob("*.json")}


@pytest.mark.parametrize("kind", tuple(smoke.QUALITY_CASE_KINDS))
def test_the_seed_request_names_each_document_within_the_smoke_source(tmp_path: Path, kind: str) -> None:
    import json

    state = smoke.State(tmp_path / "state.json")
    state.set(source_id="src-smoke")
    sets = [path.name for path in (QUICK / "snapshots").iterdir() if path.is_dir()]
    state.set(snapshot_sources={name: {"id": f"src-{name}", "replayed_at": REPLAYED_AT} for name in sets})

    spec = json.loads(smoke.quality_seed_spec(COHORT, state, kind))

    assert (spec["selection_policy_version"], spec["role"]) == (f"public-quick-v1-{kind}", "sentinel")
    expected_sources = {"src-smoke"} | ({f"src-{name}" for name in sets} if kind != "reconciliation" else set())
    assert spec["cases"] and {case["source_id"] for case in spec["cases"]} == expected_sources
    assert all("snapshot" not in case for case in spec["cases"])
    if kind == "reconciliation":
        assert all(set(case) == {"title", "source_id", "input", "rubric"} for case in spec["cases"])
    with pytest.raises(smoke.CheckFailed, match="no source"):
        smoke.quality_seed_spec(COHORT, smoke.State(tmp_path / "empty.json"), kind)
    if kind != "reconciliation":
        unreplayed = smoke.State(tmp_path / "unreplayed.json")
        unreplayed.set(source_id="src-smoke")
        with pytest.raises(smoke.CheckFailed, match="which no source replays"):
            smoke.quality_seed_spec(COHORT, unreplayed, kind)


def test_a_conversation_case_names_its_unit_by_the_replayed_date() -> None:
    teams = QUICK / "snapshots" / "teams"

    # The set's newest message, 2026-10-21T10:40, moves nine whole days, to before 11:00 on the replay's day.
    assert smoke._replayed_title("Kestrel release crew thread from 2026-10-21T10:00", teams, REPLAYED_AT) == (
        "Kestrel release crew thread from 2026-10-30T10:00"
    )
    assert smoke._replayed_title(
        "Kestrel release crew thread from 2026-10-21T10:00", teams, "2026-10-30T12:00:00Z"
    ) == "Kestrel release crew thread from 2026-10-30T10:00"
    assert smoke._replayed_title("Kestrel queue tiers", QUICK / "snapshots" / "confluence", REPLAYED_AT) == (
        "Kestrel queue tiers"
    )


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


def test_model_usage_totals_the_tokens_of_each_logged_call_per_model() -> None:
    def call(operation: str, model: str, prompt: int | None, completion: int | None) -> str:
        payload = {"event": "structured_llm_call", "operation": operation, "model": model}
        if prompt is not None:
            payload |= {"prompt_tokens": prompt, "completion_tokens": completion}
        return f"api-1  | 2026-10-30 INFO memforge.llm.structured structured_llm_call {json.dumps(payload)}"

    haiku, luna = "openrouter/anthropic/claude-haiku-5.5", "openai/gpt-6-luna"
    lines = [
        call("derive", haiku, 1200, 300), "api-1  | unrelated", call("derive", haiku, 800, 200),
        call("support", haiku, None, None), call("expectation_judgment", luna, 500, 50),
    ]

    assert smoke.model_usage(lines) == (
        f"3 calls to {haiku}, 2000 input and 500 output tokens, 1 without reported usage (derive 2, support 1); "
        f"1 call to {luna}, 500 input and 50 output tokens (expectation_judgment 1)"
    )
    with pytest.raises(smoke.CheckFailed, match="no model call"):
        smoke.model_usage(["api-1  | unrelated"])


def test_quality_outputs_reads_the_case_outputs_of_the_run(monkeypatch) -> None:
    requests = []
    monkeypatch.setattr(smoke, "api_json", lambda api, method, path: requests.append((method, path)) or {"cases": []})

    assert json.loads(smoke.quality_outputs("http://api", "aer-1")) == {"cases": []}
    assert requests == [("GET", "/api/v1/agent-evaluations/runs/aer-1/case-outputs")]


# -- Model plan --------------------------------------------------------------------------------

SPLIT_OPTIONS = [
    "--generation-provider", "openai_compatible",
    "--generation-endpoint", "https://llm.example.com/v1",
    "--generation-model", "gpt-6-luna",
    "--generation-key-env", "TEAM_LLM_KEY",
    "--embedding-provider", "openrouter",
]


def _plan(*options: str, **environ: str):
    return smoke.plan_models(smoke.parse_args(["model-plan", *options]), environ)


def test_without_model_options_the_plan_is_openrouter_with_its_key_or_none() -> None:
    assert _plan() is None
    assert _plan(OPENROUTER_API_KEY="  ") is None

    plan = _plan(OPENROUTER_API_KEY="sk-or")

    assert plan.kinds == {
        "generation": smoke.KindPlan("openrouter", "anthropic/claude-haiku-5.5", None),
        "embedding": smoke.KindPlan("openrouter", "openai/text-embedding-3-small", None),
    }
    assert _plan("--generation-model", "openai/gpt-5-mini", OPENROUTER_API_KEY="sk-or").kinds[
        "generation"
    ] == smoke.KindPlan("openrouter", "openai/gpt-5-mini", None)


def test_ollama_serves_both_kinds_at_the_ollama_url_even_with_an_openrouter_key() -> None:
    plan = _plan("--ollama", OPENROUTER_API_KEY="sk-or")

    assert {kind: (item.provider, item.model, item.endpoint) for kind, item in plan.kinds.items()} == {
        "generation": ("ollama", "qwen3:0.6b", smoke.DEFAULT_OLLAMA_URL),
        "embedding": ("ollama", "nomic-embed-text", smoke.DEFAULT_OLLAMA_URL),
    }
    moved = _plan("--ollama", "--ollama-url", "http://gpu-box:11434/")
    assert {item.endpoint for item in moved.kinds.values()} == {"http://gpu-box:11434"}


def test_each_kind_takes_its_own_provider() -> None:
    plan = _plan(*SPLIT_OPTIONS, TEAM_LLM_KEY="sk-team", OPENROUTER_API_KEY="sk-or")

    assert plan.kinds == {
        "generation": smoke.KindPlan("openai_compatible", "gpt-6-luna", "https://llm.example.com/v1", "TEAM_LLM_KEY"),
        "embedding": smoke.KindPlan("openrouter", "openai/text-embedding-3-small", None),
    }
    assert plan.label == (
        "generation openai_compatible: gpt-6-luna; embedding openrouter: openai/text-embedding-3-small"
    )
    # A kind left unnamed takes the default provider, here OpenRouter.
    unnamed = _plan(*SPLIT_OPTIONS[:8], TEAM_LLM_KEY="sk-team", OPENROUTER_API_KEY="sk-or")
    assert unnamed.kinds["embedding"].provider == "openrouter"


def test_a_direct_anthropic_key_reaches_the_reference_model() -> None:
    plan = _plan("--generation-provider", "anthropic", "--embedding-provider", "openai")

    assert plan.kinds == {
        "generation": smoke.KindPlan("anthropic", "claude-haiku-5-5", None),
        "embedding": smoke.KindPlan("openai", "text-embedding-3-small", None),
    }
    assert [smoke.provider_key_variable(item.provider) for item in plan.kinds.values()] == [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]


@pytest.mark.parametrize(
    ("options", "environ", "message"),
    [
        (["--generation-provider", "anthropic"], {}, "no embedding provider"),
        (
            ["--generation-provider", "openai_compatible"],
            {"OPENROUTER_API_KEY": "sk-or"},
            "--generation-model is required",
        ),
        (SPLIT_OPTIONS, {}, "TEAM_LLM_KEY, named by --generation-key-env, is not set"),
        (
            [*SPLIT_OPTIONS[:6], "--generation-key-env", "sk-or-v1-0123"],
            {},
            "--generation-key-env takes the name of an environment variable, not a key",
        ),
    ],
)
def test_an_incomplete_plan_is_refused(options: list[str], environ: dict[str, str], message: str) -> None:
    with pytest.raises(smoke.PlanError, match=re.escape(message)) as refused:
        _plan(*options, **environ)

    assert "sk-or" not in str(refused.value)


def test_a_refused_plan_exits_2_and_prints_its_reason_to_stderr(capsys) -> None:
    assert smoke.main(["model-plan", "--generation-provider", "openai_compatible"]) == 2

    printed = capsys.readouterr()
    assert printed.out == ""
    assert "--generation-model is required" in printed.err


def test_the_plan_names_each_ollama_server_once() -> None:
    plan = _plan("--generation-provider", "anthropic", "--embedding-provider", "ollama")
    assert smoke.shell_plan(plan).splitlines()[-1] == f"ollama_endpoints={smoke.DEFAULT_OLLAMA_URL}"

    both = _plan("--ollama", "--embedding-endpoint", "http://gpu-box:11434")
    assert smoke.shell_plan(both).splitlines()[-1] == (
        f"ollama_endpoints='http://gpu-box:11434 {smoke.DEFAULT_OLLAMA_URL}'"
    )


def test_the_shell_reads_the_plan_without_any_key(tmp_path: Path) -> None:
    environ = {"TEAM_LLM_KEY": "sk-team-secret", "OPENROUTER_API_KEY": "sk-or-secret", "PATH": "/usr/bin:/bin"}
    printed = subprocess.run(
        [sys.executable, str(SCRIPT), "model-plan", *SPLIT_OPTIONS, "--generation-model", "it's-a-model"],
        env=environ, capture_output=True, text=True, check=True,
    ).stdout
    shell = subprocess.run(
        ["bash", "-c", 'eval "$1"; printf "%s|%s" "$models_label" "$ollama_endpoints"']
        + ["_", printed],
        capture_output=True, text=True, check=True,
    ).stdout

    assert "secret" not in printed and "llm.example.com" not in printed
    assert shell == (
        "generation openai_compatible: it's-a-model; embedding openrouter: openai/text-embedding-3-small"
        "|"
    )


def test_the_shell_script_passes_every_model_option_the_helper_takes() -> None:
    script = (ROOT / "scripts" / "compose_smoke.sh").read_text()
    options = script.split("case $1 in", 1)[1].split("esac", 1)[0]
    passed = set(re.findall(r"(--(?:generation|embedding|judge)-[a-z-]+|--ollama(?:-url)?)[ )]", options))
    helper_help = subprocess.run(
        [sys.executable, str(SCRIPT), "model-plan", "--help"], capture_output=True, text=True, check=True
    ).stdout

    assert passed == set(re.findall(r"--[a-z-]+", helper_help)) - {"--help"}


def _provider_descriptors() -> list[dict]:
    """The descriptors ``GET /api/v1/model-settings/providers`` lists."""
    from memforge.llm.providers import PROVIDER_DESCRIPTORS
    from memforge.server.model_settings_api import _descriptor_response

    return [_descriptor_response(descriptor).model_dump(mode="json") for descriptor in PROVIDER_DESCRIPTORS]


class FakeSettingsApi:
    """The Settings API calls ``configure_models`` makes, answering as a validated model does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.candidates: dict[str, dict] = {}

    def __call__(self, api: str, method: str, path: str, payload=None, *, expect=(200,)):
        self.calls.append((method, path, payload))
        kind = path.split("/")[4] if path.count("/") > 4 else None
        if path == "/api/v1/model-settings/providers":
            return {"providers": _provider_descriptors()}
        if path.endswith("/discover-models"):
            models = [{"id": "gpt-6-luna"}, {"id": "openai/text-embedding-3-small"}]
            return {"ok": True, "models": models}
        if method == "PUT":
            self.candidates[kind] = payload
            return {}
        if path.endswith("/candidate/validate"):
            return {"passed": True, "promoted": True, "validation": {"facts": {"latency_ms": 42}}}
        if path == "/api/v1/model-settings":
            return {
                kind: {"active": {"provider": body["provider"], "model_id": body["model_id"], "state": "ready"}}
                for kind, body in self.candidates.items()
            }
        raise AssertionError(f"unexpected {method} {path}")


def test_each_kind_is_saved_through_its_own_provider(tmp_path: Path, monkeypatch) -> None:
    environ = {"TEAM_LLM_KEY": "sk-team-secret", "OPENROUTER_API_KEY": "sk-or-secret"}
    fake = FakeSettingsApi()
    monkeypatch.setattr(smoke, "api_json", fake)
    state = smoke.State(tmp_path / "state.json")

    detail = smoke.configure_models("http://api", _plan(*SPLIT_OPTIONS, **environ), environ, state, timeout=5)

    assert fake.candidates["generation"] == {
        "provider": "openai_compatible",
        "model_id": "gpt-6-luna",
        "endpoint": "https://llm.example.com/v1",
        "auth_strategy": "api_key",
        "api_key": "sk-team-secret",
        "token_limits": {
            "context_window_tokens": smoke.DEFAULT_CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": smoke.DEFAULT_MAX_OUTPUT_TOKENS,
        },
    }
    assert fake.candidates["embedding"] == {
        "provider": "openrouter",
        "model_id": "openai/text-embedding-3-small",
        "endpoint": None,
        "auth_strategy": "api_key",
        "api_key": "sk-or-secret",
    }
    discovered = [payload for method, path, payload in fake.calls if path.endswith("/discover-models")]
    assert [body["api_key"] for body in discovered] == ["sk-team-secret", "sk-or-secret"]
    assert detail == (
        "generation openai_compatible: gpt-6-luna validated in 42 ms; "
        "embedding openrouter: openai/text-embedding-3-small validated in 42 ms"
    )
    assert "secret" not in detail and "secret" not in state.path.read_text()
    assert state.get("models") == {
        "generation": {"provider": "openai_compatible", "model": "gpt-6-luna"},
        "embedding": {"provider": "openrouter", "model": "openai/text-embedding-3-small"},
    }


@pytest.mark.parametrize(
    ("kind", "plan", "message"),
    [
        ("generation", smoke.KindPlan("openai_compatible", "m", None), "pass --generation-endpoint"),
        ("embedding", smoke.KindPlan("anthropic", "m", None), "Anthropic serves no embedding models"),
        ("generation", smoke.KindPlan("openrouter", "m", None), "set OPENROUTER_API_KEY"),
        ("generation", smoke.KindPlan("bedrock", "m", None), "bedrock is not a provider Settings offers"),
    ],
)
def test_a_model_settings_would_refuse_is_refused_before_it_is_saved(kind: str, plan, message: str) -> None:
    descriptors = {descriptor["id"]: descriptor for descriptor in _provider_descriptors()}

    with pytest.raises(smoke.CheckFailed, match=re.escape(message)):
        smoke._binding(kind, plan, descriptors, {})


def test_a_local_ollama_is_sent_a_key_only_when_one_is_named() -> None:
    descriptors = {descriptor["id"]: descriptor for descriptor in _provider_descriptors()}
    environ = {"OLLAMA_API_KEY": "ollama-cloud-secret", "GPU_BOX_KEY": "gpu-secret"}
    plan = _plan("--ollama", **environ).kinds["generation"]

    binding = smoke._binding("generation", plan, descriptors, environ)

    assert (binding.auth_strategy, binding.api_key) == ("none", None)
    assert "api_key" not in smoke._candidate(binding, {})
    named = smoke._binding("generation", replace(plan, key_variable="GPU_BOX_KEY"), descriptors, environ)
    assert (named.auth_strategy, named.api_key) == ("api_key", "gpu-secret")


def test_a_keyless_openai_compatible_server_and_a_litellm_route_with_a_named_key() -> None:
    descriptors = {descriptor["id"]: descriptor for descriptor in _provider_descriptors()}
    local = smoke.KindPlan("openai_compatible", "local-model", "http://host.docker.internal:8000/v1")
    route = smoke.KindPlan("litellm", "voyage/voyage-3", None, "VOYAGE_API_KEY")

    assert smoke._binding("generation", local, descriptors, {}).auth_strategy == "none"
    assert smoke._binding("embedding", route, descriptors, {"VOYAGE_API_KEY": "pa-1"}).api_key == "pa-1"


def test_a_model_the_endpoint_does_not_list_is_entered_by_hand(monkeypatch) -> None:
    plan = smoke.KindPlan("openai_compatible", "local-model", "http://host.docker.internal:8000/v1")
    binding = smoke.Binding("generation", plan, "none")
    monkeypatch.setattr(
        smoke, "api_json", lambda *args, **kwargs: {"ok": True, "models_supported": False, "models": []}
    )

    assert smoke._listed_model("http://api", binding) == {}
    assert smoke._candidate(binding, {})["token_limits"] == {
        "context_window_tokens": smoke.DEFAULT_CONTEXT_WINDOW_TOKENS,
        "max_output_tokens": smoke.DEFAULT_MAX_OUTPUT_TOKENS,
    }


def test_the_restart_check_compares_each_kind_with_its_own_provider(tmp_path: Path, monkeypatch) -> None:
    active = {
        "generation": {"provider": "openai_compatible", "model_id": "gpt-6-luna", "state": "ready"},
        "embedding": {"provider": "openrouter", "model_id": "openai/text-embedding-3-small", "state": "ready"},
    }
    answers = {
        "/api/v1/health": {"status": "ok", "database": {"status": "ok"}, "vector_store": {"status": "ok"}},
        "/api/v1/model-settings": {kind: {"active": binding} for kind, binding in active.items()},
    }
    monkeypatch.setattr(smoke, "api_json", lambda api, method, path: answers[path])
    state = smoke.State(tmp_path / "state.json")
    state.set(models={
        "generation": {"provider": "openai_compatible", "model": "gpt-6-luna"},
        "embedding": {"provider": "openrouter", "model": "openai/text-embedding-3-small"},
    })

    assert smoke.check_persisted("http://api", state) == (
        "models kept (openai_compatible: gpt-6-luna, openrouter: openai/text-embedding-3-small); vector store ok"
    )
    active["embedding"]["provider"] = "openai"
    with pytest.raises(smoke.CheckFailed, match="embedding model in use changed to openai"):
        smoke.check_persisted("http://api", state)


def _judge(*options: str, **environ: str):
    return smoke.plan_judge(smoke.parse_args(["judge-env", *options]), environ)


def test_the_judge_defaults_to_the_reference_model_on_openrouter_or_none() -> None:
    assert _judge() is None
    assert _judge(OPENROUTER_API_KEY="key") == smoke.KindPlan("openrouter", "anthropic/claude-haiku-5.5", None)
    assert _judge("--judge-provider", "anthropic", OPENROUTER_API_KEY="key").model == "claude-haiku-5-5"
    with pytest.raises(smoke.PlanError, match="--judge-model is required"):
        _judge("--judge-provider", "openai_compatible")


def test_the_judge_reaches_the_api_only_through_its_env_file(capsys, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "judge-secret$1")

    assert smoke.main(["model-plan"]) == 0
    assert "judge-secret" not in capsys.readouterr().out
    assert smoke.main(["judge-env"]) == 0
    printed = capsys.readouterr().out.splitlines()

    assert printed == [
        "MEMFORGE_AGENT_EVALUATION_JUDGE_PROVIDER='openrouter'",
        "MEMFORGE_AGENT_EVALUATION_JUDGE_MODEL='anthropic/claude-haiku-5.5'",
        "MEMFORGE_AGENT_EVALUATION_JUDGE_ENDPOINT=''",
        "MEMFORGE_AGENT_EVALUATION_JUDGE_API_KEY='judge-secret$1'",
    ]
    monkeypatch.setenv("OPENROUTER_API_KEY", "it's")
    assert smoke.main(["judge-env"]) == 2
    assert "it's" not in capsys.readouterr().err


def test_the_expectations_not_met_are_listed_with_their_sentences_and_memories(monkeypatch) -> None:
    case = {
        "case_kind": "source_unit_support_v1",
        "manifest": {
            "projection": {"unit_title": {
                "kind": "GitHub file", "fields": [["Repository", "acme/kestrel"], ["Path", "docs/queue-limits.md"]],
            }},
            "work_items": [
                {"memory": {"id": "mem-a", "content": "A payload may be at most 256 KiB."}},
                {"memory": {"id": "mem-b", "content": "A tenant may have 500 jobs in flight."}},
            ],
        },
        "rubric": {"expectations": ["If a memory states the 256 KiB limit, it is judged unsupported.", "Other."]},
        "result": {"output": {"supports": [
            {"memory_id": "mem-a", "result": "supported"}, {"memory_id": "mem-b", "result": "supported"},
        ]}},
        "checks": [
            {"criterion": "expectation", "status": "completed", "label": "fail", "reason_code": "E1:fail/mem-a"},
            {"criterion": "expectation", "status": "completed", "label": "pass", "reason_code": "E2:pass"},
            {"criterion": "support_resolution", "status": "completed", "label": "pass", "reason_code": "x"},
        ],
    }
    monkeypatch.setattr(smoke, "api_json", lambda api, method, path: {"cases": [case]})

    listed = smoke.quality_expectations("http://api", "aer-1").splitlines()

    assert listed[2:] == [
        "| acme/kestrel / docs/queue-limits.md | E1: If a memory states the 256 KiB limit, it is judged unsupported. | not met "
        "| mem-a: A payload may be at most 256 KiB. (supported) |",
    ]
