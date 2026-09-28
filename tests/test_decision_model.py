"""The decision model setting and the registry of Decision tasks that passed their evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace

import pytest

from memforge.agent_sessions import (
    AGENT_SESSION_AUTHORITY_CONTRACT,
    AGENT_SESSION_AUTHORITY_TASK,
    _classify_agent_session_authority,
    canonicalize_agent_session_events,
)
from memforge.config import AppConfig
from memforge.llm import decision_model
from memforge.llm.decision_model import EVALUATED_DECISION_TASKS, DecisionTask, decision_task_model
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.memory import engine
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CROSS_DOCUMENT_RELATION_TASK,
    StructuredCrossDocumentRelationClassifier,
)
from memforge.memory.entity_resolver import (
    ENTITY_ADJUDICATION_CONTRACT,
    ENTITY_ADJUDICATION_TASK,
    EntityResolutionContext,
    EntityResolver,
)
from memforge.memory.engine import _lifecycle_decision_task_models, _source_lifecycle_operation_input_hash
from memforge.models import Entity
from memforge.memory.relation_classifier import (
    MEMORY_PAIR_REVIEW_CONTRACT,
    PAIR_REVIEW_TASK,
    StructuredMemoryPairClassifier,
)
from memforge.pipeline.revision_work import CHANGE_IMPACT_CONTRACT, CHANGE_IMPACT_TASK, RevisionWorkExecutor
from memforge.runtime import DefaultRuntimeProvider, get_effective_llm_config
from tests.test_agent_session_api import _AuthorizesAllCandidateUserEvidence, _authorized_events
from tests.test_change_impact import ImpactClient, cohort, impact_works
from tests.test_cross_document_relation_classifier import _Client as RelationClient
from tests.test_cross_document_relation_classifier import _pairs
from tests.test_entity_resolver import _SCOPE, EntityClient, FakeEntityStore, _match_first_candidate
from tests.test_revision_work import Store

MAIN_MODEL = "gateway/main-model"
DECISION_MODEL = "gateway/decision-model"
TASK = DecisionTask("fixture_task", "fixture-task-v2")
OLD_CONTRACT_VERSION = "fixture-task-v1"


def register(monkeypatch, entries: dict[str, str]) -> None:
    monkeypatch.setattr(decision_model, "EVALUATED_DECISION_TASKS", MappingProxyType(entries))


def test_no_task_has_passed_its_decision_evaluation() -> None:
    assert dict(EVALUATED_DECISION_TASKS) == {}


def test_every_decision_task_carries_its_current_contract_version() -> None:
    tasks = (
        CHANGE_IMPACT_TASK,
        PAIR_REVIEW_TASK,
        CROSS_DOCUMENT_RELATION_TASK,
        ENTITY_ADJUDICATION_TASK,
        AGENT_SESSION_AUTHORITY_TASK,
    )
    assert [task.contract_version for task in tasks] == [
        CHANGE_IMPACT_CONTRACT,
        MEMORY_PAIR_REVIEW_CONTRACT,
        CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
        ENTITY_ADJUDICATION_CONTRACT,
        AGENT_SESSION_AUTHORITY_CONTRACT,
    ]
    assert len({task.name for task in tasks}) == len(tasks)


@pytest.mark.parametrize(
    ("configured", "registered", "expected"),
    [
        pytest.param(None, {}, MAIN_MODEL, id="unset"),
        pytest.param(None, {TASK.name: TASK.contract_version}, MAIN_MODEL, id="unset-registered"),
        pytest.param(DECISION_MODEL, {}, MAIN_MODEL, id="set-unregistered"),
        pytest.param(DECISION_MODEL, {TASK.name: TASK.contract_version}, DECISION_MODEL, id="set-registered"),
        pytest.param(DECISION_MODEL, {TASK.name: OLD_CONTRACT_VERSION}, MAIN_MODEL, id="set-old-contract"),
    ],
)
def test_a_task_uses_the_decision_model_only_when_set_and_registered_at_its_contract_version(
    monkeypatch, configured, registered, expected,
) -> None:
    register(monkeypatch, registered)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model=MAIN_MODEL, base_url=None, api_key=None, timeout_s=1.0, decision_model=configured,
    ))

    assert decision_task_model(client, TASK, MAIN_MODEL) == expected


def test_a_client_that_names_no_decision_model_keeps_the_callers_model(monkeypatch) -> None:
    register(monkeypatch, {TASK.name: TASK.contract_version})

    assert decision_task_model(object(), TASK, MAIN_MODEL) == MAIN_MODEL
    assert decision_task_model(SimpleNamespace(decision_model=""), TASK, None) is None


def test_decision_model_setting_is_empty_by_default_and_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("MEMFORGE_DECISION_MODEL", raising=False)
    assert AppConfig().llm.decision_model == ""

    monkeypatch.setenv("MEMFORGE_DECISION_MODEL", DECISION_MODEL)
    assert AppConfig().llm.decision_model == DECISION_MODEL


@pytest.mark.asyncio
async def test_effective_config_takes_the_decision_model_from_the_environment_only(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_DECISION_MODEL", DECISION_MODEL)

    class StoredConfig:
        async def get_llm_config(self):
            return {
                "enrichment_model": MAIN_MODEL,
                "enrichment_api_key": "fixture-key",
                "decision_model": "gateway/stored-model",
            }

    llm = await get_effective_llm_config(StoredConfig(), AppConfig())
    client = DefaultRuntimeProvider().build_structured_llm_client(llm, max_concurrent=1)

    assert llm.enrichment_model == MAIN_MODEL
    assert llm.decision_model == DECISION_MODEL
    assert client.config.model == MAIN_MODEL
    assert client.decision_model == DECISION_MODEL


@pytest.mark.asyncio
async def test_unset_decision_model_builds_a_client_that_runs_every_task_on_the_main_model(monkeypatch) -> None:
    monkeypatch.delenv("MEMFORGE_DECISION_MODEL", raising=False)
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_MODEL", MAIN_MODEL)
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_API_KEY", "fixture-key")

    class NoStoredConfig:
        async def get_llm_config(self):
            return None

    llm = await get_effective_llm_config(NoStoredConfig(), AppConfig())
    client = DefaultRuntimeProvider().build_structured_llm_client(llm, max_concurrent=1)

    assert client.decision_model is None


class DecisionImpactClient(ImpactClient):
    """Records the model of every Change Impact call and capacity lookup."""

    decision_model = DECISION_MODEL

    def __init__(self):
        super().__init__()
        self.budget_models = []
        self.impact_models = []

    def request_budget(self, model=None):
        self.budget_models.append(model)
        return super().request_budget(model)

    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        if "<change_impact>" in prompt:
            self.impact_models.append(kwargs["model"])
        return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)


def unaffected_cohort():
    old = "# Plan\n\nAlder ships in October.\n"
    new = old + "\n# Notes\n\nThe team reviewed dashboards.\n"
    return cohort(old, new, {"w0": ("Alder ships in October.", ["Alder ships in October."])})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "expected"),
    [
        pytest.param({}, MAIN_MODEL, id="unregistered"),
        pytest.param({CHANGE_IMPACT_TASK.name: CHANGE_IMPACT_CONTRACT}, DECISION_MODEL, id="registered"),
        pytest.param({CHANGE_IMPACT_TASK.name: "change-impact-v1"}, MAIN_MODEL, id="old-contract"),
    ],
)
async def test_change_impact_records_the_model_that_answered_it(monkeypatch, registered, expected) -> None:
    register(monkeypatch, registered)
    client, store = DecisionImpactClient(), Store()

    results = await RevisionWorkExecutor(
        client=client, model=MAIN_MODEL, store=store, derivation_id="root",
    ).assess_many(unaffected_cohort())

    assert client.impact_models == [expected]
    assert expected in client.budget_models
    [work] = impact_works(store)
    assert work.manifest["model"] == expected
    assert work.manifest["budget"] == client.input_policy_identity_for(expected)
    assert results["w0"].memory.support_validation["route"] == "change_impact"
    assert results["w0"].memory.support_validation["model"] == expected


@dataclass
class DecisionRelationClient(RelationClient):
    decision_model: str = DECISION_MODEL

    def __post_init__(self):
        self.models = []

    async def classify_cross_document_relations(self, prompt, *, max_tokens, model=None):
        self.models.append(model)
        return await super().classify_cross_document_relations(prompt, max_tokens=max_tokens, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "expected"),
    [
        pytest.param({}, MAIN_MODEL, id="unregistered"),
        pytest.param(
            {CROSS_DOCUMENT_RELATION_TASK.name: CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION},
            DECISION_MODEL,
            id="registered",
        ),
    ],
)
async def test_cross_document_relations_run_on_the_model_their_registration_selects(
    monkeypatch, registered, expected,
) -> None:
    register(monkeypatch, registered)
    client = DecisionRelationClient()

    await StructuredCrossDocumentRelationClassifier(client=client, model=MAIN_MODEL).classify(_pairs(2))

    assert client.models == [expected]


def test_pair_review_is_registered_by_its_own_task_name(monkeypatch) -> None:
    register(monkeypatch, {CROSS_DOCUMENT_RELATION_TASK.name: CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION})
    client = SimpleNamespace(decision_model=DECISION_MODEL)
    assert StructuredMemoryPairClassifier(client=client, model=MAIN_MODEL)._model == MAIN_MODEL

    register(monkeypatch, {PAIR_REVIEW_TASK.name: MEMORY_PAIR_REVIEW_CONTRACT})
    assert StructuredMemoryPairClassifier(client=client, model=MAIN_MODEL)._model == DECISION_MODEL


@dataclass
class DecisionEntityClient(EntityClient):
    decision_model: str = DECISION_MODEL
    models: list[str | None] = field(default_factory=list)

    async def validate_entity_batch(self, prompt, *, max_tokens, model):
        self.models.append(model)
        return await super().validate_entity_batch(prompt, max_tokens=max_tokens, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "expected"),
    [
        pytest.param({}, MAIN_MODEL, id="unregistered"),
        pytest.param({ENTITY_ADJUDICATION_TASK.name: ENTITY_ADJUDICATION_CONTRACT}, DECISION_MODEL, id="registered"),
    ],
)
async def test_entity_adjudication_runs_on_the_model_its_registration_selects(
    monkeypatch, registered, expected,
) -> None:
    register(monkeypatch, registered)
    candidate = Entity(id=1, canonical_name="payroll service", display_name="Payroll Service")
    store = FakeEntityStore(EntityResolutionContext(
        exact_matches={}, alias_matches={}, candidates={"pay service": (candidate,)},
    ))
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.embed_texts",
        lambda texts, *_args: [[1.0, 0.0] for _ in texts],
    )
    client = DecisionEntityClient(respond=_match_first_candidate)
    resolver = EntityResolver(
        store=store,  # type: ignore[arg-type]
        embed_cfg={"base_url": "http://embed", "api_key": "key", "model": "model"},
        structured_llm_client=client,
        llm_model=MAIN_MODEL,
    )

    result = await resolver.resolve_many({"pay service": ()}, scope=_SCOPE)

    assert client.models == [expected]
    assert result.entity_id("pay service") == candidate.id


class DecisionAuthorityClient(_AuthorizesAllCandidateUserEvidence):
    decision_model = DECISION_MODEL

    def __init__(self):
        self.models = []

    async def classify_agent_session_evidence_authority(self, prompt: str, **kwargs):
        self.models.append(kwargs["model"])
        return await super().classify_agent_session_evidence_authority(prompt, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "expected"),
    [
        pytest.param({}, None, id="unregistered"),
        pytest.param(
            {AGENT_SESSION_AUTHORITY_TASK.name: AGENT_SESSION_AUTHORITY_CONTRACT}, DECISION_MODEL, id="registered",
        ),
    ],
)
async def test_agent_session_authority_runs_on_the_model_its_registration_selects(
    monkeypatch, registered, expected,
) -> None:
    """Unregistered, the window client answers on its own configured model."""

    register(monkeypatch, registered)
    client = DecisionAuthorityClient()

    classified = await _classify_agent_session_authority(
        structured_llm_client=client,
        owner_user_id="owner",
        client="codex",
        session_id="session",
        trigger="Stop",
        workspace="/workspace/mem-forge",
        repo_identifier="github.com/shno-labs/mem-forge",
        branch="main",
        events=canonicalize_agent_session_events(_authorized_events()),
    )

    assert client.models == [expected]
    assert [event["evidence_role"] for event in classified] == ["primary"]


def test_source_lifecycle_operation_identity_names_the_model_of_each_decision_task(monkeypatch) -> None:
    client = SimpleNamespace(decision_model=DECISION_MODEL)

    register(monkeypatch, {})
    unregistered = _lifecycle_decision_task_models(client, MAIN_MODEL)
    register(monkeypatch, {CHANGE_IMPACT_TASK.name: CHANGE_IMPACT_CONTRACT})
    registered = _lifecycle_decision_task_models(client, MAIN_MODEL)

    assert unregistered == {CHANGE_IMPACT_TASK.name: MAIN_MODEL, PAIR_REVIEW_TASK.name: MAIN_MODEL}
    assert registered == {CHANGE_IMPACT_TASK.name: DECISION_MODEL, PAIR_REVIEW_TASK.name: MAIN_MODEL}
    inputs = dict(
        projection=SimpleNamespace(),
        candidates=[],
        incumbents=[],
        support_hashes={},
        gate_state="enabled",
        update_mode="incremental",
        changed_hunks=None,
        update_plan_stats=None,
        llm_model=MAIN_MODEL,
    )
    monkeypatch.setattr(engine, "source_derivation_projection_identity_hash", lambda _projection: "projection")
    assert _source_lifecycle_operation_input_hash(
        **inputs, decision_task_models=unregistered,
    ) != _source_lifecycle_operation_input_hash(**inputs, decision_task_models=registered)


def test_decision_model_capacity_resolves_for_its_own_route(monkeypatch) -> None:
    capacities = {
        MAIN_MODEL: {"max_input_tokens": 200_000, "max_output_tokens": 64_000},
        DECISION_MODEL: {"max_input_tokens": 100_000, "max_output_tokens": 8_000},
    }
    monkeypatch.setattr("memforge.llm.request_budget.litellm.get_model_info", lambda model: capacities[model])
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model=MAIN_MODEL, base_url=None, api_key=None, timeout_s=1.0, decision_model=DECISION_MODEL,
    ))

    assert client.request_budget(DECISION_MODEL).output_limit == capacities[DECISION_MODEL]["max_output_tokens"]
    assert client.request_budget().output_limit == capacities[MAIN_MODEL]["max_output_tokens"]
    assert client.input_policy_identity_for(DECISION_MODEL) != client.input_policy_identity


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


@pytest.mark.asyncio
async def test_decision_model_calls_use_their_own_schema_transport_and_report_their_model(monkeypatch) -> None:
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _Completion('{"decisions":[]}')

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: False)
    telemetry = []
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model=MAIN_MODEL, base_url=None, api_key=None, timeout_s=1.0,
            decision_model=DECISION_MODEL, decision_native_schema_transport="json_schema_response_format",
        ),
        telemetry_sink=telemetry.append,
    )

    await client.classify_cross_document_relations("classify", max_tokens=512, model=DECISION_MODEL)
    await client.classify_cross_document_relations("classify", max_tokens=512)

    assert [call["model"] for call in calls] == [DECISION_MODEL, MAIN_MODEL]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "response_format" not in calls[1]
    assert [item.model for item in telemetry] == [DECISION_MODEL, MAIN_MODEL]
