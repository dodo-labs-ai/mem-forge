from __future__ import annotations

import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import memforge
import memforge.genes.teams_provider as teams_provider
import memforge.main as main
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.genes.contract import (
    SourceTypeContractError,
    validate_registered_source_types,
    validate_source_type,
)
from memforge.genes.conversation_gene import ConversationGene
from memforge.genes.document_gene import DocumentGene
from memforge.genes.teams_gene import TeamsGene
from memforge.genes.teams_provider import TeamsConversationProvider
from memforge.local_agent.source_contract import LOCAL_AGENT_SYNC_OPERATIONS
from memforge.models import GeneConfigSchema, GeneMetadata, ScopeListingKind
from memforge.source_types import (
    ConversationProviderReference,
    LocalAgentCollection,
    SourceTypeDescriptor,
    registered_source_types,
    source_type_descriptor,
)
from tests.gene_fixture import FakeGene
from tests.source_type_fixture import register_test_source_type

DEVICE = LocalAgentCollection(operation="test_sync", job_config_fields=frozenset())


def _descriptor(gene, **fields) -> SourceTypeDescriptor:
    return SourceTypeDescriptor(source_type="test_type", gene=gene, **fields)


def _refused(gene, match: str, **fields) -> None:
    with pytest.raises(SourceTypeContractError, match=match):
        validate_source_type(_descriptor(gene, **fields))


class QueryGene(FakeGene):
    listing_kinds = frozenset({ScopeListingKind.QUERY})

    async def confirm_absent(self, items):
        return {}


class CapturedGene(AgentSessionGene):
    """A Gene with no collection, as managed capture has."""


def test_every_built_in_source_type_meets_its_contract() -> None:
    validate_registered_source_types()


def test_a_document_gene_with_its_contract_is_accepted() -> None:
    validate_source_type(_descriptor(FakeGene))
    validate_source_type(_descriptor(QueryGene))


def test_a_gene_without_a_collection_needs_no_family() -> None:
    validate_source_type(_descriptor(CapturedGene))


def test_a_collecting_gene_of_neither_family_is_refused() -> None:
    class Bare(AgentSessionGene):
        @classmethod
        def metadata(cls) -> GeneMetadata:
            return GeneMetadata(
                name="bare",
                display_name="Bare",
                description="collects without a family",
                default_sync_interval_minutes=60,
                auth_method="none",
                data_shape="document",
            )

    _refused(Bare, "must subclass DocumentGene or ConversationGene")


def test_a_gene_of_both_families_is_refused() -> None:
    class Both(FakeGene, ConversationGene):
        def conversation_selection(self):
            raise NotImplementedError

    _refused(Both, "both DocumentGene and ConversationGene")


def test_a_gene_with_unimplemented_steps_is_refused() -> None:
    class Partial(DocumentGene):
        listing_kinds = frozenset()

        @classmethod
        def metadata(cls) -> GeneMetadata:
            return FakeGene.metadata()

        @classmethod
        def config_schema(cls) -> GeneConfigSchema:
            return FakeGene.config_schema()

    _refused(Partial, "does not implement authenticate, discover, fetch, list_scope, lists_scope, normalize")


def test_a_gene_that_does_not_say_whether_it_lists_is_refused() -> None:
    # Python lets an abstract class method be called on its class, so only
    # the abstract-method check stops a Gene that leaves lists_scope out.
    steps = {name: value for name, value in vars(FakeGene).items() if name != "lists_scope"}
    Unanswered = type("Unanswered", (DocumentGene,), steps)

    assert Unanswered.lists_scope({}) is None
    _refused(Unanswered, "does not implement lists_scope")


def test_lists_scope_must_be_a_class_method() -> None:
    class InstanceAnswer(FakeGene):
        def lists_scope(self, config) -> bool:
            return False

    class StaticAnswer(FakeGene):
        @staticmethod
        def lists_scope(config) -> bool:
            return False

    _refused(InstanceAnswer, "lists_scope must be a class method")
    _refused(StaticAnswer, "lists_scope must be a class method")


def test_a_document_gene_must_declare_its_listing_kinds() -> None:
    # FakeGene's steps without its listing_kinds declaration.
    steps = {
        name: value
        for name, value in vars(FakeGene).items()
        if name == "__init__" or not (name.startswith("_") or name == "listing_kinds")
    }
    Unset = type("Unset", (DocumentGene,), steps)

    class Malformed(FakeGene):
        listing_kinds = ["query"]

    _refused(Unset, "listing_kinds must be a frozenset")
    _refused(Malformed, "listing_kinds must be a frozenset")


def test_a_query_listing_needs_confirmation_by_identifier() -> None:
    class Unconfirmed(FakeGene):
        listing_kinds = frozenset({ScopeListingKind.QUERY})

    _refused(Unconfirmed, "must implement confirm_absent")


def test_answering_rediscovers_documents_needs_rediscover() -> None:
    class Unanswered(FakeGene):
        @classmethod
        def rediscovers_documents(cls, config) -> bool:
            return True

    _refused(Unanswered, "must implement rediscover")


def test_an_override_counts_however_it_is_written() -> None:
    class PlainRediscovery(FakeGene):
        def rediscovers_documents(self, config) -> bool:
            return True

    class StaticRediscovery(FakeGene):
        @staticmethod
        def rediscovers_documents(config) -> bool:
            return True

    class ClassConfirmation(FakeGene):
        listing_kinds = frozenset({ScopeListingKind.QUERY})

        @classmethod
        async def confirm_absent(cls, items):
            return {}

    _refused(PlainRediscovery, "must implement rediscover")
    _refused(StaticRediscovery, "must implement rediscover")
    validate_source_type(_descriptor(ClassConfirmation))


def test_a_subclass_of_a_built_in_gene_inherits_its_contract() -> None:
    from memforge.genes.jira_gene import JiraGene

    class CustomJira(JiraGene):
        pass

    validate_source_type(_descriptor(CustomJira, local_agent=DEVICE))


def test_a_gene_missing_its_metadata_is_named_as_unimplemented() -> None:
    class Nameless(DocumentGene):
        listing_kinds = frozenset()

    _refused(
        Nameless,
        "does not implement authenticate, config_schema, discover, fetch, list_scope, lists_scope, metadata, normalize",
    )


class DeviceQueryGene(QueryGene):
    @classmethod
    def provider_key(cls, item):
        return item.item_id

    @classmethod
    def confirmation_item(cls, document_id, provider_key):
        raise NotImplementedError


def test_a_device_collected_gene_declares_one_listing_kind() -> None:
    class Both(DeviceQueryGene):
        listing_kinds = frozenset({ScopeListingKind.QUERY, ScopeListingKind.EXISTENCE})

    _refused(Both, "exactly one listing kind", local_agent=DEVICE)
    validate_source_type(_descriptor(DeviceQueryGene, local_agent=DEVICE))


def test_a_gene_confirmed_on_a_device_names_and_rebuilds_its_items() -> None:
    class KeyedOnly(QueryGene):
        @classmethod
        def provider_key(cls, item):
            return item.item_id

    _refused(QueryGene, "must implement provider_key, confirmation_item", local_agent=DEVICE)
    _refused(KeyedOnly, "must implement confirmation_item", local_agent=DEVICE)
    # Only a query listed by a device is confirmed there.
    validate_source_type(_descriptor(QueryGene))

    class Existence(FakeGene):
        listing_kinds = frozenset({ScopeListingKind.EXISTENCE})

    validate_source_type(_descriptor(Existence, local_agent=DEVICE))


def _provider(provider, factory=teams_provider.open_teams_provider) -> ConversationProviderReference:
    return ConversationProviderReference(provider=provider, factory=factory)


def test_a_conversation_type_must_name_its_provider() -> None:
    _refused(TeamsGene, "must name its conversation_provider", local_agent=DEVICE)


def test_a_conversation_provider_must_define_the_port() -> None:
    class Incomplete:
        record_schema = None

        async def list_conversations(self):
            raise NotImplementedError

    _refused(
        TeamsGene,
        "does not define get_conversation, poll, read_artifact",
        local_agent=DEVICE,
        conversation_provider=_provider(Incomplete),
    )


def test_a_provider_whose_polls_omit_replies_must_fetch_threads() -> None:
    class Threaded(TeamsConversationProvider):
        polls_omit_replies = True

    _refused(TeamsGene, "does not define fetch_thread", local_agent=DEVICE, conversation_provider=_provider(Threaded))

    class Fetching(Threaded):
        async def fetch_thread(self, conversation, thread_root_id):
            raise NotImplementedError

    validate_source_type(_descriptor(TeamsGene, local_agent=DEVICE, conversation_provider=_provider(Fetching)))


def test_a_conversation_provider_factory_must_be_callable() -> None:
    _refused(
        TeamsGene,
        "factory is not callable",
        local_agent=DEVICE,
        conversation_provider=ConversationProviderReference(provider=TeamsConversationProvider, factory=42),
    )


def test_a_refused_built_in_fails_the_import(tmp_path) -> None:
    # A Gene that collects content without a family, named by a built-in
    # descriptor before the genes package loads and validates every type.
    (tmp_path / "probe_gene.py").write_text(
        """
from memforge.genes.agent_session_gene import AgentSessionGene


class Broken(AgentSessionGene):
    @classmethod
    def metadata(cls):
        meta = AgentSessionGene.metadata()
        return type(meta)(**{**meta.__dict__, "execution_kinds": ("server",)})
"""
    )
    probe = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
import memforge.source_types as source_types
from memforge.source_types import SourceTypeDescriptor
source_types._DESCRIPTORS["agent_session"] = SourceTypeDescriptor(source_type="agent_session", gene="probe_gene:Broken")
import memforge.genes
"""
    package_root = str(Path(memforge.__file__).parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": package_root},
    )

    assert result.returncode != 0
    assert "SourceTypeContractError" in result.stderr
    assert "must subclass DocumentGene or ConversationGene" in result.stderr


def test_registry_tables_built_at_import_hold_every_type() -> None:
    assert LOCAL_AGENT_SYNC_OPERATIONS == {
        d.local_agent.operation for d in registered_source_types() if d.local_agent is not None
    }
    assert "teams_sync" in LOCAL_AGENT_SYNC_OPERATIONS


class _Client:
    def __init__(self, region: str, events: list[str], fail_validation: bool = False) -> None:
        self.region = region
        self.events = events
        self.fail_validation = fail_validation

    async def validate(self) -> None:
        self.events.append(f"validate:{self.region}")
        if self.fail_validation:
            raise RuntimeError("session expired")

    async def close(self) -> None:
        self.events.append("close")


async def test_the_teams_provider_validates_its_session_and_closes_its_client(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(teams_provider, "TeamsChatClient", lambda region: _Client(region, events))

    async with teams_provider.open_teams_provider({"region": "AMER"}) as provider:
        events.append("collect")
        assert isinstance(provider, TeamsConversationProvider)
    with pytest.raises(ValueError, match="collection failed"):
        async with teams_provider.open_teams_provider({"region": "mars"}):
            raise ValueError("collection failed")

    assert events == ["validate:amer", "collect", "close", "validate:emea", "close"]


async def test_the_teams_provider_closes_its_client_when_validation_fails(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        teams_provider, "TeamsChatClient", lambda region: _Client(region, events, fail_validation=True)
    )

    with pytest.raises(RuntimeError, match="session expired"):
        async with teams_provider.open_teams_provider({}):
            events.append("collect")

    assert events == ["validate:emea", "close"]


async def test_device_collection_opens_the_provider_its_descriptor_names(monkeypatch) -> None:
    opened: list[dict] = []
    collected: list[object] = []
    provider = object()

    @asynccontextmanager
    async def open_provider(job_config):
        opened.append(dict(job_config))
        yield provider

    teams = source_type_descriptor("teams")
    register_test_source_type(
        monkeypatch,
        "chat_test",
        TeamsGene,
        local_agent=teams.local_agent,
        conversation_provider=ConversationProviderReference(provider=TeamsConversationProvider, factory=open_provider),
    )

    async def fake_run_device_collection(provider_arg, client, **kwargs):
        collected.append(provider_arg)
        return "collected"

    import memforge.local_agent.conversation_executor as executor

    monkeypatch.setattr(executor, "run_device_collection", fake_run_device_collection)

    result = await main._collect_conversation_on_device(
        {"job_id": "laj-1", "attempt_count": 1, "source_type": "chat_test"},
        {"region": "emea"},
        "src-chat",
        object(),
        None,
    )

    assert result == "collected"
    assert opened == [{"region": "emea"}]
    assert collected == [provider]


async def test_device_collection_refuses_a_job_without_a_conversation_type() -> None:
    for job in ({"job_id": "laj-1", "attempt_count": 1}, {"job_id": "laj-1", "attempt_count": 1, "source_type": "jira"}):
        with pytest.raises(ValueError, match="has no conversation provider"):
            await main._collect_conversation_on_device(job, {}, "src-1", object(), None)
