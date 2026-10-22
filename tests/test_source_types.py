from __future__ import annotations

import pytest

from memforge.genes import GENE_REGISTRY
from memforge.genes.conversation_gene import ConversationGene
from memforge.source_types import (
    SourceTypeDescriptor,
    load_reference,
    register_source_type,
    registered_source_types,
    source_type_descriptor,
)

BUILTIN_SOURCE_TYPES = {
    "agent_session",
    "confluence",
    "github_pages",
    "github_repo",
    "jira",
    "local_markdown",
    "teams",
}


def test_every_builtin_source_type_has_one_complete_descriptor() -> None:
    descriptors = {descriptor.source_type: descriptor for descriptor in registered_source_types()}

    assert set(descriptors) == BUILTIN_SOURCE_TYPES
    for source_type, descriptor in descriptors.items():
        gene_cls = descriptor.gene_class()
        assert gene_cls.metadata().name == source_type
        assert GENE_REGISTRY[source_type] is gene_cls
        assert descriptor.evidence_profiles, source_type
        if issubclass(gene_cls, ConversationGene):
            # Conversation Units are cut by the conversation layer, not by the adapter.
            assert descriptor.project_native is None
        else:
            assert descriptor.project_native is not None, source_type
        if descriptor.local_agent is not None and descriptor.local_agent.replay_adapter is not None:
            assert descriptor.local_agent.replay_adapter_instance().source_type == source_type


def test_a_type_exports_pdfs_exactly_when_its_gene_renders_them() -> None:
    from memforge.genes.base import Gene

    for descriptor in registered_source_types():
        renders = descriptor.gene_class().fetch_pdf is not Gene.fetch_pdf
        assert descriptor.exports_pdf is renders, descriptor.source_type
    assert {d.source_type for d in registered_source_types() if d.exports_pdf} == {"confluence"}


def test_an_unknown_source_type_has_no_descriptor() -> None:
    assert source_type_descriptor("outlook") is None
    assert source_type_descriptor(None) is None
    assert "outlook" not in GENE_REGISTRY


def test_a_second_descriptor_for_a_registered_type_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_source_type(SourceTypeDescriptor(source_type="jira", gene="memforge.genes.jira_gene:JiraGene"))


def test_a_source_type_name_must_be_lowercase() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        register_source_type(SourceTypeDescriptor(source_type="Notion", gene="x:y"))


def test_a_reference_that_cannot_load_fails_loudly() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_reference("memforge.genes.no_such_gene:NoSuchGene")
    with pytest.raises(ValueError, match="module:name"):
        load_reference("memforge.genes.jira_gene")
