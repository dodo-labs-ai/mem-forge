"""Agent sessions: knowledge a coding agent captures through the managed entrypoint."""

from __future__ import annotations

from memforge.source_adapters import agent_session as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.descriptor import SourceTypeDescriptor

DESCRIPTOR = SourceTypeDescriptor(
    source_type="agent_session",
    gene="memforge.genes.agent_session_gene:AgentSessionGene",
    namespace_fields=("client",),
    project_native=adapter.project_native,
    evidence_profiles={
        "session_summary": MARKDOWN_STRUCTURAL_PROFILE,
        "agent_concept": MARKDOWN_STRUCTURAL_PROFILE,
    },
)
