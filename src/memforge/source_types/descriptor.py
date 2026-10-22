"""The facts one source type declares for shared code to read."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from memforge.source_types.config import SourceConfigPolicy

if TYPE_CHECKING:
    from memforge.genes.base import Gene
    from memforge.genes.family import CollectionFamilyKind
    from memforge.local_agent.replay_adapter import LocalSourceReplayAdapter
    from memforge.source_adapters.contracts import CanonicalRecordSchema
    from memforge.source_projection import EvidenceRepresentationProfile


def load_reference(reference: str) -> Any:
    """Import the attribute a ``"module:name"`` reference names."""

    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"source type reference must be 'module:name': {reference!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _frozen(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class LocalAgentCollection:
    """How the owner's local agent collects a Source of this type.

    ``collects`` says whether one configuration is collected by the agent; a
    type such as Jira is collected on the server unless configured otherwise.
    ``immutable_execution_fields`` choose between the two executors, so a
    stored Source never changes them. ``package_submitter`` names the
    function that accepts one package the agent uploads, and
    ``replay_adapter`` the class that reads it back during sync. A type with
    ``accepts_artifact_inputs`` takes binary inputs for its packages.
    """

    operation: str
    job_config_fields: frozenset[str]
    collects: Callable[[Mapping[str, Any]], bool] = lambda _config: True
    immutable_execution_fields: tuple[str, ...] = ()
    package_submitter: str | None = None
    replay_adapter: str | None = None
    accepts_artifact_inputs: bool = False
    job_config_defaults: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def replay_adapter_instance(self) -> LocalSourceReplayAdapter | None:
        if self.replay_adapter is None:
            return None
        return load_reference(self.replay_adapter)()


@dataclass(frozen=True)
class SourceTypeDescriptor:
    """Every per-type fact shared sync, projection, configuration and store code reads.

    ``gene`` is the Gene class, or a ``"module:Class"`` reference imported on
    first use. ``scope_fields`` are the configuration fields that change which
    Units the Source projects; ``canonical_scope`` drops those a configured
    mode leaves unused. ``project_native`` turns one fetched item into Units and
    Observations; a type without one is projected generically, with partial
    coverage. ``exports_pdf`` says the Gene renders a PDF of each Unit, which
    a Unit's stored input must then hold. ``config`` checks and prepares the
    Source's configuration.
    """

    source_type: str
    gene: str | type[Gene]
    unit_noun: str = "item"
    namespace_fields: tuple[str, ...] = ()
    scope_fields: tuple[str, ...] = ()
    canonical_scope: Callable[[dict[str, object]], dict[str, object]] | None = None
    project_native: Callable[..., Any] | None = None
    evidence_profiles: Mapping[str, EvidenceRepresentationProfile] = field(default_factory=dict)
    legacy_evidence_profiles: Mapping[str, EvidenceRepresentationProfile] = field(default_factory=dict)
    record_schemas: Mapping[tuple[str, int], CanonicalRecordSchema] = field(default_factory=dict)
    local_agent: LocalAgentCollection | None = None
    exports_pdf: bool = False
    config: SourceConfigPolicy = field(default_factory=SourceConfigPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_profiles", _frozen(self.evidence_profiles))
        object.__setattr__(self, "legacy_evidence_profiles", _frozen(self.legacy_evidence_profiles))
        object.__setattr__(self, "record_schemas", _frozen(self.record_schemas))

    def gene_class(self) -> type[Gene]:
        return self.gene if isinstance(self.gene, type) else load_reference(self.gene)

    @property
    def family(self) -> CollectionFamilyKind | None:
        """The collection family its Gene's base class names; ``None`` for managed capture."""

        from memforge.genes.family import gene_family

        return gene_family(self.gene_class())
