"""Gene lookup and construction by source type.

Every source type registers one descriptor in :mod:`memforge.source_types`,
and the descriptor names its Gene, which loads with this package. ``GENE_REGISTRY`` is a read-only view of
those descriptors keyed by source type; the sync orchestrator uses
``create_gene()`` to instantiate a Gene and ``list_available_genes()`` to
populate the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any, cast

from memforge.genes.base import Gene
from memforge.genes.contract import validate_registered_source_types
from memforge.genes.document_gene import DocumentGene
from memforge.genes.family import CollectionFamilyKind, gene_family
from memforge.models import GeneMetadata, SourceExecutionKind
from memforge.source_types import registered_source_types, source_type_descriptor

__all__ = [
    "GENE_REGISTRY",
    "DocumentGene",
    "create_gene",
    "gene_class",
    "gene_exports_pdf",
    "list_available_genes",
    "source_lists_scope",
    "source_rediscovers_documents",
    "source_type_execution_kinds",
    "source_type_supports_sync",
    "Gene",
]

logger = logging.getLogger(__name__)


class _GeneRegistry(Mapping[str, type[Gene]]):
    """The Gene class of every registered source type."""

    def __getitem__(self, source_type: str) -> type[Gene]:
        descriptor = source_type_descriptor(source_type)
        if descriptor is None:
            raise KeyError(source_type)
        return descriptor.gene_class()

    def __iter__(self) -> Iterator[str]:
        return iter(descriptor.source_type for descriptor in registered_source_types())

    def __len__(self) -> int:
        return len(registered_source_types())


GENE_REGISTRY: Mapping[str, type[Gene]] = _GeneRegistry()


def gene_class(name: str, config: Mapping[str, Any]) -> type[Gene]:
    """The Gene class a Source of a registered source type uses with this configuration.

    A Source that replays an evaluation snapshot set uses the snapshot Gene of
    a document type (:mod:`memforge.evals.source_snapshots`), so the class
    methods asked before any Gene exists answer for the set; a conversation
    type keeps its own Gene, since its device executor replays the set.
    Raises ``KeyError`` when no source type of that name is registered.
    """
    if name not in GENE_REGISTRY:
        available = ", ".join(sorted(GENE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown gene {name!r}.  Available genes: {available}")

    cls = GENE_REGISTRY[name]
    from memforge.evals.source_snapshots import SNAPSHOT_CONFIG_FIELD, snapshot_gene_class

    if SNAPSHOT_CONFIG_FIELD in (config or {}) and gene_family(cls) is CollectionFamilyKind.DOCUMENT:
        return snapshot_gene_class(cast(type[DocumentGene], cls))
    return cls


def create_gene(name: str, config: dict, source_id: str) -> Gene:
    """Instantiate the Gene a Source of a registered source type uses (:func:`gene_class`).

    A Source that replays an evaluation snapshot set is refused while the
    deployment does not replay snapshots. Raises ``KeyError`` when no source
    type of that name is registered.
    """
    cls = gene_class(name, config)
    from memforge.evals.source_snapshots import snapshot_reference, snapshot_set_dir

    reference = snapshot_reference(config)
    if reference is not None:
        snapshot_set_dir(reference)
    logger.debug("Creating gene %r (class=%s, source_id=%s)", name, cls.__qualname__, source_id)
    return cls(config=config, source_id=source_id)


def list_available_genes() -> list[GeneMetadata]:
    """Return metadata for every registered Gene, sorted by name."""
    return [cls.metadata() for _, cls in sorted(GENE_REGISTRY.items())]


def source_type_execution_kinds(source_type: str) -> tuple[SourceExecutionKind, ...]:
    """Return the declared sync execution kinds for a registered source type."""
    gene_cls = GENE_REGISTRY.get(str(source_type or "").strip())
    if gene_cls is None:
        return ()
    return gene_cls.metadata().execution_kinds


def source_type_supports_sync(source_type: str) -> bool:
    """Whether the source type participates in the ordinary sync lifecycle."""
    return bool(source_type_execution_kinds(source_type))


def gene_exports_pdf(gene: Gene) -> bool:
    """Whether the Gene's source type renders a PDF of each Unit."""
    descriptor = source_type_descriptor(gene.metadata().name)
    return descriptor is not None and descriptor.exports_pdf


def source_rediscovers_documents(source_type: str, config: Mapping[str, Any]) -> bool:
    """Whether a configured Source can ask its provider for one Document by id."""
    gene_cls = _configured_gene_class(source_type, config)
    return (
        gene_cls is not None
        and gene_family(gene_cls) is CollectionFamilyKind.DOCUMENT
        and gene_cls.rediscovers_documents(config)
    )


def source_lists_scope(source_type: str, config: Mapping[str, Any]) -> bool:
    """Whether a configured document Source can list its whole scope (``DocumentGene.lists_scope``)."""
    gene_cls = _configured_gene_class(source_type, config)
    return (
        gene_cls is not None
        and gene_family(gene_cls) is CollectionFamilyKind.DOCUMENT
        and gene_cls.lists_scope(config)
    )


def _configured_gene_class(source_type: str, config: Mapping[str, Any]) -> type[Gene] | None:
    try:
        return gene_class(str(source_type or "").strip(), config)
    except KeyError:
        return None


# Every registered Gene loads with this package, and every source type is
# checked against its family's contract, so a connector that cannot load or
# leaves out part of its contract fails startup instead of going missing.
for _descriptor in registered_source_types():
    _descriptor.gene_class()
validate_registered_source_types()
