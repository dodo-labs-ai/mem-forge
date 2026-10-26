"""Which collection family a Gene belongs to.

A source type's family follows from the base class of its Gene and from
nothing else: a :class:`~memforge.genes.document_gene.DocumentGene` collects
documents, and a :class:`~memforge.genes.conversation_gene.ConversationGene`
collects conversations through the conversation layer. Every registered
source type has a Gene of one of the two; the source-type contract
(:mod:`memforge.genes.contract`) refuses any other.
"""

from __future__ import annotations

from enum import Enum

from memforge.genes.base import Gene
from memforge.genes.conversation_gene import ConversationGene
from memforge.genes.document_gene import DocumentGene

__all__ = ["CollectionFamilyKind", "gene_family"]


class CollectionFamilyKind(str, Enum):
    """The collection family of a source type."""

    DOCUMENT = "document"
    CONVERSATION = "conversation"


def gene_family(gene_cls: type[Gene]) -> CollectionFamilyKind | None:
    """The family of ``gene_cls``, or ``None`` for a class of neither family, which the source-type contract refuses."""

    if issubclass(gene_cls, DocumentGene):
        return CollectionFamilyKind.DOCUMENT
    if issubclass(gene_cls, ConversationGene):
        return CollectionFamilyKind.CONVERSATION
    return None
