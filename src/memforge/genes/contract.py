"""Startup validation of every registered source type against its family's contract.

A connector that leaves out part of its family's contract fails startup with
a message naming the source type and the missing piece, instead of running
and quietly never detecting a deletion. The checks run once every type has
registered, because loading a Gene imports modules that build tables from the
registry.

What each family requires:

- **Every type with a collection** (a sync run on the server or a local-agent
  collection) has a Gene of exactly one family, and the Gene implements every
  abstract method. Only a type without a collection, which is managed capture,
  has a Gene of neither family.
- **A document Gene** declares ``listing_kinds`` and answers ``lists_scope``
  as a class method, because a scope change asks it before any Gene exists.
  Python lets an abstract class method be called on its class, so the
  abstract-method check above is what refuses a Gene that leaves it out. A
  Gene that can produce a query listing implements ``confirm_absent``; one
  that answers ``rediscovers_documents`` implements ``rediscover``. A
  document type collected on a device produces exactly one listing kind,
  which its device snapshot takes.
- **A conversation type** names its provider in its descriptor
  (``conversation_provider``), and the provider class defines every member of
  the conversation provider port; a provider whose polls can omit replies
  also defines the thread members.
"""

from __future__ import annotations

import inspect

from memforge.conversations.port import REQUIRED_PROVIDER_MEMBERS, THREAD_PROVIDER_MEMBERS
from memforge.genes.conversation_gene import ConversationGene
from memforge.genes.document_gene import DocumentGene
from memforge.models import ScopeListingKind
from memforge.source_types import SourceTypeDescriptor, registered_source_types

__all__ = ["SourceTypeContractError", "validate_registered_source_types", "validate_source_type"]


class SourceTypeContractError(TypeError):
    """A registered source type does not meet its family's contract."""


def validate_registered_source_types() -> None:
    """Check every registered source type; the first violation raises."""

    for descriptor in registered_source_types():
        validate_source_type(descriptor)


def validate_source_type(descriptor: SourceTypeDescriptor) -> None:
    """Check one source type against its family's contract."""

    gene_cls = descriptor.gene_class()
    name = descriptor.source_type
    if gene_cls.__abstractmethods__:
        missing = ", ".join(sorted(gene_cls.__abstractmethods__))
        raise SourceTypeContractError(f"{name}: {gene_cls.__name__} does not implement {missing}")
    families = [base for base in (DocumentGene, ConversationGene) if issubclass(gene_cls, base)]
    collected = bool(gene_cls.metadata().execution_kinds) or descriptor.local_agent is not None
    if len(families) > 1:
        raise SourceTypeContractError(f"{name}: {gene_cls.__name__} subclasses both DocumentGene and ConversationGene")
    if not families:
        if collected:
            raise SourceTypeContractError(
                f"{name}: {gene_cls.__name__} collects content, so it must subclass DocumentGene or ConversationGene"
            )
        return
    if families[0] is DocumentGene:
        _validate_document_gene(descriptor, gene_cls)
    else:
        _validate_conversation_type(descriptor)


def _validate_document_gene(descriptor: SourceTypeDescriptor, gene_cls: type[DocumentGene]) -> None:
    name = descriptor.source_type
    kinds = getattr(gene_cls, "listing_kinds", None)
    if not isinstance(kinds, frozenset) or not all(isinstance(kind, ScopeListingKind) for kind in kinds):
        raise SourceTypeContractError(
            f"{name}: {gene_cls.__name__}.listing_kinds must be a frozenset of ScopeListingKind values"
        )
    if not isinstance(inspect.getattr_static(gene_cls, "lists_scope"), classmethod):
        raise SourceTypeContractError(f"{name}: {gene_cls.__name__}.lists_scope must be a class method")
    if ScopeListingKind.QUERY in kinds and not _overrides(gene_cls, "confirm_absent"):
        raise SourceTypeContractError(
            f"{name}: {gene_cls.__name__} can produce a query listing, so it must implement confirm_absent"
        )
    if _overrides(gene_cls, "rediscovers_documents") and not _overrides(gene_cls, "rediscover"):
        raise SourceTypeContractError(
            f"{name}: {gene_cls.__name__} answers rediscovers_documents, so it must implement rediscover"
        )
    if descriptor.local_agent is not None and len(kinds) != 1:
        raise SourceTypeContractError(
            f"{name}: {gene_cls.__name__} is collected on a device, so it must declare exactly one listing kind"
        )


def _overrides(gene_cls: type[DocumentGene], member: str) -> bool:
    """Whether ``gene_cls`` defines ``member`` itself or inherits it from a class other than DocumentGene."""

    owner = next(klass for klass in gene_cls.__mro__ if member in vars(klass))
    return owner is not DocumentGene


def _validate_conversation_type(descriptor: SourceTypeDescriptor) -> None:
    name = descriptor.source_type
    reference = descriptor.conversation_provider
    if reference is None:
        raise SourceTypeContractError(f"{name}: a conversation type must name its conversation_provider")
    provider_cls = reference.provider_class()
    required = list(REQUIRED_PROVIDER_MEMBERS)
    if getattr(provider_cls, "polls_omit_replies", False):
        required += THREAD_PROVIDER_MEMBERS
    missing = [member for member in required if not hasattr(provider_cls, member)]
    if missing:
        raise SourceTypeContractError(f"{name}: {provider_cls.__name__} does not define {', '.join(missing)}")
    if not callable(reference.factory_function()):
        raise SourceTypeContractError(f"{name}: conversation_provider.factory is not callable")
