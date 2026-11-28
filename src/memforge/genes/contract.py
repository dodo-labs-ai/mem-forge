"""Startup validation of every registered source type against its family's contract.

A connector that leaves out part of its family's contract fails startup with
a message naming the source type and the missing piece, instead of running
and quietly never detecting a deletion. The checks run once every type has
registered, because loading a Gene imports modules that build tables from the
registry.

What each family requires:

- **Every type** has a Gene of exactly one family, and the Gene implements
  every abstract method.
- **A document Gene** declares ``listing_kinds`` and answers ``lists_scope``
  as a class method, because a scope change asks it before any Gene exists.
  Python lets an abstract class method be called on its class, so the
  abstract-method check above is what refuses a Gene that leaves it out. A
  Gene that can produce a query listing implements ``confirm_absent``; one
  that answers ``rediscovers_documents`` implements ``rediscover``. A
  document type has no ``local_agent`` collection: it runs on a device
  through its executor.
- **Every type's form description** (``config_schema()``) can be read by
  the shared form check and the admin UI: each condition names an earlier
  field in group-then-field order (a panel's condition may also name a field
  of its own group) and one of that field's choices; no field repeats a
  choice value; every widget, panel and group purpose is one the admin UI
  knows. These rules read declarations only and never run a Gene's hooks.
- **A conversation type** names its provider in its descriptor
  (``conversation_provider``), and the provider class defines every member of
  the conversation provider port; a provider whose polls can omit replies
  also defines the thread members.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from memforge.conversations.port import REQUIRED_PROVIDER_MEMBERS, THREAD_PROVIDER_MEMBERS
from memforge.genes.conversation_gene import ConversationGene
from memforge.genes.document_gene import DocumentGene
from memforge.models import (
    CONFIG_GROUP_PURPOSES,
    KNOWN_PANELS,
    KNOWN_WIDGETS,
    ConfigCondition,
    ConfigField,
    GeneConfigSchema,
    ScopeListingKind,
    comparison_string,
)
from memforge.source_types import SOURCE_EXECUTOR_DEVICE, SOURCE_EXECUTORS, SourceTypeDescriptor, registered_source_types

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
    _validate_config_schema(name, gene_cls.config_schema())
    families = [base for base in (DocumentGene, ConversationGene) if issubclass(gene_cls, base)]
    if len(families) > 1:
        raise SourceTypeContractError(f"{name}: {gene_cls.__name__} subclasses both DocumentGene and ConversationGene")
    if not families:
        raise SourceTypeContractError(f"{name}: {gene_cls.__name__} must subclass DocumentGene or ConversationGene")
    if families[0] is DocumentGene:
        _validate_document_gene(descriptor, gene_cls)
    else:
        _validate_conversation_type(descriptor)


def _validate_config_schema(name: str, schema: GeneConfigSchema) -> None:
    group_order = {group.key: (group.order, index) for index, group in enumerate(schema.groups)}
    for group in schema.groups:
        if group.purpose not in CONFIG_GROUP_PURPOSES:
            raise SourceTypeContractError(f"{name}: group {group.key!r} has an unknown purpose {group.purpose!r}")
    ordered = sorted(
        enumerate(schema.fields),
        key=lambda item: (group_order.get(item[1].group, (len(schema.groups), 0)), item[1].order, item[0]),
    )
    position = {field.key: index for index, (_, field) in enumerate(ordered)}
    fields = {field.key: field for field in schema.fields}

    def check(conditions: tuple[ConfigCondition, ...], owner: str, earlier: Callable[[ConfigField], bool]) -> None:
        for condition in conditions:
            named = fields.get(condition.field)
            if named is None or not earlier(named):
                raise SourceTypeContractError(
                    f"{name}: {owner} has a condition on {condition.field!r}, which is not an earlier field"
                )
            declared = {comparison_string(option.value) for option in named.choices}
            unknown = [value for value in condition.values if comparison_string(value) not in declared]
            if unknown:
                raise SourceTypeContractError(
                    f"{name}: {owner} has a condition on {condition.field!r} naming {unknown[0]!r}, "
                    "which is not one of its choices"
                )

    for field in schema.fields:

        def before_field(named: ConfigField, field: ConfigField = field) -> bool:
            return position[named.key] < position[field.key]

        check(field.visible_when, f"field {field.key!r}", before_field)
        values = [comparison_string(option.value) for option in field.choices]
        if len(values) != len(set(values)):
            raise SourceTypeContractError(f"{name}: field {field.key!r} repeats a choice value")
        for option in field.choices:
            check(option.visible_when, f"choice {option.value!r} of field {field.key!r}", before_field)
        if field.widget is not None and field.widget not in KNOWN_WIDGETS:
            raise SourceTypeContractError(f"{name}: field {field.key!r} names an unknown widget {field.widget!r}")
    for group in schema.groups:
        for panel in group.panels:
            if panel.name not in KNOWN_PANELS:
                raise SourceTypeContractError(f"{name}: group {group.key!r} names an unknown panel {panel.name!r}")

            def in_or_before_group(named: ConfigField, group_key: str = group.key) -> bool:
                return group_order.get(named.group, (len(schema.groups), 0)) <= group_order[group_key]

            check(panel.visible_when, f"panel {panel.name!r}", in_or_before_group)


def _validate_document_gene(descriptor: SourceTypeDescriptor, gene_cls: type[DocumentGene]) -> None:
    name = descriptor.source_type
    if descriptor.local_agent is not None:
        raise SourceTypeContractError(f"{name}: a document type runs on a device through its executor, not local_agent")
    contract = descriptor.acquisition_contract
    if isinstance(contract, bool) or not isinstance(contract, int) or contract < 1:
        raise SourceTypeContractError(f"{name}: a document type must declare a positive integer acquisition_contract")
    if not descriptor.executors or not descriptor.executors <= SOURCE_EXECUTORS:
        raise SourceTypeContractError(
            f"{name}: a document type's executors must be a non-empty subset of {sorted(SOURCE_EXECUTORS)}"
        )
    config_keys = {field.key for field in gene_cls.config_schema().fields}
    if unknown := [key for key in descriptor.device_location_fields if key not in config_keys]:
        raise SourceTypeContractError(f"{name}: device_location_fields names no configuration field: {unknown}")
    if descriptor.device_location_fields and SOURCE_EXECUTOR_DEVICE not in descriptor.executors:
        raise SourceTypeContractError(f"{name}: a type with device_location_fields must run on the device")
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


def _overrides(gene_cls: type[DocumentGene], member: str) -> bool:
    """Whether ``gene_cls`` defines ``member`` itself or inherits it from a class other than DocumentGene."""

    owner = next(klass for klass in gene_cls.__mro__ if member in vars(klass))
    return owner is not DocumentGene


def _validate_conversation_type(descriptor: SourceTypeDescriptor) -> None:
    name = descriptor.source_type
    if descriptor.acquisition_contract is not None:
        raise SourceTypeContractError(f"{name}: a conversation type has no acquisition_contract")
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
