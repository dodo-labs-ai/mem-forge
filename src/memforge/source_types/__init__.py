"""One descriptor per source type.

A source type's facts live beside it, in one descriptor: its Gene, how its
items project into Source Units, which configuration fields name its provider
and its scope, how the local agent collects it, and what its artifacts need.
Shared code reads the descriptor and never branches on a source type string.

Built-in types register when this package loads. A built-in that fails to load
fails the import, so a broken connector is never silently missing.
"""

from __future__ import annotations

import importlib

from memforge.source_types.config import SourceConfigPolicy
from memforge.source_types.descriptor import (
    ConversationProviderReference,
    LocalAgentCollection,
    SourceTypeDescriptor,
    load_reference,
)

__all__ = [
    "ConversationProviderReference",
    "LocalAgentCollection",
    "SourceConfigPolicy",
    "SourceTypeDescriptor",
    "load_reference",
    "register_source_type",
    "registered_source_types",
    "source_type_descriptor",
]

_BUILTIN_SOURCE_TYPES = (
    "agent_session",
    "confluence",
    "github_pages",
    "github_repo",
    "jira",
    "local_markdown",
    "notion",
    "teams",
)

_DESCRIPTORS: dict[str, SourceTypeDescriptor] = {}


def register_source_type(descriptor: SourceTypeDescriptor) -> SourceTypeDescriptor:
    """Register one source type; a second descriptor for the same type is refused.

    Registration belongs to import time: the evidence-representation and
    local-agent tables are built from the registry when their modules load.
    Built-in types register when this package loads.
    """

    key = descriptor.source_type
    if not key or key != key.strip().lower():
        raise ValueError(f"source type must be a lowercase name: {descriptor.source_type!r}")
    existing = _DESCRIPTORS.get(key)
    if existing is not None and existing is not descriptor:
        raise ValueError(f"source type {key!r} is already registered")
    _DESCRIPTORS[key] = descriptor
    return descriptor


def source_type_descriptor(source_type: object) -> SourceTypeDescriptor | None:
    """The descriptor of a registered source type, or ``None`` for an unknown one.

    The name must match exactly; a caller that accepts loose input normalizes it first.
    """

    return _DESCRIPTORS.get(source_type) if isinstance(source_type, str) else None


def registered_source_types() -> tuple[SourceTypeDescriptor, ...]:
    """Every registered descriptor, ordered by source type."""

    return tuple(_DESCRIPTORS[key] for key in sorted(_DESCRIPTORS))


for _name in _BUILTIN_SOURCE_TYPES:
    register_source_type(importlib.import_module(f"{__name__}.{_name}").DESCRIPTOR)
