"""Register a test-only source type for the duration of one test."""

from __future__ import annotations

from typing import Any

import pytest

from memforge import source_types
from memforge.genes.base import Gene
from memforge.genes.contract import validate_source_type
from memforge.source_types import SourceTypeDescriptor, register_source_type


def register_test_source_type(
    monkeypatch: pytest.MonkeyPatch, source_type: str, gene: type[Gene], **descriptor_fields: Any
) -> None:
    """Register ``gene`` as ``source_type``; it must meet its family's contract, as at startup."""

    # The test registers into a copy of the registry, which monkeypatch restores afterwards.
    monkeypatch.setattr(source_types, "_DESCRIPTORS", dict(source_types._DESCRIPTORS))
    descriptor = SourceTypeDescriptor(source_type=source_type, gene=gene, **descriptor_fields)
    validate_source_type(descriptor)
    register_source_type(descriptor)
