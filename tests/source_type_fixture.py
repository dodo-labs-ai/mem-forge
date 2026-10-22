"""Register a test-only source type for the duration of one test."""

from __future__ import annotations

import pytest

from memforge import source_types
from memforge.genes.base import Gene
from memforge.source_types import SourceTypeDescriptor, register_source_type


def register_test_source_type(monkeypatch: pytest.MonkeyPatch, source_type: str, gene: type[Gene]) -> None:
    # The test registers into a copy of the registry, which monkeypatch restores afterwards.
    monkeypatch.setattr(source_types, "_DESCRIPTORS", dict(source_types._DESCRIPTORS))
    register_source_type(SourceTypeDescriptor(source_type=source_type, gene=gene))
