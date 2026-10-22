"""Register a test-only source type for the duration of one test."""

from __future__ import annotations

import pytest

from memforge import source_types
from memforge.genes.base import Gene
from memforge.source_types import SourceTypeDescriptor


def register_test_source_type(monkeypatch: pytest.MonkeyPatch, source_type: str, gene: type[Gene]) -> None:
    monkeypatch.setitem(
        source_types._DESCRIPTORS,
        source_type,
        SourceTypeDescriptor(source_type=source_type, gene=gene),
    )
