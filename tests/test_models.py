"""Tests for data models and utility functions."""

from __future__ import annotations

from memforge.models import (
    Memory,
    MemoryStatus,
    RawMemory,
    GeneMetadata,
    ContentItem,
    Visibility,
    content_hash,
    generate_memory_id,
    slugify,
)
from datetime import datetime


class TestSlugify:
    def test_basic(self):
        assert slugify("PAY Architecture Doc") == "pay-architecture-doc"

    def test_special_chars(self):
        assert slugify("Hello, World! (2024)") == "hello-world-2024"

    def test_unicode(self):
        assert slugify("Docs & Notes") == "docs-notes"

    def test_empty(self):
        assert slugify("") == "untitled"

    def test_truncation(self):
        long = "a" * 200
        assert len(slugify(long)) <= 120


class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_content(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_sha256_length(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA-256 hex digest


class TestGenerateMemoryId:
    def test_format(self):
        mid = generate_memory_id()
        assert mid.startswith("mem-")
        assert len(mid) == 12  # "mem-" + 8 hex chars

    def test_unique(self):
        ids = {generate_memory_id() for _ in range(100)}
        assert len(ids) == 100


class TestMemoryDataclass:
    def test_defaults(self):
        mem = Memory(
            id="mem-12345678",
            memory_type="fact",
            content="test content",
            content_hash="abc123",
        )
        assert mem.visibility == Visibility.WORKSPACE.value
        assert mem.owner_user_id is None
        assert mem.status == "active"
        assert mem.confidence == 0.7
        assert mem.corroboration_count == 1
        assert mem.contradiction_count == 0
        assert mem.entity_refs == []

    def test_custom_values(self):
        mem = Memory(
            id="mem-12345678",
            memory_type="decision",
            content="Team chose gRPC",
            content_hash="def456",
            visibility=Visibility.WORKSPACE.value,
            project_key="PAY",
            confidence=0.95,
        )
        assert mem.memory_type == "decision"
        assert mem.visibility == Visibility.WORKSPACE.value
        assert mem.project_key == "PAY"
        assert mem.confidence == 0.95


class TestMemoryStatus:
    def test_retired_replaces_decayed_as_canonical_hidden_status(self):
        assert MemoryStatus.RETIRED == "retired"


class TestRawMemory:
    def test_minimal(self):
        rm = RawMemory(content="test", memory_type="fact")
        assert rm.confidence == 0.7
        assert rm.entity_refs == []
        assert rm.valid_from is None


class TestGeneMetadata:
    def test_creation(self):
        meta = GeneMetadata(
            name="confluence",
            display_name="Confluence",
            description="Wiki pages",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )
        assert meta.name == "confluence"
        assert meta.default_sync_interval_minutes == 1440


class TestContentItem:
    def test_to_doc_ref(self):
        item = ContentItem(
            item_id="confluence-123",
            title="Test Page",
            source_url="https://wiki.example.com/123",
            last_modified=datetime(2026, 4, 8),
            space_or_project="PAY",
            version="5",
        )
        ref = item.to_doc_ref("src-conf-001")
        assert ref.doc_id == "confluence-123"
        assert ref.source == "src-conf-001"
        assert ref.title == "Test Page"
        assert ref.space_or_project == "PAY"
