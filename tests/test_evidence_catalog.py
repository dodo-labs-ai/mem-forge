from __future__ import annotations

from memforge.models import MemoryExtractionResult, RawMemory
from memforge.pipeline.evidence_catalog import (
    EvidenceAuthoritySpan,
    EvidenceCatalog,
)
from memforge.source_derivation import (
    memory_extraction_output_payload,
    memory_extraction_result_from_output_payload,
)


def _memory(*, block_id: str | None, quote: str | None) -> RawMemory:
    return RawMemory(
        content="Durable tracing procedure.",
        memory_type="procedure",
        evidence_block_id=block_id,
        evidence_quote=quote,
    )


def test_valid_block_id_admits_candidate_when_quote_cannot_be_refined() -> None:
    source = "validation\\_rule\\_agent records traceId for later CLS lookup."
    catalog = EvidenceCatalog.from_text(source, observation_id="obs-message")

    resolved = catalog.resolve(
        _memory(
            block_id="EB-001",
            quote="a paraphrase that is not source text",
        )
    )

    assert resolved is not None
    assert resolved.refinement == "block_fallback"
    assert resolved.memory.evidence_quote == source
    assert resolved.memory.extraction_context == source
    assert resolved.memory.source_observation_id == "obs-message"
    assert resolved.memory.evidence_block_id is None
    assert resolved.memory.evidence_resolved_from_block is True
    assert resolved.memory.evidence_range_start == 0
    assert resolved.memory.evidence_range_end == len(source)


def test_markdown_escape_difference_refines_to_exact_source_text() -> None:
    source = "validation\\_rule\\_agent records x\\_request\\_id."
    catalog = EvidenceCatalog.from_text(source)

    resolved = catalog.resolve(
        _memory(
            block_id="EB-001",
            quote="validation_rule_agent records x_request_id.",
        )
    )

    assert resolved is not None
    assert resolved.refinement == "canonical_quote"
    assert resolved.memory.evidence_quote == source
    assert resolved.memory.evidence_range_start == 0
    assert resolved.memory.evidence_range_end == len(source)


def test_fullwidth_markdown_escape_difference_refines_to_exact_source_text() -> None:
    source = "validation＼_rule＼_agent records x＼_request＼_id."
    catalog = EvidenceCatalog.from_text(source)

    resolved = catalog.resolve(
        _memory(
            block_id="EB-001",
            quote="validation_rule_agent records x_request_id.",
        )
    )

    assert resolved is not None
    assert resolved.refinement == "canonical_quote"
    assert resolved.memory.evidence_quote == source


def test_markdown_link_label_or_url_can_refine_to_exact_source_span() -> None:
    source = 'See [MCP guide](https://example.test/mcp "MCP guide") before requesting CAM.'
    catalog = EvidenceCatalog.from_text(source)

    label = catalog.resolve(_memory(block_id="EB-001", quote="MCP guide"))
    url = catalog.resolve(
        _memory(block_id="EB-001", quote="https://example.test/mcp")
    )
    expanded = catalog.resolve(
        _memory(
            block_id="EB-001",
            quote="MCP guide (https://example.test/mcp)",
        )
    )

    assert label is not None and label.memory.evidence_quote == "MCP guide"
    assert url is not None and url.memory.evidence_quote == "https://example.test/mcp"
    assert expanded is not None
    assert expanded.memory.evidence_quote == (
        '[MCP guide](https://example.test/mcp "MCP guide")'
    )


def test_invalid_block_id_rejects_even_when_quote_appears_elsewhere() -> None:
    catalog = EvidenceCatalog.from_text("A durable rule.")

    assert (
        catalog.resolve(
            _memory(block_id="EB-999", quote="A durable rule.")
        )
        is None
    )


def test_quote_only_compatibility_requires_one_unique_block() -> None:
    catalog = EvidenceCatalog.from_text("Keep A7.\n\nKeep A7.")

    assert catalog.resolve(_memory(block_id=None, quote="Keep A7.")) is None


def test_changed_ranges_create_only_selectable_changed_blocks() -> None:
    document = "Old durable rule.\nNew durable rule.\nUnaffected context."
    start = document.index("New durable rule.")
    end = start + len("New durable rule.")
    catalog = EvidenceCatalog.from_spans(
        (EvidenceAuthoritySpan(document[start:end], source_start=start),)
    )

    assert [block.text for block in catalog.blocks] == ["New durable rule."]
    assert catalog.blocks[0].source_start == start
    assert catalog.blocks[0].source_end == end


def test_large_text_is_bounded_without_splitting_utf8_characters() -> None:
    catalog = EvidenceCatalog.from_text("中" * 5_000, max_block_bytes=101)

    assert len(catalog.blocks) > 1
    assert all(len(block.text.encode("utf-8")) <= 101 for block in catalog.blocks)
    assert "".join(block.text for block in catalog.blocks) == "中" * 5_000


def test_batch_local_block_id_is_not_persisted_in_derivation_output() -> None:
    resolved = EvidenceCatalog.from_text("A durable rule.").resolve(
        _memory(block_id="EB-001", quote="A durable rule.")
    )
    assert resolved is not None

    payload = memory_extraction_output_payload(
        MemoryExtractionResult(memories=[resolved.memory])
    )
    restored = memory_extraction_result_from_output_payload(payload)

    assert "evidence_block_id" not in payload["memories"][0]
    assert restored.memories[0].evidence_block_id is None
    assert restored.memories[0].evidence_resolved_from_block is True
    assert restored.memories[0].evidence_range_start == 0
    assert restored.memories[0].evidence_range_end == len("A durable rule.")


def test_quote_only_compatibility_rebinds_one_unique_block_despite_stale_hint() -> None:
    catalog = EvidenceCatalog.from_spans(
        (
            EvidenceAuthoritySpan("First rule.", observation_id="obs-first"),
            EvidenceAuthoritySpan("Second rule.", observation_id="obs-second"),
        )
    )
    memory = _memory(block_id=None, quote="Second rule.")
    memory.source_observation_id = "obs-stale"

    resolved = catalog.resolve(memory)

    assert resolved is not None
    assert resolved.memory.source_observation_id == "obs-second"
