"""Gateway tool-use responses sometimes serialize container fields as JSON
strings (for example ``{"memories": "[...]"}``). The structured schemas must
decode those before validation so a stringified container still parses.
"""

from memforge.llm.structured import (
    MemoryCandidate,
    MemoryExtractionResponse,
    RerankResponse,
)


def test_memories_stringified_array_is_decoded():
    payload = {"memories": '[{"content": "pay-api uses PostgreSQL 15", "memory_type": "fact"}]'}
    parsed = MemoryExtractionResponse.model_validate(payload)
    assert len(parsed.memories) == 1
    assert parsed.memories[0].memory_type == "fact"


def test_native_array_is_unchanged():
    payload = {"memories": [{"content": "x", "memory_type": "fact"}]}
    parsed = MemoryExtractionResponse.model_validate(payload)
    assert parsed.memories[0].content == "x"


def test_nested_stringified_list_field_is_decoded():
    payload = {"memories": [{"content": "x", "memory_type": "fact", "entity_refs": '["pay-api", "postgresql"]'}]}
    parsed = MemoryExtractionResponse.model_validate(payload)
    assert parsed.memories[0].entity_refs == ["pay-api", "postgresql"]


def test_string_scalar_field_is_not_decoded():
    cand = MemoryCandidate.model_validate({"content": "[brackets] in text", "memory_type": "fact"})
    assert cand.content == "[brackets] in text"


def test_rerank_stringified_int_list_is_decoded():
    assert RerankResponse.model_validate({"ranking": "[3, 1, 2]"}).ranking == [3, 1, 2]
