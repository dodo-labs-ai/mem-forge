import pytest

from memforge.memory.origin import MemoryOriginKind, classify_memory_origin, references_source_row


def test_memory_origin_kind_is_derived_from_the_provenance_source_type() -> None:
    assert classify_memory_origin("user_memory") is MemoryOriginKind.DIRECT_USER
    assert classify_memory_origin("user_correction") is MemoryOriginKind.DIRECT_USER
    assert classify_memory_origin("agent_session") is MemoryOriginKind.MANAGED_CAPTURE
    assert classify_memory_origin("jira") is MemoryOriginKind.CONFIGURED_SOURCE


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        (None, False),
        ("user_memory", False),
        ("user_correction", False),
        ("src-jira", True),
        ("src-managed-capture", True),
        ("src-missing", True),
        ("", True),
    ],
)
def test_source_row_reference_does_not_infer_existence(source_id, expected) -> None:
    assert references_source_row(source_id) is expected
