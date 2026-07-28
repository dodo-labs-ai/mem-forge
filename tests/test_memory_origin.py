from memforge.memory.origin import MemoryOriginKind, classify_memory_origin


def test_memory_origin_kind_is_derived_from_the_provenance_source_type() -> None:
    assert classify_memory_origin("user_memory") is MemoryOriginKind.DIRECT_USER
    assert classify_memory_origin("user_correction") is MemoryOriginKind.DIRECT_USER
    assert classify_memory_origin("agent_session") is MemoryOriginKind.MANAGED_CAPTURE
    assert classify_memory_origin("jira") is MemoryOriginKind.CONFIGURED_SOURCE
