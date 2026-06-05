from memforge.models import Memory, Visibility, content_hash


def test_visibility_enum_values():
    assert Visibility.WORKSPACE.value == "workspace"
    assert Visibility.PRIVATE.value == "private"
    assert {v.value for v in Visibility} == {"workspace", "private"}


def test_memory_defaults_to_workspace_without_owner():
    memory = Memory(
        id="mem-1",
        memory_type="fact",
        content="x",
        content_hash=content_hash("x"),
    )
    assert memory.visibility == Visibility.WORKSPACE.value
    assert memory.owner_user_id is None
    assert not hasattr(memory, "scope")
