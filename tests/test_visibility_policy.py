# tests/test_visibility_policy.py
from memforge.memory.visibility_policy import default_visibility


def test_all_sources_default_to_workspace():
    from memforge.models import Visibility
    # Every source currently writes workspace rows; private intake is deferred until
    # the read-side access predicate can filter a private row.
    expected = (Visibility.WORKSPACE.value, None)
    assert default_visibility("agent_session") == expected
    assert default_visibility("confluence") == expected
    assert default_visibility("jira") == expected
    assert default_visibility(None) == expected
