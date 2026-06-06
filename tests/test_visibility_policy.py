# tests/test_visibility_policy.py
from memforge.memory.visibility_policy import default_visibility


def test_agent_session_defaults_to_private_when_user_id_provided():
    from memforge.models import Visibility
    vis, owner = default_visibility("agent_session", user_id="u-7")
    assert vis == Visibility.PRIVATE.value
    assert owner == "u-7"


def test_agent_session_falls_back_to_local_dev_user():
    from memforge.models import Visibility
    from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
    vis, owner = default_visibility("agent_session", user_id=None)
    assert vis == Visibility.PRIVATE.value
    assert owner == LOCAL_DEV_USER_ID


def test_other_sources_stay_workspace():
    from memforge.models import Visibility
    vis, owner = default_visibility("confluence", user_id="u-7")
    assert vis == Visibility.WORKSPACE.value
    assert owner is None
    vis, owner = default_visibility("jira", user_id="u-7")
    assert vis == Visibility.WORKSPACE.value
    assert owner is None
    vis, owner = default_visibility(None, user_id="u-7")
    assert vis == Visibility.WORKSPACE.value
    assert owner is None
