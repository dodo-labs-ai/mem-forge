from memforge.local_agent.readiness import connection_status_from_browser_session


def test_active_browser_session_maps_to_provider_neutral_ready_status():
    assert connection_status_from_browser_session({"status": "active"}) == {
        "state": "ready",
        "reason": None,
    }


def test_missing_or_expired_browser_session_requires_authentication():
    for status in ("missing", "expired", "failed", "unknown"):
        assert connection_status_from_browser_session({"status": status}) == {
            "state": "action_required",
            "reason": "authentication",
        }


def test_principal_conflict_takes_precedence_over_authentication_status():
    assert connection_status_from_browser_session({
        "status": "active",
        "principal_changed": True,
    }) == {
        "state": "action_required",
        "reason": "identity_conflict",
    }
