"""Origin-source selection for the leading source glyph in the admin UI.

`_pick_origin_source_type` decides which of a memory's sources represents it in
list and detail views: the extraction origin when present, else the first
attached source. Pairs are (source_type, support_kind, client), ordered oldest-first.
The function returns (source_type, client).
"""

from memforge.server.admin_api import _pick_origin_source_type


def test_no_sources_has_no_origin():
    assert _pick_origin_source_type([]) == (None, None)


def test_extracted_origin_wins_over_corroborated():
    pairs = [("jira", "corroborated", None), ("confluence", "extracted", None)]
    source_type, client = _pick_origin_source_type(pairs)
    assert source_type == "confluence"
    assert client is None


def test_first_source_used_without_extracted():
    pairs = [("teams", "corroborated", None), ("jira", "corroborated", None)]
    source_type, client = _pick_origin_source_type(pairs)
    assert source_type == "teams"
    assert client is None


def test_missing_support_kind_falls_back_to_first():
    source_type, client = _pick_origin_source_type([("github_pages", None, None)])
    assert source_type == "github_pages"
    assert client is None


def test_first_extracted_wins_when_several():
    pairs = [("teams", "corroborated", None), ("jira", "extracted", None), ("confluence", "extracted", None)]
    source_type, client = _pick_origin_source_type(pairs)
    assert source_type == "jira"
    assert client is None


def test_extracted_origin_returns_client_for_agent_session():
    pairs = [("agent_session", "extracted", "codex"), ("jira", "corroborated", None)]
    source_type, client = _pick_origin_source_type(pairs)
    assert source_type == "agent_session"
    assert client == "codex"


def test_first_source_returns_client_when_no_extracted():
    pairs = [("agent_session", "corroborated", "claude-code")]
    source_type, client = _pick_origin_source_type(pairs)
    assert source_type == "agent_session"
    assert client == "claude-code"
