"""Every registered record schema version keeps the presentation and reading it was registered with.

A revision is compared with its successor under the profile it was stored
with, so a registered version's compiled output never changes without a new
version. The golden digests in ``fixtures/registered_readings.json`` pin both
what is persisted (presentation, origins, text views) and what is compared
(each Fragment's reading and the schema's declared roles). A new schema
version adds its golden; an intended change to an existing one is a new
version instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.source_adapters.contracts import canonical_record_profile
from memforge.source_projection import SourceObservationRevision
from memforge.source_representation import _CANONICAL_RECORD_SCHEMAS

GOLDENS = Path(__file__).parent / "fixtures" / "registered_readings.json"
_AUTHOR = {"displayName": "Ana", "key": "ana"}
RECORDS = {
    "jira-issue-core": {
        "summary": "Payroll cutoff", "description": "<p>A7 processing requires approval.</p>",
        "native_description": "A7 processing requires approval.",
        "status": {"id": "1", "name": "Open", "self": "https://jira.example.test/rest/api/2/status/1"},
        "priority": {"id": "2", "name": "High"}, "assignee": {"accountId": "a1", "displayName": "Ana"},
        "labels": ["payroll"], "resolution": None, "issue_key": "PAYR-123", "issue_id": "10012",
    },
    "jira-comment": {
        "body": "<p>Retain A7 for the payroll run.</p><ul><li>Cutoff is Friday.</li></ul>",
        "native_body": "Retain A7 for the payroll run.\n\n* Cutoff is Friday.", "issue_key": "PAYR-123",
        "author": _AUTHOR, "created": "2026-07-01T10:00:00.000+0000", "updated": "2026-07-02T09:00:00.000+0000",
        "attachments": [],
    },
    "jira-changelog": {
        "id": "9", "created": "2026-07-03T08:00:00.000+0000", "author": _AUTHOR, "issue_key": "PAYR-123",
        "items": [{"field": "description", "fromString": "Previous rule.", "toString": "New rule."}],
    },
    "teams-message": {
        "content": "Decision: keep A7.\n\nRollout starts Monday.", "attachments": [], "deleted": None,
        "from": "Ana", "time": "2026-07-14T10:00:00Z", "edited_time": "2026-07-14T10:05:00Z",
        "reply_to_id": "msg-0", "conversation_id": "conv-1", "conversation_name": "team_alpha",
    },
    "confluence-page-storage": {
        "title": "Payroll rules",
        "body": "<h1>Rules</h1><p>Only when approved:</p><ul><li>A7</li></ul>"
                "<table><tr><th>Rule</th></tr><tr><td>Approval required.</td></tr></table>",
    },
    "notion-row-properties": {
        "properties": [{"name": "Status", "type": "status", "value": "Done"},
                       {"name": "Owner", "type": "people", "value": "Ana"}],
    },
}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def registered_readings(name: str, version: int) -> dict[str, str]:
    """The persisted and the compared output of one registered schema version for its sample record."""
    schema = _CANONICAL_RECORD_SCHEMAS[(name, version)]
    record = {**RECORDS[name], "representation": schema.qualified_name}
    index = build_revision_fragment_index(SourceObservationRevision(
        id="rev", observation_id="obs", semantic_hash="hash", content=json.dumps(record, sort_keys=True),
        evidence_profile=canonical_record_profile(name, version),
    ))
    assert index.complete and index.fragments
    fragments = index.fragments
    return {
        "presentation": _digest([
            (f.fragment_type, f.anchor.range_start, f.anchor.range_end, f.primary_eligible, f.raw_content_sha256,
             f.presentation_text, f.presentation_sha256, f.content_value, f.interpretation_origins,
             (f.text_view.kind, f.text_view.version,
              [(o.anchor.range_start, o.anchor.range_end, o.raw_content_sha256) for o in f.text_view.origins]))
            for f in fragments
        ]),
        "reading": _digest({
            "fragments": [
                (f.fragment_type, f.anchor.range_start, f.anchor.range_end, f.reading.core, f.reading.field,
                 f.reading.role, f.reading.label, f.reading.framing)
                for f in fragments
            ],
            "displayed": [field.json_pointer for field in schema.fields if not field.governs_interpretation],
            "model_interpretation": schema.model_interpretation,
        }),
    }


def test_every_registered_schema_version_has_a_golden_reading() -> None:
    goldens = json.loads(GOLDENS.read_text())
    assert sorted(goldens) == sorted(f"{name}:{version}" for name, version in _CANONICAL_RECORD_SCHEMAS)


@pytest.mark.parametrize("name, version", sorted(_CANONICAL_RECORD_SCHEMAS))
def test_a_registered_schema_version_reads_as_it_was_registered(name, version) -> None:
    goldens = json.loads(GOLDENS.read_text())

    assert registered_readings(name, version) == goldens[f"{name}:{version}"]
