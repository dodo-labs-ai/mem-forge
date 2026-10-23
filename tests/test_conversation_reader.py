"""Reading a conversation Unit's members from its current Observation revisions."""

from __future__ import annotations

import pytest

from memforge.conversations.reader import SourceProjectionUnitReader, _members
from memforge.conversations.units import UnitKind, UnitRef
from memforge.source_projection import SourceObservationRevision

UNIT = UnitRef(
    unit_id="unit-1",
    document_id="doc-1",
    provider_key="teams-block:v1:window",
    kind=UnitKind.WINDOW,
    conversation_id="19:chat@thread.v2",
)


def test_a_revision_that_is_neither_a_message_record_nor_an_artifact_fails_the_read():
    unprofiled = SourceObservationRevision(
        id="rev-1",
        observation_id="obs-1",
        semantic_hash="hash",
        content='{"content": "hello", "time": "2026-10-01T09:00:00+00:00"}',
        metadata={"provider_key": "m1"},
    )

    with pytest.raises(ValueError, match="rev-1 of Unit unit-1 is not a message record"):
        _members(UNIT, [unprofiled])


class _ProjectionWithoutRevision:
    """A projection that listed a Unit as current but holds no current revision of it."""

    async def get_current_source_unit_revision(self, source_unit_id: str):
        return None


async def test_reading_a_listed_unit_without_a_current_revision_fails():
    reader = SourceProjectionUnitReader(_ProjectionWithoutRevision())

    with pytest.raises(ValueError, match="Unit unit-1 was listed as current but has no current revision"):
        await reader.read([UNIT])
