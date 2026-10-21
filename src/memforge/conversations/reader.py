"""The reads the conversation layer makes (:class:`ConversationUnitReader`), answered from the current Source Projection.

A Unit's membership is the member records of its current revision, each with
the Source Artifacts its revision holds for it. A Unit is found by the
conversation its locator records, its kind and recorded span, or a message it
holds.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Protocol

from memforge.conversations.records import MESSAGE_OBSERVATION_TYPE, MessageIdentity, record_revision
from memforge.conversations.undated import stored_member_time
from memforge.conversations.units import Member, MemberArtifact, StoredUnit, UnitKind, UnitRef
from memforge.source_artifacts import source_artifact_revision_from_metadata, stored_source_artifact_from_observation
from memforge.source_projection import (
    SourceObservationRevision,
    SourceUnit,
    SourceUnitInventoryFilter,
    SourceUnitInventoryPage,
    SourceUnitRevision,
)

_PAGE_SIZE = 500


class ConversationProjectionStore(Protocol):
    async def list_current_source_units_page(
        self, source_id: str, *, filters: SourceUnitInventoryFilter, cursor: str | None = None, limit: int = 200
    ) -> SourceUnitInventoryPage: ...

    async def list_current_observation_holders(
        self,
        source_id: str,
        *,
        observation_type: str,
        provider_keys: Collection[str],
        locator_equals: Mapping[str, str],
    ) -> tuple[tuple[str, SourceUnit], ...]: ...

    async def get_current_source_unit_revision(self, source_unit_id: str) -> SourceUnitRevision | None: ...

    async def get_source_observation_revisions(
        self, revision_ids: Collection[str]
    ) -> Mapping[str, SourceObservationRevision]: ...


class ConversationUnitReader(Protocol):
    """The current Units of one conversation, as the conversation layer reads them.

    Every method returns current Units only, fully paged. A ``StoredUnit``
    carries its complete current membership; a ``UnitRef`` names a Unit
    without reading its members.
    """

    async def holders(
        self, source_id: str, conversation_id: str, message_ids: Collection[str]
    ) -> Mapping[str, tuple[UnitRef, ...]]:
        """The Units holding each of these messages, by message id."""
        ...

    async def thread_units(self, source_id: str, conversation_id: str) -> tuple[UnitRef, ...]:
        """Every thread Unit of the conversation."""
        ...

    async def read(self, units: Collection[UnitRef]) -> tuple[StoredUnit, ...]:
        """These Units with their members."""
        ...

    async def units_spanning(
        self, source_id: str, conversation_id: str, start: datetime, end: datetime
    ) -> tuple[StoredUnit, ...]:
        """The Units whose recorded span meets ``[start, end]``."""
        ...

    async def units_ending_before(
        self, source_id: str, conversation_id: str, cutoff: datetime
    ) -> tuple[StoredUnit, ...]:
        """The Units whose recorded span ends before ``cutoff``."""
        ...

    async def units_of(self, source_id: str, conversation_id: str) -> tuple[UnitRef, ...]:
        """Every Unit of the conversation."""
        ...

    async def conversations_held(self, source_id: str) -> frozenset[str]:
        """The conversations of every current conversation Unit of the Source."""
        ...


class SourceProjectionUnitReader:
    """Reads conversation Units from the current Source Projection; implements ``ConversationUnitReader``."""

    def __init__(self, store: ConversationProjectionStore) -> None:
        self._store = store

    async def holders(
        self, source_id: str, conversation_id: str, message_ids: Collection[str]
    ) -> dict[str, tuple[UnitRef, ...]]:
        if not message_ids:
            return {}
        holders: dict[str, list[UnitRef]] = {}
        for message_id, unit in await self._store.list_current_observation_holders(
            source_id,
            observation_type=MESSAGE_OBSERVATION_TYPE,
            provider_keys=message_ids,
            locator_equals={"conversation_id": conversation_id},
        ):
            holders.setdefault(message_id, []).append(unit_ref(unit))
        return {message_id: tuple(units) for message_id, units in holders.items()}

    async def thread_units(self, source_id: str, conversation_id: str) -> tuple[UnitRef, ...]:
        return tuple(
            unit_ref(unit)
            for unit in await self._list(
                source_id,
                SourceUnitInventoryFilter(
                    unit_type=UnitKind.THREAD.value, locator_equals={"conversation_id": conversation_id}
                ),
            )
        )

    async def read(self, units: Collection[UnitRef]) -> tuple[StoredUnit, ...]:
        stored: list[StoredUnit] = []
        for ref in {unit.unit_id: unit for unit in units}.values():
            revision = await self._store.get_current_source_unit_revision(ref.unit_id)
            if revision is None:
                continue
            observations = await self._store.get_source_observation_revisions(revision.observation_revision_ids)
            stored.append(StoredUnit(ref=ref, members=_members(ref, observations.values())))
        return tuple(stored)

    async def units_spanning(
        self, source_id: str, conversation_id: str, start: datetime, end: datetime
    ) -> tuple[StoredUnit, ...]:
        return await self.read(
            await self._of_kinds(source_id, conversation_id, observed_from_lte=end.isoformat(), observed_to_gte=start.isoformat())
        )

    async def units_ending_before(
        self, source_id: str, conversation_id: str, cutoff: datetime
    ) -> tuple[StoredUnit, ...]:
        return await self.read(await self._of_kinds(source_id, conversation_id, observed_to_lt=cutoff.isoformat()))

    async def units_of(self, source_id: str, conversation_id: str) -> tuple[UnitRef, ...]:
        return await self._of_kinds(source_id, conversation_id)

    async def conversations_held(self, source_id: str) -> frozenset[str]:
        """The conversations of every current conversation Unit of the Source."""

        held: set[str] = set()
        for kind in UnitKind:
            held.update(
                unit_ref(unit).conversation_id
                for unit in await self._list(source_id, SourceUnitInventoryFilter(unit_type=kind.value))
            )
        return frozenset(held)

    async def _of_kinds(self, source_id: str, conversation_id: str, **span: str) -> tuple[UnitRef, ...]:
        units: list[UnitRef] = []
        for kind in UnitKind:
            units.extend(
                unit_ref(unit)
                for unit in await self._list(
                    source_id,
                    SourceUnitInventoryFilter(
                        unit_type=kind.value, locator_equals={"conversation_id": conversation_id}, **span
                    ),
                )
            )
        return tuple(units)

    async def _list(self, source_id: str, filters: SourceUnitInventoryFilter) -> list[SourceUnit]:
        units: list[SourceUnit] = []
        cursor: str | None = None
        while True:
            page = await self._store.list_current_source_units_page(
                source_id, filters=filters, cursor=cursor, limit=_PAGE_SIZE
            )
            units.extend(page.units)
            if page.next_cursor is None:
                return units
            cursor = page.next_cursor


def unit_ref(unit: SourceUnit) -> UnitRef:
    """A conversation Unit as its locator records it; a Unit without its Document or conversation is refused."""

    document_id = str(unit.locator.get("document_id") or "")
    conversation_id = str(unit.locator.get("conversation_id") or "")
    if not document_id or not conversation_id:
        raise ValueError(f"conversation Unit {unit.id} does not record its Document and conversation")
    kind = UnitKind(unit.unit_type)
    thread_root_id = unit.locator.get("thread_root_id")
    return UnitRef(
        unit_id=unit.id,
        document_id=document_id,
        provider_key=unit.provider_key,
        kind=kind,
        conversation_id=conversation_id,
        thread_root_id=str(thread_root_id) if kind is UnitKind.THREAD and thread_root_id else None,
    )


def _members(ref: UnitRef, revisions: Collection[SourceObservationRevision]) -> tuple[Member, ...]:
    """The message members of one Unit revision, each with the artifacts its revision holds for it."""

    records: dict[str, tuple[str, Mapping[str, object], str | None]] = {}
    artifacts: dict[str, list[MemberArtifact]] = {}
    for revision in revisions:
        provider_key = str(revision.metadata.get("provider_key") or "")
        artifact = source_artifact_revision_from_metadata(
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            source_id="",
            source_unit_id=ref.unit_id,
            metadata=revision.metadata,
        )
        if artifact is not None:
            stored = stored_source_artifact_from_observation(
                revision=artifact,
                observation_provider_key=provider_key,
                locator={},
                parent_observation_type=MESSAGE_OBSERVATION_TYPE,
                parent_provider_key="",
            )
            artifacts.setdefault(artifact.parent_observation_id, []).append(
                MemberArtifact(
                    provider_key=stored.provider_key,
                    provider_revision=stored.provider_revision,
                    filename=stored.filename,
                    media_type=stored.media_type,
                    size_bytes=stored.size_bytes,
                    sha256=stored.sha256,
                    uri=stored.uri,
                )
            )
        elif "source_artifact" in revision.metadata:
            raise ValueError(f"Source Artifact revision {revision.id} of Unit {ref.unit_id} cannot be read")
        elif provider_key and revision.evidence_profile is not None and revision.evidence_profile.schema_name:
            records[revision.observation_id] = (provider_key, json.loads(revision.content), revision.observed_at)
    members = [
        Member(
            identity=MessageIdentity(ref.conversation_id, message_id),
            created_at=stored_member_time(record, observed_at=observed_at),
            record=record,
            revision=record_revision(record),
            artifacts=tuple(sorted(artifacts.get(observation_id, ()), key=lambda artifact: artifact.provider_key)),
        )
        for observation_id, (message_id, record, observed_at) in records.items()
    ]
    return tuple(sorted(members, key=lambda member: (member.created_at, member.identity.message_id)))
