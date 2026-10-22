"""The Gene of a conversation connector: the sync run collects it through the conversation layer.

A connector whose Gene is a :class:`ConversationGene` belongs to the
conversation family; nothing else declares the family. The sync run binds the
Gene to its collection attempt, and discovery then plans from that attempt's
Collection Manifest: each Unit to project becomes one item, and the Units to
remove wait with their reasons for the run's absence step. A conversation
Source never proves absence from a scope listing or from a run-level
complete snapshot.

The connector's executor collects on a user's device and uploads the
manifest before the run starts; the Gene issues the collection request the
attempt answers (:meth:`ConversationGene.issue_collection_request`).
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memforge.conversations.collections import StoredCollections
from memforge.conversations.layer import ConversationLayer, ConversationSelection, SourcePlan
from memforge.conversations.manifest import CollectionManifest, CollectionRequest
from memforge.conversations.port import ConversationAccess
from memforge.conversations.projection import (
    is_unit_payload,
    normalize_unit,
    unit_content_item,
    unit_payload,
    unit_raw_content,
)
from memforge.conversations.reader import (
    ConversationProjectionStore,
    ConversationUnitReader,
    SourceProjectionUnitReader,
)
from memforge.genes.base import Gene
from memforge.genes.local_adapter_packages import open_packaged_source_artifact
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.source_projection import SourceRemoval

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollectionAttempt:
    """The collection attempt one Source sync run of a conversation Source belongs to."""

    workspace_id: str
    collection_attempt_id: str


class ConversationRunNotBound(RuntimeError):
    """The Gene was used for a sync run before the run bound it to its collection attempt."""


class CollectionMissing(RuntimeError):
    """The collection attempt the run belongs to has no Collection Manifest."""


class ConversationGene(Gene):
    """A conversation connector's Gene."""

    record_schema: str
    """The canonical record schema of the connector's messages, as ``name:version``."""

    @abstractmethod
    def conversation_selection(self) -> ConversationSelection:
        """The selected conversations and time-scope policy of this Source's configuration."""

    @abstractmethod
    def selected_conversation_ids(self, projection_scope: Mapping[str, Any]) -> tuple[str, ...]:
        """The conversations a projection scope of this Source selects."""

    def stored_unit_payload(self, body: bytes) -> Mapping[str, Any] | None:
        """The conversation Unit payload of a stored input this connector wrote in another form, if it did."""

        return None

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._run: _ConversationRun | None = None
        self._payloads: dict[str, Mapping[str, Any]] = {}

    def bind_conversation_run(
        self,
        *,
        projection_store: ConversationProjectionStore,
        collections: StoredCollections,
        collection_attempt_id: str,
    ) -> None:
        """Bind discovery to one collection attempt of this Source."""

        reader = SourceProjectionUnitReader(projection_store)
        self._run = _ConversationRun(
            layer=ConversationLayer(reader),
            reader=reader,
            collections=collections,
            collection_attempt_id=collection_attempt_id,
        )

    async def issue_collection_request(
        self, collections: StoredCollections, collection_attempt_id: str, *, now: datetime
    ) -> CollectionRequest:
        """What one collection attempt of this Source collects, continuing from the attempt the Source received last.

        The attempt keeps the first request issued to it, so its manifest
        answers one request however often the executor asks.
        """

        request = ConversationLayer.collection_request(
            selection=self.conversation_selection(),
            now=now,
            previous=await collections.latest_manifest(self.source_id),
        )
        return await collections.issue_request(self.source_id, collection_attempt_id, request, issued_at=now)

    @property
    def collection_attempt_id(self) -> str:
        return self._require_run().collection_attempt_id

    def begin_scope_transition(self, *, previous_scope: Mapping[str, Any], target_scope: Mapping[str, Any]) -> None:
        """Remove, in this run, every conversation the transition deselects."""

        run = self._require_run()
        run.deselected = frozenset(self.selected_conversation_ids(previous_scope)) - frozenset(
            self.selected_conversation_ids(target_scope)
        )

    def removals(self) -> dict[str, SourceRemoval]:
        """The Documents this attempt removes, each with its reason and the attempt that decided it.

        Each conversation's count of removed Units, by reason, is logged, so a
        run that removes far more than its conversations changed stands out.
        """

        evidence = f"collection attempt {self.collection_attempt_id}"
        removals = self._require_plan().removals
        counts = Counter((removal.unit.conversation_id, removal.reason.value) for removal in removals)
        for (conversation_id, reason), units in sorted(counts.items()):
            logger.info(
                "conversation_units_removed source=%s attempt=%s conversation=%s reason=%s units=%d",
                self.source_id,
                self.collection_attempt_id,
                conversation_id,
                reason,
                units,
            )
        return {
            removal.unit.document_id: SourceRemoval(reason=removal.reason, evidence=evidence)
            for removal in removals
        }

    def unreadable_conversations(self) -> tuple[str, ...]:
        """The name of each selected conversation this attempt could not read.

        Such a conversation keeps its Support; the run reports it so the user
        can restore access or deselect it. It is named by the label the Source
        last read it with, or by its id when it was never read.
        """

        return tuple(
            (evidence.label.name if evidence.label is not None else None) or evidence.conversation_id
            for evidence in self._require_plan().unreadable
        )

    async def completes_scope_transition(self) -> bool:
        """Whether this attempt, once its removals committed, leaves the Source in its target scope.

        Every target conversation the Source can read has poll evidence in the
        manifest, and no current Unit belongs to a conversation outside the
        selection. A target the Source cannot read, or that is gone, does not
        hold the transition open.
        """

        run = self._require_run()
        manifest = run.manifest
        if manifest is None or run.plan is None:
            raise RuntimeError("the conversation plan exists once discovery has run")
        selected = set(self.conversation_selection().conversation_ids)
        for conversation_id in selected:
            evidence = manifest.conversation(conversation_id)
            if evidence is None:
                return False
            if evidence.status.access is ConversationAccess.READABLE and not evidence.polls:
                return False
        return await run.reader.conversations_held(self.source_id) <= selected

    async def list_scope(self) -> None:
        return None

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Plan this attempt from its Collection Manifest and yield each Unit to project; ``since`` is unused."""

        plan = await self.preview()
        record_schema = self._require_run().record_schema
        self._payloads.clear()
        for unit in plan.units:
            item = unit_content_item(unit, source_id=self.source_id)
            self._payloads[item.item_id] = unit_payload(unit, record_schema=record_schema)
            yield item

    async def preview(self) -> SourcePlan:
        """Plan this attempt from its Collection Manifest without projecting anything."""

        run = self._require_run()
        manifest = await run.collections.manifest(self.source_id, run.collection_attempt_id)
        if manifest is None:
            raise CollectionMissing(f"collection attempt {run.collection_attempt_id} has no Collection Manifest")
        run.manifest = manifest
        bodies = await run.collections.bodies(self.source_id, run.collection_attempt_id)
        run.plan = await run.layer.plan(
            manifest=manifest,
            bodies={identity: body.record for identity, body in bodies.items()},
            uploaded_artifacts=await run.collections.uploaded_artifacts(self.source_id, bodies),
            selection=self.conversation_selection(),
            attempt_started_at=manifest.received_at,
            deselected_conversation_ids=run.deselected,
        )
        return run.plan

    async def fetch(self, item: ContentItem) -> RawContent:
        payload = self._payloads.get(item.item_id)
        if payload is None:
            raise ValueError(f"conversation Unit {item.item_id} was not planned in this run")
        return unit_raw_content(item, payload, with_artifacts=True)

    def raw_from_stored_input(self, item: ContentItem, body: bytes, content_type: str) -> RawContent:
        payload = self.stored_unit_payload(body)
        if payload is None:
            payload = json.loads(body.decode("utf-8"))
        if not is_unit_payload(payload):
            raise ValueError(f"stored input of {item.item_id} is not a conversation Unit")
        return unit_raw_content(item, payload, with_artifacts=False)

    def open_source_artifact(self, artifact):
        """Open the stored bytes of one member artifact: an upload of this attempt, or a committed copy."""

        return open_packaged_source_artifact(self, artifact)

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        return normalize_unit(raw)

    def _require_run(self) -> _ConversationRun:
        if self._run is None:
            raise ConversationRunNotBound("the sync run did not bind this Gene to a collection attempt")
        return self._run

    def _require_plan(self) -> SourcePlan:
        plan = self._require_run().plan
        if plan is None:
            raise RuntimeError("the conversation plan exists once discovery has run")
        return plan


class _ConversationRun:
    def __init__(
        self,
        *,
        layer: ConversationLayer,
        reader: ConversationUnitReader,
        collections: StoredCollections,
        collection_attempt_id: str,
    ) -> None:
        self.layer = layer
        self.reader = reader
        self.collections = collections
        self.collection_attempt_id = collection_attempt_id
        self.deselected: frozenset[str] = frozenset()
        self.manifest: CollectionManifest | None = None
        self.plan: SourcePlan | None = None

    @property
    def record_schema(self) -> str:
        if self.manifest is None:
            raise RuntimeError("the collection's record schema is known once its manifest is read")
        return self.manifest.record_schema
