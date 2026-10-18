"""Model settings: reading, saving and validating the model bindings of one server.

Settings edits the candidate binding of a kind and never the active one. A
candidate becomes the active binding only when a capability check, made
through the runtime's own clients, passes. The active binding stays in use
whatever its state; its state says whether the latest check of it passed.

A binding that is read from the environment is never stored, so the outcome of
its check is kept by this object for as long as the server runs.

One check of a binding runs at a time. A caller that asks for a check while one
of the same binding is running waits for it and takes its outcome, so no number
of simultaneous requests makes more than one model call.

The embedding binding and the Memory vector index belong together: an index
holds the vectors of one Embedding Profile. An embedding candidate that passes
its check becomes active at once only when its profile is the one the index in
use was built with. Otherwise it waits, validated, for a rebuild of the index,
which is started on request, runs beside the server and makes the candidate
active in the same transaction that puts the rebuilt index into use.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from memforge.config import AppConfig
from memforge.llm.bindings import (
    BindingSlot,
    BindingState,
    KeySource,
    ModelBinding,
    ValidationResult,
    binding_from_record,
    binding_record,
    build_binding,
    embedding_identity,
    embedding_profile,
    key_last4,
    key_source,
    new_stored_secret,
    resolve_active_binding,
    resolve_credential,
    resolve_invocation,
    state_after_change,
    state_after_check,
    stored_secret_id,
    validation_record,
)
from memforge.llm.capability import (
    GENERATION_CHECK_TIMEOUT_S,
    CapabilityCheck,
    check_embedding,
    check_generation,
)
from memforge.llm.providers import (
    MODEL_KINDS,
    AuthStrategy,
    ModelConfigError,
    ModelKind,
    ResolvedInvocation,
    provider_descriptor,
)
from memforge.llm.embedding_profile import EmbeddingProfile
from memforge.runtime import (
    ResolvedModels,
    RuntimeProvider,
    generation_token_limits,
    memory_vector_indexes,
)
from memforge.vector_index_errors import (
    REINDEX_IN_PROGRESS,
    REINDEX_NOT_REQUIRED,
    REINDEX_NOT_REQUIRED_MESSAGE,
    VectorReindexRefused,
)

if TYPE_CHECKING:
    from memforge.memory.reindex import MemoryVectorIndexes, ReindexProgress, ReindexTarget
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "BindingView",
    "CandidateDraft",
    "KeyView",
    "KindView",
    "ListedNativeSchema",
    "ModelHealth",
    "ModelSettings",
    "ValidationOutcome",
    "VectorIndexStatus",
    "VectorIndexView",
    "check_binding",
]

# What the model listing of a binding's provider declares about the binding's
# model answering in a JSON schema, given the binding and its API key; ``None``
# when the listing does not answer, does not hold the model or declares nothing.
type ListedNativeSchema = Callable[[ModelBinding, str | None], Awaitable[bool | None]]

# ``ready``: the embedding model in use is the one the index was built with.
# ``reindex_required``: it is another one, or the stored vectors cannot be
# adopted; vector search is off until the index is rebuilt. ``reindexing``: a
# rebuild is running. ``awaiting_validation``: no index is in use yet, because
# the embedding model has not passed a check. ``not_configured``: no embedding
# model. ``not_managed``: the deployment keeps the index by its own means.
VectorIndexStatus = Literal[
    "ready",
    "reindex_required",
    "reindexing",
    "awaiting_validation",
    "not_configured",
    "not_managed",
]

@dataclass(frozen=True)
class CandidateDraft:
    """The settings of a candidate as Settings submits them."""

    provider: str
    model_id: str
    endpoint: str | None = None
    auth_strategy: AuthStrategy | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    token_limits: Mapping[str, Any] = field(default_factory=dict)
    # Whether the model answers in a response schema, as the model listing
    # declares it; ``None`` when it declares nothing.
    native_schema: bool | None = None
    # A new API key. ``None`` keeps the key already saved for the endpoint.
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class KeyView:
    """What may be shown of a binding's API key: where it is kept and how it ends."""

    source: KeySource | None = None
    is_set: bool = False
    last4: str | None = None


@dataclass(frozen=True)
class BindingView:
    binding: ModelBinding
    key: KeyView
    # The Embedding Profile of an embedding binding whose latest check vouches for it.
    profile: EmbeddingProfile | None = None
    # Whether this embedding candidate has passed its check and becomes active
    # once the vector index is rebuilt for it.
    waits_for_reindex: bool = False


@dataclass(frozen=True)
class VectorIndexView:
    """The Memory vector index as Settings shows it."""

    status: VectorIndexStatus
    # Whether search reads the index in use: there is one, and the embedding
    # model in use is the one it was built with. A rebuild does not change
    # this before its switch.
    serves_search: bool = False
    # The binding a rebuild started now would build for; ``None`` when there
    # is nothing to rebuild for.
    reindex_target: BindingSlot | None = None
    # The record of the index in use.
    in_use: Mapping[str, Any] | None = None
    # The vectors the index in use holds now, when they were counted.
    vector_count: int | None = None
    # The record of the latest rebuild, whatever became of it.
    reindex: Mapping[str, Any] | None = None
    reindex_running: bool = False

    @property
    def reindex_required(self) -> bool:
        return self.status == "reindex_required"


@dataclass(frozen=True)
class KindView:
    kind: ModelKind
    active: BindingView | None
    candidate: BindingView | None


@dataclass(frozen=True)
class ValidationOutcome:
    """The result of one capability check and what it did to the bindings."""

    kind: ModelKind
    validation: ValidationResult
    # Whether the checked candidate became the active binding.
    promoted: bool
    state: BindingState
    # Whether the candidate passed and now waits for a rebuild of the vector index.
    reindex_required: bool = False


@dataclass(frozen=True)
class ModelHealth:
    """Whether a kind has an active binding, and the state of that binding."""

    configured: bool
    state: BindingState | None = None
    # For the embedding model: the state of the Memory vector index.
    vector_index: VectorIndexStatus | None = None


# What one check is a check of: the database a stored binding is in, the slot,
# the saved version and the digest of the binding.
_CheckSubject = tuple[int | None, BindingSlot, str | None, str]


@dataclass
class _CheckQueue:
    """The checks of one slot of one kind, which run one at a time."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # How many checks have finished, and the subject and outcome of the latest.
    finished: int = 0
    subject: _CheckSubject | None = None
    outcome: ValidationOutcome | None = None


class ModelSettings:
    """The model bindings of one server, for whichever of its databases a call names."""

    def __init__(
        self,
        config: AppConfig,
        runtime_provider: RuntimeProvider,
        *,
        listed_native_schema: ListedNativeSchema | None = None,
    ) -> None:
        self._config = config
        self._runtime_provider = runtime_provider
        # Asks a provider's model listing what it declares about a model
        # answering in a JSON schema; ``None`` refreshes nothing.
        self._listed_native_schema = listed_native_schema
        # The state and latest check of each active binding that is not stored.
        self._unstored_checks: dict[ModelKind, tuple[BindingState, ValidationResult]] = {}
        self._check_queues: dict[tuple[ModelKind, BindingSlot], _CheckQueue] = {
            (kind, slot): _CheckQueue() for kind in MODEL_KINDS for slot in ("active", "candidate")
        }
        # The rebuilds of the vector index this server is running.
        self._reindex_tasks: set[asyncio.Task[Any]] = set()

    @property
    def writable(self) -> bool:
        """Whether bindings are stored and edited here, or managed by the deployment."""
        return self._config.server.llm_config_writable

    async def view(self, db: "Database", kind: ModelKind) -> KindView:
        """Return the active and the candidate binding of ``kind`` as Settings shows them."""
        active = await self._active(db, kind)
        candidate = await self._candidate(db, kind)
        waits = False
        if kind == "embedding" and candidate is not None and self._indexes(db) is not None:
            waits = _waits_for_reindex(candidate, await db.get_active_vector_index())
        return KindView(
            kind=kind,
            active=None if active is None else await self._binding_view(db, active),
            candidate=(
                None
                if candidate is None
                else await self._binding_view(db, candidate, waits_for_reindex=waits)
            ),
        )

    async def health(self, db: "Database") -> dict[ModelKind, ModelHealth]:
        """Report, for each kind, whether a binding is in use and its state."""
        report: dict[ModelKind, ModelHealth] = {}
        for kind in MODEL_KINDS:
            active = await self._active(db, kind)
            vector_index = (await self.vector_index(db)).status if kind == "embedding" else None
            report[kind] = (
                ModelHealth(configured=False, vector_index=vector_index)
                if active is None
                else ModelHealth(configured=True, state=active.state, vector_index=vector_index)
            )
        return report

    async def vector_index(self, db: "Database", *, count_vectors: bool = False) -> VectorIndexView:
        """Report the Memory vector index in use and whether it fits the embedding model.

        An installation with no index in use gets its first one here, when the
        Embedding Profile of the active binding is known and the stored vectors,
        if there are any, have its length. ``count_vectors`` also counts the
        vectors the index holds.
        """
        indexes = self._indexes(db)
        if indexes is None:
            return VectorIndexView(status="not_managed", serves_search=True)
        active = await self._active(db, "embedding")
        candidate = await self._candidate(db, "embedding")
        profile = embedding_profile(active)
        in_use = await db.get_active_vector_index()
        if in_use is None and profile is not None:
            in_use = await indexes.settle(profile)
        running = await db.vector_reindex_running()
        serves_search = (
            active is not None
            and in_use is not None
            and _fingerprint_in_use(active, in_use) == in_use["fingerprint"]
        )
        if running:
            status: VectorIndexStatus = "reindexing"
        elif active is None:
            status = "not_configured"
        elif in_use is None:
            status = "awaiting_validation" if profile is None else "reindex_required"
        else:
            status = "ready" if serves_search else "reindex_required"
        reindex_target: BindingSlot | None = None
        if not running:
            # Whether stored vectors fit a model that has not answered yet is
            # not known, so there is a rebuild to start for it whenever some
            # are stored.
            stored_vectors = (
                active is not None and in_use is None and profile is None and await indexes.holds_stored_vectors()
            )
            reindex_target = _reindex_slot(active, candidate, in_use, stored_vectors=stored_vectors)
        vector_count: int | None = None
        if count_vectors and in_use is not None:
            opened = await indexes.in_use()
            if opened is not None:
                vector_count = int(await asyncio.to_thread(opened.collection.count))
        return VectorIndexView(
            status=status,
            serves_search=serves_search,
            reindex_target=reindex_target,
            in_use=in_use,
            vector_count=vector_count,
            reindex=await db.get_latest_vector_reindex(),
            reindex_running=running,
        )

    async def save_candidate(self, db: "Database", kind: ModelKind, draft: CandidateDraft) -> KindView:
        """Store ``draft`` as the candidate of ``kind``.

        A draft without a key keeps a key already saved for this kind, the
        candidate's or else the active binding's. A saved key is bound to the
        origin it was saved for: a draft for another origin has to bring the
        key again. A draft that takes its credentials from the process
        environment is accepted only for its provider's own endpoint.
        """
        self._require_writable()
        await self._refuse_embedding_change_during_reindex(db, kind)
        binding = build_binding(
            kind=kind,
            provider=draft.provider,
            model_id=draft.model_id,
            endpoint=draft.endpoint,
            auth_strategy=draft.auth_strategy,
            options=draft.options,
            parameters=draft.parameters,
            token_limits=draft.token_limits,
            native_schema=draft.native_schema,
            config_source="settings",
        )
        _require_provider_endpoint_for_environment_credentials(binding)
        previous = await self._candidate(db, kind)
        new_key = (draft.api_key or "").strip()
        secret: dict[str, str] | None = None
        if binding.auth_strategy == "api_key":
            if new_key:
                secret_ref, secret = new_stored_secret(new_key, origin=binding.origin)
            else:
                secret_ref = await self._kept_secret_ref(db, binding)
            binding = replace(binding, secret_ref=secret_ref)
        if previous is not None:
            binding = _carry_over_check(binding, previous)
        await db.put_model_binding(kind, "candidate", binding_record(binding), secret=secret)
        return await self.view(db, kind)

    async def discard_candidate(self, db: "Database", kind: ModelKind) -> KindView:
        """Delete the candidate of ``kind``; the active binding is untouched."""
        self._require_writable()
        await self._refuse_embedding_change_during_reindex(db, kind)
        await db.delete_model_binding(kind, "candidate")
        return await self.view(db, kind)

    async def validate_candidate(self, db: "Database", kind: ModelKind) -> ValidationOutcome:
        """Check the candidate of ``kind`` and make it the active binding when it passes.

        A candidate that fails stays the candidate, with the outcome recorded
        on it. The outcome applies only to the candidate that was checked: one
        saved again while its check ran is neither promoted nor marked.

        A generation candidate is checked with the response schema support
        its provider's listing declares now (``_refresh_native_schema``).

        An embedding candidate that passes becomes active at once only when
        its vectors belong in the index in use, or when the installation has
        no index yet and can start one for it. Otherwise it stays the
        candidate, validated, until the index is rebuilt for it.
        """
        self._require_writable()
        await self._refuse_embedding_change_during_reindex(db, kind)

        async def read() -> ModelBinding:
            binding = await self._candidate(db, kind)
            if binding is None or binding.saved_at is None:
                raise ModelConfigError("candidate_not_found", f"No {kind} model is waiting to be validated.")
            return binding

        async def check(binding: ModelBinding) -> ValidationOutcome:
            binding = await self._refresh_native_schema(db, "candidate", binding)
            validation = await self._validation(db, binding)
            state = state_after_check(binding.state, passed=validation.passed)
            assert binding.saved_at is not None
            promote = validation.passed
            first_index: Mapping[str, Any] | None = None
            if promote and kind == "embedding":
                promote, first_index = await self._index_for_promotion(
                    db,
                    replace(binding, state=state, validation=validation),
                )
            if promote:
                applied = await db.promote_model_candidate(
                    kind,
                    updated_at=binding.saved_at,
                    state=state,
                    validation=validation_record(validation),
                    vector_index=first_index,
                )
            else:
                applied = await db.record_model_binding_validation(
                    kind,
                    "candidate",
                    updated_at=binding.saved_at,
                    state=state,
                    validation=validation_record(validation),
                )
            if not applied:
                raise ModelConfigError(
                    "candidate_changed",
                    f"The {kind} model settings were saved again while they were being validated.",
                )
            return ValidationOutcome(
                kind=kind,
                validation=validation,
                promoted=promote,
                state=state,
                reindex_required=validation.passed and not promote,
            )

        return await self._check_once(db, kind, "candidate", read=read, check=check)

    async def validate_active(self, db: "Database", kind: ModelKind) -> ValidationOutcome | None:
        """Check the active binding of ``kind`` again and record the outcome.

        The binding stays in use whatever the outcome. A generation binding is
        checked with the response schema support its provider's listing
        declares now (``_refresh_native_schema``). Returns ``None`` when the
        kind has no active binding.
        """

        async def read() -> ModelBinding | None:
            return await self._active(db, kind)

        async def check(binding: ModelBinding) -> ValidationOutcome:
            binding = await self._refresh_native_schema(db, "active", binding)
            validation = await self._validation(db, binding)
            state = state_after_check(binding.state, passed=validation.passed)
            if binding.saved_at is None:
                self._unstored_checks[kind] = (state, validation)
            else:
                await db.record_model_binding_validation(
                    kind,
                    "active",
                    updated_at=binding.saved_at,
                    state=state,
                    validation=validation_record(validation),
                )
            if kind == "embedding" and validation.passed:
                await self._settle_first_index(db, replace(binding, state=state, validation=validation))
            return ValidationOutcome(kind=kind, validation=validation, promoted=False, state=state)

        return await self._check_once(db, kind, "active", read=read, check=check)

    async def _check_once(
        self,
        db: "Database",
        kind: ModelKind,
        slot: BindingSlot,
        *,
        read: Callable[[], Awaitable[ModelBinding | None]],
        check: Callable[[ModelBinding], Awaitable[ValidationOutcome]],
    ) -> ValidationOutcome | None:
        """Run ``check`` on the binding ``read`` returns, one check of the slot at a time.

        A caller that waited while a check of the same binding finished takes
        that outcome. Any other caller checks the binding as it is once its
        turn comes, so the state a check leaves follows from the state the
        check before it left.
        """
        queue = self._check_queues[kind, slot]
        requested = await read()
        if requested is None:
            return None
        subject = _check_subject(db, slot, requested)
        finished = queue.finished
        async with queue.lock:
            if queue.finished != finished and queue.subject == subject:
                return queue.outcome
            binding = await read()
            if binding is None:
                return None
            outcome = await check(binding)
            queue.finished += 1
            queue.subject = _check_subject(db, slot, binding)
            queue.outcome = outcome
            return outcome

    async def _refresh_native_schema(self, db: "Database", slot: BindingSlot, binding: ModelBinding) -> ModelBinding:
        """Return ``binding`` with the response schema support its provider's listing declares now.

        Support is what the provider observes about the model, and a provider
        may add it or drop it, so every check of a stored generation binding
        whose listing declares it asks the listing again and stores the
        answer. It is not part of what a check vouches for: a request that
        asks for the schema natively is validated as before and falls back to
        the schema in the prompt when the provider rejects it. A listing that
        does not answer, does not hold the model or declares nothing leaves
        the stored support as it is.
        """
        listed = self._listed_native_schema
        if listed is None or binding.kind != "generation" or binding.saved_at is None:
            return binding
        listing = provider_descriptor(binding.provider).listing
        if listing is None or not listing.gives_native_schema:
            return binding
        try:
            declared = await listed(binding, await resolve_credential(db, self._config, binding))
            if declared is None or declared == binding.native_schema:
                return binding
            recorded = await db.record_model_binding_native_schema(
                binding.kind, slot, updated_at=binding.saved_at, native_schema=declared
            )
        except Exception as exc:
            logger.warning(
                "Could not read the response schema support of the %s generation model (%s)",
                slot,
                getattr(exc, "code", type(exc).__name__),
            )
            return binding
        if not recorded:
            return binding
        logger.info("The listing declares response schema support of the %s generation model: %s", slot, declared)
        return replace(binding, native_schema=declared)

    async def validate_unready_active_bindings(self, db: "Database") -> None:
        """Check every active binding whose latest check is missing or out of date.

        This is what a server does once it is up: settings carried over from an
        earlier version, bindings read from the environment and bindings last
        checked under another LiteLLM version have not passed a check as they
        stand. A ready generation binding is not checked again, but takes the
        response schema support its provider's listing declares now, as a
        check would. A failure of one kind does not stop the check of the other.
        """
        for kind in MODEL_KINDS:
            try:
                active = await self._active(db, kind)
                if active is None:
                    continue
                if active.state == "ready":
                    await self._refresh_native_schema(db, "active", active)
                    continue
                outcome = await self.validate_active(db, kind)
            except Exception as exc:
                logger.warning(
                    "Could not check the active %s model (%s)",
                    kind,
                    getattr(exc, "code", type(exc).__name__),
                )
                continue
            if outcome is not None:
                logger.info(
                    "Checked the active %s model: state=%s error_code=%s",
                    kind,
                    outcome.state,
                    outcome.validation.error_code,
                )

    async def reindex_running(self, db: "Database") -> bool:
        """Whether the vector index of ``db`` is being rebuilt, here or by another process."""
        return self._indexes(db) is not None and await db.vector_reindex_running()

    async def prepare_vector_index(self, db: "Database") -> None:
        """Bring the vector index of a starting server into order.

        A rebuild that a stopped process left unfinished is recorded as
        interrupted, what it or a replaced index left behind is deleted, and
        an installation with no index in use gets its first one when the
        active embedding binding has passed its check.
        """
        indexes = self._indexes(db)
        if indexes is None:
            return
        try:
            await indexes.clean_up()
            await self._settle_first_index(db, await self._active(db, "embedding"))
        except Exception:
            logger.exception("Could not prepare the Memory vector index")

    async def start_reindex(
        self,
        db: "Database",
        *,
        on_switch: Callable[[], None] | None = None,
    ) -> VectorIndexView:
        """Start a rebuild of the vector index beside the server and report it.

        The rebuild is for the embedding candidate that waits for it, or else
        for the active embedding binding when the index in use was built with
        another model, was adopted, or is missing. An active binding that has
        not passed a check as it stands is checked first, and the rebuild is
        refused when it fails. ``on_switch`` is called once the rebuilt index
        is in use.
        """
        indexes, target = await self._reindex_target(db)
        record = await self._admit_reindex(indexes, target)
        task = asyncio.create_task(self._run_reindex(indexes, record, target, on_switch=on_switch))
        self._reindex_tasks.add(task)
        task.add_done_callback(self._reindex_tasks.discard)
        return await self.vector_index(db, count_vectors=True)

    async def run_reindex(
        self,
        db: "Database",
        *,
        progress: "ReindexProgress | None" = None,
    ) -> Mapping[str, Any]:
        """Rebuild the vector index and return the record of the rebuild once it has ended."""
        indexes, target = await self._reindex_target(db)
        record = await self._admit_reindex(indexes, target)
        return await indexes.rebuild(record, target, progress=progress)

    async def shutdown(self) -> None:
        """Stop the rebuilds this server is running; each is recorded as interrupted."""
        tasks = list(self._reindex_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _admit_reindex(
        self,
        indexes: "MemoryVectorIndexes",
        target: "ReindexTarget",
    ) -> Mapping[str, Any]:
        try:
            return await indexes.start_rebuild(target)
        except VectorReindexRefused as exc:
            raise ModelConfigError(exc.code, exc.message) from exc

    async def _run_reindex(
        self,
        indexes: "MemoryVectorIndexes",
        record: Mapping[str, Any],
        target: "ReindexTarget",
        *,
        on_switch: Callable[[], None] | None,
    ) -> None:
        await indexes.rebuild(record, target, on_switch=on_switch)

    async def _reindex_target(self, db: "Database") -> tuple["MemoryVectorIndexes", "ReindexTarget"]:
        """Decide which embedding binding a rebuild started now is for."""
        from memforge.memory.reindex import ReindexTarget

        indexes = self._indexes(db)
        if indexes is None:
            raise ModelConfigError(
                "reindex_not_available",
                "The vector index of this deployment is not rebuilt from here.",
            )
        active = await self._active(db, "embedding")
        if active is not None and embedding_profile(active) is None:
            # A rebuild needs the vector length of the model, which a check observes.
            await self.validate_active(db, "embedding")
            active = await self._active(db, "embedding")
        candidate = await self._candidate(db, "embedding")
        in_use = await db.get_active_vector_index()
        slot = _reindex_slot(active, candidate, in_use)
        if slot != "candidate" and embedding_profile(active) is None:
            raise ModelConfigError(
                "embedding_not_validated",
                "The embedding model has to pass its check before the vector index is rebuilt.",
            )
        if slot is None:
            raise ModelConfigError(REINDEX_NOT_REQUIRED, REINDEX_NOT_REQUIRED_MESSAGE)
        binding = candidate if slot == "candidate" else active
        assert binding is not None
        profile = embedding_profile(binding)
        assert profile is not None and binding.validation is not None
        promotion: Mapping[str, Any] | None = None
        if slot == "candidate":
            promotion = {
                "updated_at": binding.saved_at,
                "state": binding.state,
                "validation": validation_record(binding.validation),
            }
        return indexes, ReindexTarget(
            profile=profile,
            invocation=await resolve_invocation(db, self._config, binding),
            binding_slot=slot,
            promotion=promotion,
        )

    async def _index_for_promotion(
        self,
        db: "Database",
        candidate: ModelBinding,
    ) -> tuple[bool, Mapping[str, Any] | None]:
        """Decide whether a validated embedding candidate may become active now.

        Returns whether it may, and the first index to record with it for an
        installation that has no index in use. Stored vectors are adopted for
        the candidate only when it names the model the active binding names:
        they were made by the model in use.
        """
        indexes = self._indexes(db)
        profile = embedding_profile(candidate)
        if indexes is None or profile is None:
            return True, None
        in_use = await db.get_active_vector_index()
        if in_use is not None:
            return profile.fingerprint == in_use["fingerprint"], None
        first_index = await indexes.first_index_record(profile)
        if first_index is None:
            return False, None
        active = await self._active(db, "embedding")
        if first_index["adopted"] and active is not None and embedding_identity(active) != profile.identity:
            return False, None
        return True, first_index

    async def _settle_first_index(self, db: "Database", active: ModelBinding | None) -> None:
        indexes = self._indexes(db)
        profile = embedding_profile(active)
        if indexes is not None and profile is not None:
            await indexes.settle(profile)

    async def _refuse_embedding_change_during_reindex(self, db: "Database", kind: ModelKind) -> None:
        """Keep the embedding candidate as it is while the index is rebuilt for it."""
        if kind != "embedding" or self._indexes(db) is None:
            return
        if await db.vector_reindex_running():
            raise ModelConfigError(
                REINDEX_IN_PROGRESS,
                "The vector index is being rebuilt. Change the embedding model once the rebuild has finished.",
            )

    def _indexes(self, db: "Database") -> "MemoryVectorIndexes | None":
        """The vector indexes this service manages for ``db``, if the runtime provider has it do so."""
        return memory_vector_indexes(self._runtime_provider, db, self._config)

    async def discovery_key(
        self,
        db: "Database",
        kind: ModelKind,
        *,
        origin: str,
    ) -> tuple[str | None, bool]:
        """Decide which saved key a model listing request to ``origin`` may carry.

        Returns the key and whether the operator has to enter one: a key saved
        for another origin is never sent to this one.
        """
        holders = await self._key_holders(db, kind)
        for holder in holders:
            if holder.origin == origin:
                try:
                    return await resolve_credential(db, self._config, holder), False
                except ModelConfigError:
                    continue
        return None, bool(holders)

    def _require_writable(self) -> None:
        if not self.writable:
            raise ModelConfigError(
                "model_settings_read_only",
                "Model settings are managed by the deployment environment.",
            )

    async def _active(self, db: "Database", kind: ModelKind) -> ModelBinding | None:
        """The active binding of ``kind``, with the check this server made of it."""
        binding = await resolve_active_binding(db, self._config, kind)
        if binding is None or binding.saved_at is not None:
            return binding
        recorded = self._unstored_checks.get(kind)
        if recorded is None:
            return binding
        state, validation = recorded
        return replace(
            binding,
            state=state_after_change(state, previous_digest=validation.digest, digest=binding.digest),
            validation=validation,
        )

    async def _candidate(self, db: "Database", kind: ModelKind) -> ModelBinding | None:
        if not self.writable:
            return None
        stored = await db.get_model_binding(kind, "candidate")
        return None if stored is None else binding_from_record(stored)

    async def _key_holders(self, db: "Database", kind: ModelKind) -> list[ModelBinding]:
        """The bindings of ``kind`` that refer to a key: the candidate first, then the active one."""
        bindings = (await self._candidate(db, kind), await self._active(db, kind))
        return [binding for binding in bindings if binding is not None and binding.secret_ref]

    async def _kept_secret_ref(self, db: "Database", binding: ModelBinding) -> str:
        holders = await self._key_holders(db, binding.kind)
        for holder in holders:
            if holder.origin == binding.origin and holder.secret_ref:
                return holder.secret_ref
        if not holders:
            raise ModelConfigError("api_key_required", "Enter an API key.")
        raise ModelConfigError(
            "api_key_origin_changed",
            f"The saved API key belongs to {holders[0].origin}. Enter the API key for {binding.origin}.",
        )

    async def _validation(self, db: "Database", binding: ModelBinding) -> ValidationResult:
        check = await self._check(db, binding)
        return ValidationResult(
            checked_at=datetime.now(timezone.utc).isoformat(),
            digest=binding.digest,
            facts=check.facts,
            error_code=check.error_code,
        )

    async def _check(self, db: "Database", binding: ModelBinding) -> CapabilityCheck:
        try:
            invocation = await resolve_invocation(db, self._config, binding)
        except ModelConfigError as exc:
            return CapabilityCheck(error_code=exc.code)
        return await check_binding(self._config, self._runtime_provider, binding, invocation)

    async def _binding_view(
        self,
        db: "Database",
        binding: ModelBinding,
        *,
        waits_for_reindex: bool = False,
    ) -> BindingView:
        return BindingView(
            binding=binding,
            key=await self._key_view(db, binding),
            profile=embedding_profile(binding),
            waits_for_reindex=waits_for_reindex,
        )

    async def _key_view(self, db: "Database", binding: ModelBinding) -> KeyView:
        source = key_source(binding)
        if source is None:
            return KeyView()
        if source == "saved":
            secret = await db.get_model_secret(stored_secret_id(binding.secret_ref) or "")
            if secret is None:
                return KeyView(source=source)
            return KeyView(source=source, is_set=True, last4=secret["last4"] or None)
        try:
            value = await resolve_credential(db, self._config, binding)
        except ModelConfigError:
            return KeyView(source=source)
        return KeyView(source=source, is_set=True, last4=key_last4(value or "") or None)


async def check_binding(
    config: AppConfig,
    runtime_provider: RuntimeProvider,
    binding: ModelBinding,
    invocation: ResolvedInvocation,
) -> CapabilityCheck:
    """Make the capability check a binding is validated with.

    ``invocation`` is ``binding`` compiled with its credential. An embedding
    model is checked through an Embedding Runtime; a generation model through
    the structured client the runtime builds for it, with the runtime's token
    limits and a request timeout no longer than the check's.
    """
    if binding.kind == "embedding":
        return await check_embedding(invocation)
    llm = config.llm
    client = runtime_provider.build_structured_llm_client(
        ResolvedModels(
            generation=invocation,
            embedding=None,
            request_timeout_s=min(llm.request_timeout_s, GENERATION_CHECK_TIMEOUT_S),
            **generation_token_limits(llm, binding.token_limits),
            input_budget_fraction=llm.input_budget_fraction,
        ),
        max_concurrent=llm.enrichment_max_concurrent,
    )
    if client is None:
        return CapabilityCheck(error_code="not_configured")
    return await check_generation(client)


def _check_subject(db: "Database", slot: BindingSlot, binding: ModelBinding) -> _CheckSubject:
    """Name what a check of ``binding`` is a check of.

    A stored binding belongs to its database. One read from the environment is
    the same binding whichever database a call names.
    """
    return (None if binding.saved_at is None else id(db), slot, binding.saved_at, binding.digest)


def _require_provider_endpoint_for_environment_credentials(binding: ModelBinding) -> None:
    """Refuse a draft that would send credentials from the process environment elsewhere.

    With the environment strategy the SDK attaches whatever credential the
    process environment holds for the route. Settings may pair that strategy
    only with the endpoint the provider descriptor names, or with none for a
    descriptor that leaves the endpoint to the route, so that saving a base URL
    never directs such a credential to another server.
    """
    if binding.auth_strategy != "environment":
        return
    descriptor = provider_descriptor(binding.provider)
    if binding.endpoint == descriptor.default_endpoint:
        return
    if descriptor.default_endpoint is None:
        message = (
            "With credentials from the environment the route decides the endpoint. "
            "Leave the base URL empty, or enter an API key for this endpoint."
        )
    else:
        message = (
            f"With credentials from the environment, {descriptor.label} is reached at "
            f"{descriptor.default_endpoint} only. Enter an API key to use another base URL."
        )
    raise ModelConfigError("environment_endpoint_not_allowed", message)


def _carry_over_check(binding: ModelBinding, previous: ModelBinding) -> ModelBinding:
    """Give a saved candidate the state and latest check of the one it replaces.

    A passed check stays on record, and a change to what it vouched for, the
    key included, leaves the candidate to be validated again. A failed check
    is kept only while the settings and the key it failed with are unchanged.
    """
    state = state_after_change(previous.state, previous_digest=previous.digest, digest=binding.digest)
    validation = previous.validation
    if validation is not None and not validation.passed and previous.digest != binding.digest:
        validation = None
    return replace(binding, state=state, validation=validation)


def _fingerprint_in_use(active: ModelBinding, in_use: Mapping[str, Any]) -> str:
    """The fingerprint the active embedding binding embeds with, as far as it is known.

    A binding that has not passed a check as it stands is taken to answer with
    the vector length of the index in use, so that only its settings decide
    whether it is the model the index was built with.
    """
    profile = embedding_profile(active) or embedding_identity(active).profile(int(in_use["dimension"]))
    return profile.fingerprint


def _waits_for_reindex(candidate: ModelBinding, in_use: Mapping[str, Any] | None) -> bool:
    """Whether a candidate has passed its check and only the vector index keeps it from use."""
    profile = embedding_profile(candidate)
    if profile is None or candidate.state != "ready":
        return False
    return in_use is None or profile.fingerprint != in_use["fingerprint"]


def _reindex_slot(
    active: ModelBinding | None,
    candidate: ModelBinding | None,
    in_use: Mapping[str, Any] | None,
    *,
    stored_vectors: bool = False,
) -> BindingSlot | None:
    """Which embedding binding a rebuild of the vector index would be for, if any.

    A candidate that waits for the index comes first. Otherwise the active
    binding, when the index in use is missing, was built with another model,
    or was adopted with its model assumed.

    An active binding that has not passed a check as it stands is judged by
    its settings, as ``_fingerprint_in_use`` reads them. With no index in use
    there is a rebuild for it only when ``stored_vectors`` says the
    installation holds vectors: without any, its first answer gives the
    installation an empty index that needs no rebuild.
    """
    if candidate is not None and _waits_for_reindex(candidate, in_use):
        return "candidate"
    if active is None:
        return None
    if in_use is None:
        return "active" if stored_vectors or embedding_profile(active) is not None else None
    if in_use["adopted"] or _fingerprint_in_use(active, in_use) != in_use["fingerprint"]:
        return "active"
    return None
