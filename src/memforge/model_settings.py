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
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

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
    provider_descriptor,
)
from memforge.runtime import ResolvedModels, RuntimeProvider

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "BindingView",
    "CandidateDraft",
    "KeyView",
    "KindView",
    "ModelHealth",
    "ModelSettings",
    "ValidationOutcome",
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


@dataclass(frozen=True)
class ModelHealth:
    """Whether a kind has an active binding, and the state of that binding."""

    configured: bool
    state: BindingState | None = None


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

    def __init__(self, config: AppConfig, runtime_provider: RuntimeProvider) -> None:
        self._config = config
        self._runtime_provider = runtime_provider
        # The state and latest check of each active binding that is not stored.
        self._unstored_checks: dict[ModelKind, tuple[BindingState, ValidationResult]] = {}
        self._check_queues: dict[tuple[ModelKind, BindingSlot], _CheckQueue] = {
            (kind, slot): _CheckQueue() for kind in MODEL_KINDS for slot in ("active", "candidate")
        }

    @property
    def writable(self) -> bool:
        """Whether bindings are stored and edited here, or managed by the deployment."""
        return self._config.server.llm_config_writable

    async def view(self, db: "Database", kind: ModelKind) -> KindView:
        """Return the active and the candidate binding of ``kind`` as Settings shows them."""
        active = await self._active(db, kind)
        candidate = await self._candidate(db, kind)
        return KindView(
            kind=kind,
            active=None if active is None else await self._binding_view(db, active),
            candidate=None if candidate is None else await self._binding_view(db, candidate),
        )

    async def health(self, db: "Database") -> dict[ModelKind, ModelHealth]:
        """Report, for each kind, whether a binding is in use and its state."""
        report: dict[ModelKind, ModelHealth] = {}
        for kind in MODEL_KINDS:
            active = await self._active(db, kind)
            report[kind] = (
                ModelHealth(configured=False)
                if active is None
                else ModelHealth(configured=True, state=active.state)
            )
        return report

    async def save_candidate(self, db: "Database", kind: ModelKind, draft: CandidateDraft) -> KindView:
        """Store ``draft`` as the candidate of ``kind``.

        A draft without a key keeps a key already saved for this kind, the
        candidate's or else the active binding's. A saved key is bound to the
        origin it was saved for: a draft for another origin has to bring the
        key again. A draft that takes its credentials from the process
        environment is accepted only for its provider's own endpoint.
        """
        self._require_writable()
        binding = build_binding(
            kind=kind,
            provider=draft.provider,
            model_id=draft.model_id,
            endpoint=draft.endpoint,
            auth_strategy=draft.auth_strategy,
            options=draft.options,
            parameters=draft.parameters,
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
            binding = _carry_over_check(binding, previous, key_replaced=bool(new_key))
        await db.put_model_binding(kind, "candidate", binding_record(binding), secret=secret)
        return await self.view(db, kind)

    async def discard_candidate(self, db: "Database", kind: ModelKind) -> KindView:
        """Delete the candidate of ``kind``; the active binding is untouched."""
        self._require_writable()
        await db.delete_model_binding(kind, "candidate")
        return await self.view(db, kind)

    async def validate_candidate(self, db: "Database", kind: ModelKind) -> ValidationOutcome:
        """Check the candidate of ``kind`` and make it the active binding when it passes.

        A candidate that fails stays the candidate, with the outcome recorded
        on it. The outcome applies only to the candidate that was checked: one
        saved again while its check ran is neither promoted nor marked.
        """
        self._require_writable()

        async def read() -> ModelBinding:
            binding = await self._candidate(db, kind)
            if binding is None or binding.saved_at is None:
                raise ModelConfigError("candidate_not_found", f"No {kind} model is waiting to be validated.")
            return binding

        async def check(binding: ModelBinding) -> ValidationOutcome:
            validation = await self._validation(db, binding)
            state = state_after_check(binding.state, passed=validation.passed)
            assert binding.saved_at is not None
            if validation.passed:
                applied = await db.promote_model_candidate(
                    kind,
                    updated_at=binding.saved_at,
                    state=state,
                    validation=validation_record(validation),
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
                promoted=validation.passed,
                state=state,
            )

        return await self._check_once(db, kind, "candidate", read=read, check=check)

    async def validate_active(self, db: "Database", kind: ModelKind) -> ValidationOutcome | None:
        """Check the active binding of ``kind`` again and record the outcome.

        The binding stays in use whatever the outcome. Returns ``None`` when the
        kind has no active binding.
        """

        async def read() -> ModelBinding | None:
            return await self._active(db, kind)

        async def check(binding: ModelBinding) -> ValidationOutcome:
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

    async def validate_unready_active_bindings(self, db: "Database") -> None:
        """Check every active binding whose latest check is missing or out of date.

        This is what a server does once it is up: settings carried over from an
        earlier version, bindings read from the environment and bindings last
        checked under another LiteLLM version have not passed a check as they
        stand. A failure of one kind does not stop the check of the other.
        """
        for kind in MODEL_KINDS:
            try:
                active = await self._active(db, kind)
                if active is None or active.state == "ready":
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
        if binding.kind == "embedding":
            return await check_embedding(invocation)
        llm = self._config.llm
        client = self._runtime_provider.build_structured_llm_client(
            ResolvedModels(
                generation=invocation,
                embedding=None,
                request_timeout_s=min(llm.request_timeout_s, GENERATION_CHECK_TIMEOUT_S),
                max_input_tokens=llm.max_input_tokens,
                context_window_tokens=llm.context_window_tokens,
                max_output_tokens=llm.max_output_tokens,
                input_budget_fraction=llm.input_budget_fraction,
            ),
            max_concurrent=llm.enrichment_max_concurrent,
        )
        if client is None:
            return CapabilityCheck(error_code="not_configured")
        return await check_generation(client)

    async def _binding_view(self, db: "Database", binding: ModelBinding) -> BindingView:
        return BindingView(binding=binding, key=await self._key_view(db, binding))

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


def _carry_over_check(
    binding: ModelBinding,
    previous: ModelBinding,
    *,
    key_replaced: bool,
) -> ModelBinding:
    """Give a saved candidate the state and latest check of the one it replaces.

    A passed check stays on record, and a change to what it vouched for leaves
    the candidate to be validated again. A failed check is kept only while the
    settings and the key it failed with are unchanged.
    """
    state = state_after_change(previous.state, previous_digest=previous.digest, digest=binding.digest)
    validation = previous.validation
    if validation is not None and not validation.passed:
        if key_replaced or previous.digest != binding.digest:
            validation = None
    return replace(binding, state=state, validation=validation)
