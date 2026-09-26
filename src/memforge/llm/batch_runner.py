"""One execution path for every structured model call that carries work items.

A caller describes its work: stable item IDs, optional ordered shared context,
a renderer, a decoder and the client method to call. The runner owns capacity
fit, packing, bounded concurrency, exact coverage, per-row acceptance, one
correction per item and typed per-item failures.

The model returns one row per item, and the decoder validates each row on its
own. Valid rows are accepted at once and never sent again. The rejected items
are re-asked once, together, in one request that holds only them and names each
item's exact error; an item still rejected after that is an ``invalid_response``
failure. A row that names an ID the request did not supply makes the whole
response unreadable, because its IDs can no longer be trusted to match their
rows. A request is split in half only when the failing item cannot be named:
when it fails on its size, or when its output cannot be read into rows even after
one correction of the whole request. It knows no business meaning: the caller
decides what an ``ItemFailure`` means and merges the per-chunk results of an item
whose shared context did not fit one request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from memforge.llm.failure_trace import failure_trace_context, validation_trace
from memforge.llm.request_budget import RequestBudget
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    OUTPUT_TRUNCATED,
    PAYLOAD_TOO_LARGE,
    StructuredLlmError,
    StructuredLlmImage,
    structured_llm_max_concurrent,
)
from memforge.pipeline.bounded_work import collect_bounded

logger = logging.getLogger(__name__)

ItemId = str
Part = TypeVar("Part")
Result = TypeVar("Result")
State = TypeVar("State")
Found = TypeVar("Found")

type FailureCategory = Literal[
    "capacity_exceeded", "deadline_exceeded", "provider_error", "invalid_response", "request_error",
]

OUTPUT_INVALID = "output_invalid"
# Contract: one bounded correction, with the same input and refs.
_MAX_CORRECTIONS = 1
# The one correction of a response that cannot be read into rows.
_CORRECTION_BLOCK = (
    "\n\n<correction>\n"
    "The previous response was rejected: {error}\n"
    "Return exactly one result for every requested ID. "
    "Use only the IDs and refs supplied above.\n"
    "</correction>"
)
# The one re-ask of the rejected items, each listed with its own error.
_REASK_BLOCK = (
    "\n\n<correction>\n"
    "The previous response was rejected for the IDs requested here:\n"
    "{errors}\n"
    "Return exactly one corrected result for every requested ID. "
    "Use only the IDs and refs supplied above.\n"
    "</correction>"
)
_CAPACITY_ERROR_CODES = frozenset({INPUT_CAPACITY_EXCEEDED, PAYLOAD_TOO_LARGE})
_UNJUDGEABLE: frozenset[FailureCategory] = frozenset({"capacity_exceeded", "invalid_response"})


@dataclass(frozen=True)
class LlmRequest:
    prompt: str
    response_format: type[BaseModel]
    # The output the task asks for; the runner bounds it by the route's capacity.
    max_tokens: int
    images: tuple[StructuredLlmImage, ...] = ()


# A client task method, called as ``call(prompt, max_tokens=..., model=...)``
# plus ``images=...`` when the request carries images.
StructuredCall = Callable[..., Awaitable[BaseModel]]


class BudgetedClient(Protocol):
    def request_budget(self, model: str | None = None) -> RequestBudget: ...

    def request_fits(
        self, prompt: str, *, response_format: type[BaseModel], max_tokens: int,
        model: str | None = None, images: tuple[StructuredLlmImage, ...] = (),
        reserve_correction: bool = True,
    ) -> bool: ...


class RejectedRow(ValueError):
    """One item's row that fails the task's validation.

    A decoder yields it in place of the item's result. Its message names the item
    by the ID the model sees and states the exact error, for the item's re-ask;
    ``cause`` keeps the validation error it came from, for the failure trace.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class RequestTooLarge(Exception):
    """Raised by a renderer that cannot assemble a request for this slice of work."""


@dataclass(frozen=True)
class ItemFailure:
    """Why one item has no result.

    ``capacity_exceeded`` means the item alone exceeds the route's input
    capacity; ``invalid_response`` means the item's row stayed rejected after its
    re-ask, or a response that holds only this item could not be read into rows
    after its correction; the caller's stage records
    these two, and its Unit commits. The rest leave the Source Unit revision
    uncommitted: ``deadline_exceeded`` and ``provider_error`` are transient and
    the sync retries them at once, while ``request_error``, a request that failed
    without a response to validate, is left for the next sync
    (``failure_retryable`` in ``llm.structured`` is the one rule). ``error`` is the exception that ended
    the item; every failure that is not unjudgeable carries one, for the caller to raise. For a chain item, ``part`` is the index of the
    first part it could not read.
    """

    category: FailureCategory
    error_code: str
    error: Exception | None = None
    part: int | None = None

    @property
    def unjudgeable(self) -> bool:
        """The item alone cannot be judged, so sending it again would fail again.

        Only these two failures are recorded by the caller's stage; every other
        failure leaves the work to be retried.
        """

        return self.category in _UNJUDGEABLE


@dataclass(frozen=True)
class ItemTask(Generic[Part, Result]):
    """Independent items, optionally sharing ordered, separable context parts.

    ``decode`` yields one ``(item_id, result)`` per row, or ``(item_id,
    RejectedRow)`` for a row whose meaning is invalid, and raises ``ValueError``
    only when the response cannot be read into rows. A row whose ID the task
    does not know is still yielded, under that ID, so the runner can reject the
    response. ``label`` names an item as the model sees it.
    """

    item_ids: Sequence[ItemId]
    render: Callable[[tuple[ItemId, ...], tuple[Part, ...]], LlmRequest]
    decode: Callable[
        [BaseModel, tuple[ItemId, ...], tuple[Part, ...]], Iterable[tuple[ItemId, Result | RejectedRow]],
    ]
    call: StructuredCall
    context: Sequence[Part] = ()
    journal: RequestJournal | None = None
    label: Callable[[ItemId], str] = str


@dataclass(frozen=True)
class ChainStep(Generic[Part, State]):
    item_ids: tuple[ItemId, ...]
    parts: tuple[Part, ...]
    states: Mapping[ItemId, State]
    # Index of ``parts[0]`` in the task's full part order.
    position: int
    total: int
    # Items whose first part is fully read once this step is read; only these may finish here.
    may_finish: frozenset[ItemId]


@dataclass(frozen=True)
class ChainTask(Generic[Part, State]):
    """Items that read the parts in order, carrying compact state between steps.

    Each item reads at least its first part and at most every part. While any
    first part is unread, a step reads no further than the farthest unread
    first-part boundary of its lane, so the lane's first parts are packed
    together, by capacity alone, ahead of the rest. An item may finish only at a
    step that completes its own first part (``may_finish``); it leaves the chain
    at the first step whose decoded state is ``finished``. The step that reads
    the last part must finish every item still reading. ``decode`` and ``label``
    follow ``ItemTask``; a row that breaks these rules is rejected alone.
    """

    initial_states: Mapping[ItemId, State]
    parts: Sequence[Part]
    # Per item: how many leading parts it reads before it may finish.
    first_part_end: Mapping[ItemId, int]
    finished: Callable[[State], bool]
    render: Callable[[ChainStep[Part, State]], LlmRequest]
    decode: Callable[[BaseModel, ChainStep[Part, State]], Iterable[tuple[ItemId, State | RejectedRow]]]
    call: StructuredCall
    journal: RequestJournal | None = None
    label: Callable[[ItemId], str] = str


@dataclass(frozen=True)
class PlannedRequest(Generic[Part]):
    item_ids: tuple[ItemId, ...]
    context: tuple[Part, ...]
    request: LlmRequest


@dataclass(frozen=True)
class BatchPlan(Generic[Part]):
    """The requests ``run_items`` would send, and the items that alone exceed the route's capacity."""

    requests: tuple[PlannedRequest[Part], ...]
    unfit: tuple[ItemId, ...]


class RequestJournal(Protocol):
    """Optional per-request persistence that lets a retried run reuse completed requests."""

    async def stage(self, request: LlmRequest, item_ids: Sequence[ItemId]) -> tuple[str, BaseModel | None]:
        """Return the request's work ID and its stored response, if it already completed."""

    async def record(self, work_id: str, outcome: BaseModel | ItemFailure) -> None: ...


@dataclass
class BatchStats:
    calls: int = 0  # model calls sent, corrections and re-asks included
    corrections: int = 0  # whole requests resent because their output could not be read into rows
    reasks: int = 0  # requests that re-asked rejected rows
    splits: int = 0  # requests halved after a size failure or output that could not be read into rows
    reused: int = 0  # requests answered from the journal without a call
    prompt_chars: int = 0
    failed_requests: int = 0  # requests whose items ended as ItemFailure


@dataclass
class _Lane:
    item_ids: tuple[ItemId, ...]
    position: int
    part_limit: int


class LlmBatchRunner:
    """Run one task's model requests on one client route."""

    def __init__(self, client: BudgetedClient, *, model: str | None) -> None:
        self._client = client
        self._model = model
        self.stats = BatchStats()
        self._max_concurrent = structured_llm_max_concurrent(client)

    def fits(self, request: LlmRequest, *, correction: bool = False) -> bool:
        """Whether the request fits; a first attempt keeps room for its correction."""

        bounded = self._bounded(request)
        return self._client.request_fits(
            bounded.prompt, response_format=bounded.response_format, max_tokens=bounded.max_tokens,
            model=self._model, images=bounded.images, reserve_correction=not correction,
        )

    def fit(self, render: Callable[[], LlmRequest]) -> LlmRequest | None:
        """Render one request bounded by the route, or None when it cannot fit."""

        try:
            request = self._bounded(render())
        except RequestTooLarge:
            return None
        return request if self.fits(request) else None

    def plan_items(
        self, item_ids: Sequence[ItemId], render: Callable[[tuple[ItemId, ...], tuple[Part, ...]], LlmRequest],
        context: Sequence[Part] = (),
    ) -> BatchPlan[Part]:
        """Pack items as ``run_items`` would, without sending."""

        planned, unfit = self._pack_items(item_ids, render, context)
        return BatchPlan(tuple(planned), tuple(unfit))

    async def run_items(self, task: ItemTask[Part, Result]) -> dict[ItemId, tuple[Result, ...] | ItemFailure]:
        """Return every item's results, one per context chunk in context order, or its failure."""

        item_ids = tuple(task.item_ids)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("batch item IDs must be unique")
        planned, unfit = self._pack_items(item_ids, task.render, task.context)
        outcomes = await collect_bounded(
            planned,
            lambda entry: self._run_request(task, entry.item_ids, entry.context, entry.request),
            max_concurrent=self._max_concurrent,
        )
        failures = {item_id: ItemFailure("capacity_exceeded", INPUT_CAPACITY_EXCEEDED) for item_id in unfit}
        chunks: dict[ItemId, list[Result]] = {item_id: [] for item_id in item_ids}
        for outcome in outcomes:
            for item_id, result in outcome.items():
                if isinstance(result, ItemFailure):
                    failures.setdefault(item_id, result)
                else:
                    chunks[item_id].append(result)
        return {
            item_id: failures[item_id] if item_id in failures else tuple(chunks[item_id])
            for item_id in item_ids
        }

    async def run_chain(self, task: ChainTask[Part, State]) -> dict[ItemId, State | ItemFailure]:
        """Return every item's state at the step it finished, or its failure."""

        total = len(task.parts)
        outside = sorted(item_id for item_id in task.initial_states if not 0 <= task.first_part_end[item_id] <= total)
        if outside:
            raise ValueError(f"first parts end outside the {total} chain parts: {', '.join(outside)}")
        states = dict(task.initial_states)
        outcomes: dict[ItemId, State | ItemFailure] = {}
        await self._run_lane(task, _Lane(tuple(states), 0, total), states, outcomes)
        return {item_id: outcomes[item_id] for item_id in task.initial_states}

    async def run_one(
        self, request: LlmRequest, *, call: StructuredCall,
        decode: Callable[[BaseModel], Result] = lambda response: response,
    ) -> Result | ItemFailure:
        """Run one indivisible request; it is never split."""

        only = "request"
        task = ItemTask(
            item_ids=(only,),
            render=lambda _ids, _parts: request,
            decode=lambda response, _ids, _parts: ((only, decode(response)),),
            call=call,
        )
        outcome = (await self.run_items(task))[only]
        return outcome if isinstance(outcome, ItemFailure) else outcome[0]

    def _bounded(self, request: LlmRequest) -> LlmRequest:
        budget = self._client.request_budget(self._model)
        return replace(request, max_tokens=budget.output_reserve(request.max_tokens))

    def _pack_items(
        self, item_ids: Sequence[ItemId], render: Callable[[tuple[ItemId, ...], tuple[Part, ...]], LlmRequest],
        context: Sequence[Part],
    ) -> tuple[list[PlannedRequest[Part]], list[ItemId]]:
        """Pack consecutive items with the whole context; chunk the context for an item that needs it."""

        item_ids = tuple(item_ids)
        context = tuple(context)
        planned: list[PlannedRequest[Part]] = []
        unfit: list[ItemId] = []

        def attempt(ids: tuple[ItemId, ...], parts: tuple[Part, ...]) -> PlannedRequest[Part] | None:
            request = self.fit(lambda: render(ids, parts))
            return None if request is None else PlannedRequest(ids, parts, request)

        start = 0
        while start < len(item_ids):
            remaining = item_ids[start:]
            found = _longest(len(remaining), lambda count: attempt(remaining[:count], context))
            if found is not None:
                planned.append(found)
                start += len(found.item_ids)
                continue
            item = remaining[:1]
            chunks = self._context_chunks(item, context, attempt)
            if chunks is None:
                unfit.append(item[0])
            else:
                planned.extend(chunks)
            start += 1
        return planned, unfit

    @staticmethod
    def _context_chunks(
        item: tuple[ItemId], context: tuple[Part, ...],
        attempt: Callable[[tuple[ItemId, ...], tuple[Part, ...]], PlannedRequest[Part] | None],
    ) -> list[PlannedRequest[Part]] | None:
        """Read the context in consecutive chunks for one item; None if one part alone does not fit."""

        if len(context) <= 1:
            return None
        chunks = []
        position = 0
        while position < len(context):
            remaining = context[position:]
            found = _longest(len(remaining), lambda count: attempt(item, remaining[:count]))
            if found is None:
                return None
            chunks.append(found)
            position += len(found.context)
        return chunks

    async def _run_request(
        self, task: ItemTask[Part, Result], item_ids: tuple[ItemId, ...], context: tuple[Part, ...],
        request: LlmRequest | None = None,
    ) -> dict[ItemId, Result | ItemFailure]:
        """Send one packed request, accept its valid rows and re-ask the rest once; halves run in this worker."""

        if request is None:
            request = self.fit(lambda: task.render(item_ids, context))
        if request is None:
            outcome: dict[ItemId, Result | RejectedRow] | ItemFailure = ItemFailure(
                "capacity_exceeded", INPUT_CAPACITY_EXCEEDED,
            )
        else:
            outcome = await self._send(
                request, item_ids, task.call, lambda response: task.decode(response, item_ids, context),
                task.journal, task.label,
            )
            if not isinstance(outcome, ItemFailure):
                return await self._with_reask(
                    outcome, render=lambda ids: task.render(ids, context),
                    decode=lambda response, ids: task.decode(response, ids, context),
                    call=task.call, journal=task.journal, label=task.label,
                )
        if len(item_ids) > 1 and _splits_items(outcome):
            self._record_split(outcome, items=len(item_ids), parts=len(context))
            middle = len(item_ids) // 2
            results = await self._run_request(task, item_ids[:middle], context)
            results.update(await self._run_request(task, item_ids[middle:], context))
            return results
        self.stats.failed_requests += 1
        return dict.fromkeys(item_ids, outcome)

    async def _with_reask(
        self, rows: dict[ItemId, Result | RejectedRow], *,
        render: Callable[[tuple[ItemId, ...]], LlmRequest],
        decode: Callable[[BaseModel, tuple[ItemId, ...]], Iterable[tuple[ItemId, Result | RejectedRow]]],
        call: StructuredCall, journal: RequestJournal | None, label: Callable[[ItemId], str],
    ) -> dict[ItemId, Result | ItemFailure]:
        """Keep the accepted rows and re-ask the rejected items once, together, naming each item's error.

        The re-ask renders only the rejected items with the request's own context,
        packed by capacity; an item whose re-ask cannot fit, or whose row is still
        rejected, is an invalid response.
        """

        rejected = {item_id: row for item_id, row in rows.items() if isinstance(row, RejectedRow)}
        results: dict[ItemId, Result | ItemFailure] = {
            item_id: row for item_id, row in rows.items() if not isinstance(row, RejectedRow)
        }
        if not rejected:
            return results
        logger.warning(
            "llm_batch_rows_rejected items=%d rejected=%d error_classes=%s", len(rows), len(rejected),
            ",".join(sorted({type(row.__cause__ or row).__name__ for row in rejected.values()})),
        )

        def corrected(ids: tuple[ItemId, ...]) -> LlmRequest:
            request = render(ids)
            errors = "\n".join(f"- {rejected[item_id]}" for item_id in ids)
            return replace(request, prompt=request.prompt + _REASK_BLOCK.format(errors=errors))

        ids = tuple(rejected)
        start = 0
        while start < len(ids):
            remaining = ids[start:]
            found = _longest(len(remaining), lambda count: self._fit_reask(remaining[:count], corrected))
            if found is None:
                results[remaining[0]] = _invalid(rejected[remaining[0]])
                start += 1
                continue
            sub, request = found
            results.update(await self._send_reask(sub, request, rejected, corrected, decode, call, journal, label))
            start += len(sub)
        return results

    def _fit_reask(
        self, ids: tuple[ItemId, ...], corrected: Callable[[tuple[ItemId, ...]], LlmRequest],
    ) -> tuple[tuple[ItemId, ...], LlmRequest] | None:
        """A re-ask is the correction itself, so it keeps no room for another one."""

        try:
            request = self._bounded(corrected(ids))
        except RequestTooLarge:
            return None
        return (ids, request) if self.fits(request, correction=True) else None

    async def _send_reask(
        self, ids: tuple[ItemId, ...], request: LlmRequest, rejected: Mapping[ItemId, RejectedRow],
        corrected: Callable[[tuple[ItemId, ...]], LlmRequest],
        decode: Callable[[BaseModel, tuple[ItemId, ...]], Iterable[tuple[ItemId, Result | RejectedRow]]],
        call: StructuredCall, journal: RequestJournal | None, label: Callable[[ItemId], str],
    ) -> dict[ItemId, Result | ItemFailure]:
        """Send one re-ask; a re-ask that fails on its size or cannot be read into rows is halved."""

        self.stats.reasks += 1
        outcome = await self._send(
            request, ids, call, lambda response: decode(response, ids), journal, label, correct=False,
        )
        if not isinstance(outcome, ItemFailure):
            return {
                item_id: _invalid(row) if isinstance(row, RejectedRow) else row for item_id, row in outcome.items()
            }
        if len(ids) > 1 and _splits_items(outcome):
            self._record_split(outcome, items=len(ids), parts=0)
            middle = len(ids) // 2
            results: dict[ItemId, Result | ItemFailure] = {}
            for half in (ids[:middle], ids[middle:]):
                found = self._fit_reask(half, corrected)
                if found is None:
                    results.update({item_id: _invalid(rejected[item_id]) for item_id in half})
                    continue
                results.update(
                    await self._send_reask(half, found[1], rejected, corrected, decode, call, journal, label),
                )
            return results
        self.stats.failed_requests += 1
        return dict.fromkeys(ids, outcome)

    def _next_step(
        self, task: ChainTask[Part, State], lane: _Lane, states: Mapping[ItemId, State],
    ) -> tuple[ChainStep[Part, State], LlmRequest] | None:
        """Return the lane's next step with the longest fitting part prefix, or None."""

        lane_states = {item_id: states[item_id] for item_id in lane.item_ids}
        total = len(task.parts)
        # The lane's first parts are read before the rest; once they are, any length that fits.
        boundary = max(
            (end for item_id in lane.item_ids if (end := task.first_part_end[item_id]) > lane.position),
            default=total,
        )

        def attempt(count: int) -> tuple[ChainStep[Part, State], LlmRequest] | None:
            read = lane.position + count
            step = ChainStep(
                lane.item_ids, tuple(task.parts[lane.position:read]), lane_states, lane.position, total,
                frozenset(item_id for item_id in lane.item_ids if read >= task.first_part_end[item_id]),
            )
            request = self.fit(lambda: task.render(step))
            return None if request is None else (step, request)

        return _longest(min(lane.part_limit, boundary - lane.position), attempt)

    async def _run_lane(
        self, task: ChainTask[Part, State], lane: _Lane, states: dict[ItemId, State],
        outcomes: dict[ItemId, State | ItemFailure],
    ) -> None:
        """Read the remaining parts in order; split lanes run in this worker one after the other."""

        while lane.item_ids and lane.position < len(task.parts):
            found = self._next_step(task, lane, states)
            if found is None:
                if len(lane.item_ids) == 1:
                    outcomes[lane.item_ids[0]] = ItemFailure(
                        "capacity_exceeded", INPUT_CAPACITY_EXCEEDED, part=lane.position,
                    )
                    return
                for half in _halve_lane(lane):
                    await self._run_lane(task, half, states, outcomes)
                return
            step, request = found
            outcome = await self._send(
                request, step.item_ids, task.call,
                lambda response, step=step: _chain_rows(task, step, task.decode(response, step)),
                task.journal, task.label,
            )
            if not isinstance(outcome, ItemFailure):
                # A re-asked item reads the same parts from the same position with its unchanged state.
                read = await self._with_reask(
                    outcome, render=lambda ids, step=step: task.render(_substep(step, ids)),
                    decode=lambda response, ids, step=step: _chain_rows(
                        task, _substep(step, ids), task.decode(response, _substep(step, ids)),
                    ),
                    call=task.call, journal=task.journal, label=task.label,
                )
                ended = set()
                for item_id, row in read.items():
                    if isinstance(row, ItemFailure):
                        outcomes[item_id] = replace(row, part=step.position)
                        ended.add(item_id)
                        continue
                    states[item_id] = row
                    if task.finished(row):
                        outcomes[item_id] = row
                        ended.add(item_id)
                lane.position += len(step.parts)
                lane.item_ids = tuple(item_id for item_id in lane.item_ids if item_id not in ended)
                continue
            if _splits_items(outcome) and len(step.item_ids) > 1:
                self._record_split(outcome, items=len(step.item_ids), parts=len(step.parts))
                for half in _halve_lane(lane):
                    await self._run_lane(task, half, states, outcomes)
                return
            if _is_size_failure(outcome) and len(step.parts) > 1:
                self._record_split(outcome, items=1, parts=len(step.parts))
                lane.part_limit = len(step.parts) // 2
                continue
            self.stats.failed_requests += 1
            outcomes.update(dict.fromkeys(lane.item_ids, replace(outcome, part=step.position)))
            return
        # Only a chain without parts ends with items still reading.
        outcomes.update({item_id: states[item_id] for item_id in lane.item_ids})

    async def _send(
        self, request: LlmRequest, item_ids: tuple[ItemId, ...], call: StructuredCall,
        decode: Callable[[BaseModel], Iterable[tuple[ItemId, Result | RejectedRow]]],
        journal: RequestJournal | None, label: Callable[[ItemId], str], *, correct: bool = True,
    ) -> dict[ItemId, Result | RejectedRow] | ItemFailure:
        """Send one request, or reuse its journaled response, and read it into one row per item.

        A response that reads into rows is journaled even when some rows are
        rejected, so a retried run reuses its accepted rows without a call.
        """

        work_id = None
        if journal is not None:
            work_id, stored = await journal.stage(request, item_ids)
            if stored is not None:
                self.stats.reused += 1
                try:
                    return _rows(decode(stored), item_ids, label)
                except ValueError as error:
                    # The stored response no longer reads into rows under this task's rules.
                    return ItemFailure("invalid_response", OUTPUT_INVALID, error)
        outcome = await self._call_with_correction(request, item_ids, call, decode, work_id, label, correct)
        if isinstance(outcome, ItemFailure):
            if journal is not None:
                await journal.record(work_id, outcome)
            return outcome
        response, rows = outcome
        if journal is not None:
            await journal.record(work_id, response)
        return rows

    async def _call_with_correction(
        self, request: LlmRequest, item_ids: tuple[ItemId, ...], call: StructuredCall,
        decode: Callable[[BaseModel], Iterable[tuple[ItemId, Result | RejectedRow]]], work_id: str | None,
        label: Callable[[ItemId], str], correct: bool,
    ) -> tuple[BaseModel, dict[ItemId, Result | RejectedRow]] | ItemFailure:
        """Call once; a response that cannot be read into rows gets one correction of the whole request."""

        lineage = {} if work_id is None else {"work_id": work_id}
        image_kwargs = {"images": request.images} if request.images else {}
        prompt = request.prompt
        corrections = 0
        rejected_captures = []
        while True:
            self.stats.calls += 1
            self.stats.prompt_chars += len(prompt)
            try:
                with failure_trace_context(**lineage, correction_attempt=corrections + 1):
                    response = await call(prompt, max_tokens=request.max_tokens, model=self._model, **image_kwargs)
            except StructuredLlmError as error:
                return _failure_from_error(error)
            capture = None
            try:
                async with validation_trace(response, **lineage) as capture:
                    rows = _rows(decode(response), item_ids, label)
                    rejected_rows = [row for row in rows.values() if isinstance(row, RejectedRow)]
                    if capture is not None and rejected_rows:
                        # The response is kept, so its rejected rows are traced here; each is re-asked.
                        for row in rejected_rows:
                            capture.failed(row.__cause__ or row, stage="business_validation")
                        await capture.persist()
            except ValueError as error:
                if capture is not None:
                    rejected_captures.append(capture)
                logger.warning(
                    "llm_batch_output_rejected items=%d attempt=%d error_class=%s",
                    len(item_ids), corrections + 1, type(error).__name__,
                )
                corrected = replace(request, prompt=request.prompt + _CORRECTION_BLOCK.format(error=error))
                if not correct or corrections == _MAX_CORRECTIONS or not self.fits(corrected, correction=True):
                    return ItemFailure("invalid_response", OUTPUT_INVALID, error)
                corrections += 1
                self.stats.corrections += 1
                prompt = corrected.prompt
                continue
            for rejected in rejected_captures:
                await rejected.recovered()
            return response, rows

    def _record_split(self, failure: ItemFailure, *, items: int, parts: int) -> None:
        self.stats.splits += 1
        logger.info("llm_batch_split cause=%s items=%d parts=%d", failure.error_code, items, parts)


def _longest(limit: int, attempt: Callable[[int], Found | None]) -> Found | None:
    """Return ``attempt(n)`` for the largest ``n`` in ``1..limit`` that succeeds, trying ``limit`` first."""

    if limit < 1:
        return None
    if (whole := attempt(limit)) is not None:
        return whole
    low, high, best = 0, limit - 1, None
    while low < high:
        middle = (low + high + 1) // 2
        if (found := attempt(middle)) is None:
            high = middle - 1
        else:
            low, best = middle, found
    return best


def _halve_lane(lane: _Lane) -> list[_Lane]:
    middle = len(lane.item_ids) // 2
    return [
        _Lane(lane.item_ids[:middle], lane.position, lane.part_limit),
        _Lane(lane.item_ids[middle:], lane.position, lane.part_limit),
    ]


def _substep(step: ChainStep[Part, State], item_ids: tuple[ItemId, ...]) -> ChainStep[Part, State]:
    """The same step for some of its items: same parts and position, their own states."""

    return ChainStep(
        item_ids, step.parts, {item_id: step.states[item_id] for item_id in item_ids}, step.position, step.total,
        step.may_finish & frozenset(item_ids),
    )


def _chain_rows(
    task: ChainTask[Part, State], step: ChainStep[Part, State], pairs: Iterable[tuple[ItemId, State | RejectedRow]],
) -> Iterable[tuple[ItemId, State | RejectedRow]]:
    """Hold each decoded row to the chain contract; a row that breaks it is rejected alone."""

    last = step.position + len(step.parts) == step.total
    for item_id, row in pairs:
        if isinstance(row, RejectedRow) or item_id not in step.states:
            yield item_id, row
        elif task.finished(row) and item_id not in step.may_finish:
            yield item_id, RejectedRow(f"{task.label(item_id)} finished before reading its first part")
        elif last and not task.finished(row):
            yield item_id, RejectedRow(f"{task.label(item_id)} did not finish at the last part")
        else:
            yield item_id, row


def _rows(
    pairs: Iterable[tuple[ItemId, Result | RejectedRow]], item_ids: tuple[ItemId, ...], label: Callable[[ItemId], str],
) -> dict[ItemId, Result | RejectedRow]:
    """One row per requested item: its decoded result, or why it has none.

    A row that names an ID this request did not supply means the response's IDs
    cannot be trusted to match their rows (an answer may sit under its neighbour's
    ID), so the whole response is unreadable and ``ValueError`` is raised. An
    item with no row or with several rows is rejected by itself.
    """

    requested = set(item_ids)
    found: dict[ItemId, list[Result | RejectedRow]] = {}
    unknown: list[str] = []
    for item_id, row in pairs:
        if item_id in requested:
            found.setdefault(item_id, []).append(row)
        else:
            unknown.append(_named(label, item_id))
    if unknown:
        raise ValueError(
            "the response names IDs this request did not supply, so its rows cannot be matched to the "
            f"requested IDs: {', '.join(sorted(set(unknown)))}"
        )
    rows: dict[ItemId, Result | RejectedRow] = {}
    for item_id in item_ids:
        answers = found.get(item_id, [])
        if not answers:
            rows[item_id] = RejectedRow(f"no result was returned for {label(item_id)}")
        elif len(answers) > 1:
            rows[item_id] = RejectedRow(f"{label(item_id)} was returned more than once")
        else:
            rows[item_id] = answers[0]
    return rows


def _named(label: Callable[[ItemId], str], item_id: ItemId) -> str:
    """The item as the model sees it; an ID the task does not know is shown as returned."""

    try:
        return label(item_id)
    except KeyError:
        return item_id


def _invalid(row: RejectedRow) -> ItemFailure:
    """An item whose row is still rejected after its one re-ask."""

    return ItemFailure("invalid_response", OUTPUT_INVALID, row)


def _failure_from_error(error: StructuredLlmError) -> ItemFailure:
    if error.terminal_category == "deadline_exceeded":
        category: FailureCategory = "deadline_exceeded"
    elif error.error_code in _CAPACITY_ERROR_CODES:
        category = "capacity_exceeded"
    elif error.terminal_category in {"provider_error", "invalid_response"}:
        category = error.terminal_category
    else:
        category = "request_error"
    return ItemFailure(category, error.error_code, error)


def _is_size_failure(failure: ItemFailure) -> bool:
    """A smaller request may succeed: time, input size or output length ran out."""

    return failure.category in {"deadline_exceeded", "capacity_exceeded"} or failure.error_code == OUTPUT_TRUNCATED


def _splits_items(failure: ItemFailure) -> bool:
    """Halving the items may help: the request was too large, or its output could not be read into rows.

    A request-level ``invalid_response`` means the response could not be read
    into per-item rows, so the failing item cannot be named.
    """

    return _is_size_failure(failure) or failure.category == "invalid_response"
