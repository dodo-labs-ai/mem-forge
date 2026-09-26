"""One execution path for every structured model call that carries work items.

A caller describes its work: stable item IDs, optional ordered shared context,
a renderer, a decoder and the client method to call. The runner owns capacity
fit, packing, bounded concurrency, exact coverage, one correction per request,
splitting and typed per-item failures. A multi-item request that fails on its
size, or whose output is still invalid after its correction, is split in half
until each item stands alone, so one item never fails the others. It knows no
business meaning: the caller decides what an ``ItemFailure`` means and merges
the per-chunk results of an item whose shared context did not fit one request.
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

type FailureCategory = Literal["capacity_exceeded", "deadline_exceeded", "provider_error", "invalid_response"]

OUTPUT_INVALID = "output_invalid"
# Contract: one bounded correction per request, with the same input and refs.
_MAX_CORRECTIONS = 1
_CORRECTION_BLOCK = (
    "\n\n<correction>\n"
    "The previous response was rejected: {error}\n"
    "Return exactly one result for every requested ID. "
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


class RequestTooLarge(Exception):
    """Raised by a renderer that cannot assemble a request for this slice of work."""


@dataclass(frozen=True)
class ItemFailure:
    """Why one item has no result.

    ``capacity_exceeded`` means the item alone exceeds the route's input
    capacity; ``invalid_response`` means the model's output for the item alone
    stayed invalid after its correction. ``error`` is the exception that ended
    the item, when there is one. For a chain item, ``part`` is the index of the
    first part it could not read.
    """

    category: FailureCategory
    error_code: str
    error: Exception | None = None
    part: int | None = None

    @property
    def unjudgeable(self) -> bool:
        """The item alone cannot be judged, so sending it again would fail again.

        Every other failure (a timeout or a provider error) is transient.
        """

        return self.category in _UNJUDGEABLE


@dataclass(frozen=True)
class ItemTask(Generic[Part, Result]):
    """Independent items, optionally sharing ordered, separable context parts."""

    item_ids: Sequence[ItemId]
    render: Callable[[tuple[ItemId, ...], tuple[Part, ...]], LlmRequest]
    decode: Callable[[BaseModel, tuple[ItemId, ...], tuple[Part, ...]], Iterable[tuple[ItemId, Result]]]
    call: StructuredCall
    context: Sequence[Part] = ()
    journal: RequestJournal | None = None


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
    the last part must finish every item still reading.
    """

    initial_states: Mapping[ItemId, State]
    parts: Sequence[Part]
    # Per item: how many leading parts it reads before it may finish.
    first_part_end: Mapping[ItemId, int]
    finished: Callable[[State], bool]
    render: Callable[[ChainStep[Part, State]], LlmRequest]
    decode: Callable[[BaseModel, ChainStep[Part, State]], Iterable[tuple[ItemId, State]]]
    call: StructuredCall
    journal: RequestJournal | None = None


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
    calls: int = 0  # model calls sent, corrections included
    corrections: int = 0
    splits: int = 0  # requests halved after a size failure or output still invalid
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
        """Send one packed request; halves run in this worker, so fan-out stays bounded."""

        if request is None:
            request = self.fit(lambda: task.render(item_ids, context))
        if request is None:
            outcome: dict[ItemId, Result] | ItemFailure = ItemFailure("capacity_exceeded", INPUT_CAPACITY_EXCEEDED)
        else:
            outcome = await self._send(
                request, item_ids, task.call, lambda response: task.decode(response, item_ids, context), task.journal,
            )
            if not isinstance(outcome, ItemFailure):
                return outcome
        if len(item_ids) > 1 and _splits_items(outcome):
            self._record_split(outcome, items=len(item_ids), parts=len(context))
            middle = len(item_ids) // 2
            results = await self._run_request(task, item_ids[:middle], context)
            results.update(await self._run_request(task, item_ids[middle:], context))
            return results
        self.stats.failed_requests += 1
        return dict.fromkeys(item_ids, outcome)

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
                lambda response, step=step: _chain_states(task, step, task.decode(response, step)), task.journal,
            )
            if not isinstance(outcome, ItemFailure):
                states.update(outcome)
                lane.position += len(step.parts)
                finished = {item_id for item_id, state in outcome.items() if task.finished(state)}
                outcomes.update({item_id: outcome[item_id] for item_id in finished})
                lane.item_ids = tuple(item_id for item_id in lane.item_ids if item_id not in finished)
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
        decode: Callable[[BaseModel], Iterable[tuple[ItemId, Result]]], journal: RequestJournal | None,
    ) -> dict[ItemId, Result] | ItemFailure:
        """Send one request, or reuse its journaled response, which was validated when recorded."""

        work_id = None
        if journal is not None:
            work_id, stored = await journal.stage(request, item_ids)
            if stored is not None:
                self.stats.reused += 1
                try:
                    return _covered(decode(stored), item_ids)
                except ValueError as error:
                    # The stored response no longer decodes under this task's rules.
                    return ItemFailure("invalid_response", OUTPUT_INVALID, error)
        outcome = await self._call_with_correction(request, item_ids, call, decode, work_id)
        if isinstance(outcome, ItemFailure):
            if journal is not None:
                await journal.record(work_id, outcome)
            return outcome
        response, results = outcome
        if journal is not None:
            await journal.record(work_id, response)
        return results

    async def _call_with_correction(
        self, request: LlmRequest, item_ids: tuple[ItemId, ...], call: StructuredCall,
        decode: Callable[[BaseModel], Iterable[tuple[ItemId, Result]]], work_id: str | None,
    ) -> tuple[BaseModel, dict[ItemId, Result]] | ItemFailure:
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
                    results = _covered(decode(response), item_ids)
            except ValueError as error:
                if capture is not None:
                    rejected_captures.append(capture)
                logger.warning(
                    "llm_batch_output_rejected items=%d attempt=%d error_class=%s",
                    len(item_ids), corrections + 1, type(error).__name__,
                )
                corrected = replace(request, prompt=request.prompt + _CORRECTION_BLOCK.format(error=error))
                if corrections == _MAX_CORRECTIONS or not self.fits(corrected, correction=True):
                    return ItemFailure("invalid_response", OUTPUT_INVALID, error)
                corrections += 1
                self.stats.corrections += 1
                prompt = corrected.prompt
                continue
            for rejected in rejected_captures:
                await rejected.recovered()
            return response, results

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


def _chain_states(
    task: ChainTask[Part, State], step: ChainStep[Part, State], pairs: Iterable[tuple[ItemId, State]],
) -> list[tuple[ItemId, State]]:
    """Hold a decoded step to the chain contract, so a violation gets the one correction."""

    states = _covered(pairs, step.item_ids)
    early = sorted(item_id for item_id, state in states.items() if task.finished(state) and item_id not in step.may_finish)
    if early:
        raise ValueError(f"items finished before reading their first part: {', '.join(early)}")
    if step.position + len(step.parts) == step.total:
        unfinished = sorted(item_id for item_id, state in states.items() if not task.finished(state))
        if unfinished:
            raise ValueError(f"items did not finish at the last part: {', '.join(unfinished)}")
    return list(states.items())


def _covered(pairs: Iterable[tuple[ItemId, Result]], item_ids: tuple[ItemId, ...]) -> dict[ItemId, Result]:
    """Require exactly one result for every requested item and nothing else."""

    expected = set(item_ids)
    results: dict[ItemId, Result] = {}
    for item_id, result in pairs:
        if item_id not in expected:
            raise ValueError("the response names an ID that was not requested")
        if item_id in results:
            raise ValueError("the response returns an ID more than once")
        results[item_id] = result
    missing = len(expected) - len(results)
    if missing:
        raise ValueError(f"the response omits {missing} of {len(expected)} requested IDs")
    return {item_id: results[item_id] for item_id in item_ids}


def _failure_from_error(error: StructuredLlmError) -> ItemFailure:
    if error.terminal_category == "deadline_exceeded":
        category: FailureCategory = "deadline_exceeded"
    elif error.error_code in _CAPACITY_ERROR_CODES:
        category = "capacity_exceeded"
    elif error.terminal_category == "provider_error":
        category = "provider_error"
    else:
        category = "invalid_response"
    return ItemFailure(category, error.error_code, error)


def _is_size_failure(failure: ItemFailure) -> bool:
    """A smaller request may succeed: time, input size or output length ran out."""

    return failure.category in {"deadline_exceeded", "capacity_exceeded"} or failure.error_code == OUTPUT_TRUNCATED


def _splits_items(failure: ItemFailure) -> bool:
    """Halving the items may help: the request was too large, or one item's output spoils the rest."""

    return _is_size_failure(failure) or failure.category == "invalid_response"
