from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import re

import pytest
from pydantic import BaseModel

from memforge.derivation_work import DerivationWorkJournal
from memforge.llm.batch_runner import (
    OUTPUT_INVALID,
    ChainStep,
    ChainTask,
    ItemFailure,
    ItemTask,
    LlmBatchRunner,
    LlmRequest,
    RejectedRow,
    RequestTooLarge,
)
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    OUTPUT_TRUNCATED,
    PAYLOAD_TOO_LARGE,
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
)
from tests.llm_fixture import FIXTURE_MODEL, FixtureBudgetClient

# Above every fixture output limit, so each request shows the runner's bound.
REQUESTED_OUTPUT = 500


class Row(BaseModel):
    id: str
    read: list[str]


class Rows(BaseModel):
    rows: list[Row]


def ids(count: int) -> tuple[str, ...]:
    return tuple(f"i{index:02d}" for index in range(count))


def parts(count: int) -> tuple[str, ...]:
    return tuple(f"p{index}" for index in range(count))


def prompt_ids(prompt: str) -> list[str]:
    return re.findall(r"\bi\d+\b", prompt)


def prompt_parts(prompt: str) -> list[str]:
    return re.findall(r"\bp\d+\b", prompt)


def answer(prompt: str) -> Rows:
    """One row per requested item; a correction's error lines are not requests."""
    request = prompt.split("<correction>")[0]
    return Rows(rows=[Row(id=item_id, read=prompt_parts(request)) for item_id in prompt_ids(request)])


def failing(error: StructuredLlmError, *, when: Callable[[str], bool]) -> Callable[[str], Rows]:
    def respond(prompt: str) -> Rows:
        if when(prompt):
            raise error
        return answer(prompt)

    return respond


def render(item_ids, context) -> LlmRequest:
    return LlmRequest(" ".join(["ask", *item_ids, *context]), Rows, REQUESTED_OUTPUT)


def decode(response, item_ids, context):
    return [(row.id, tuple(row.read)) for row in response.rows]


def item_task(client, item_ids, context=(), **overrides) -> ItemTask:
    fields = {"render": render, "decode": decode, "call": client.call, "context": context, **overrides}
    return ItemTask(item_ids=item_ids, **fields)


def render_step(step: ChainStep) -> LlmRequest:
    return render(step.item_ids, step.parts)


def carry(response, step: ChainStep):
    return [(row.id, step.states[row.id] + tuple(row.read)) for row in response.rows]


def chain_task(client, item_ids, chain_parts, **overrides) -> ChainTask:
    """Every item reads every part; its state is the parts it read."""
    fields = {
        "render": render_step, "decode": carry, "call": client.call,
        "first_part_end": dict.fromkeys(item_ids, len(chain_parts)),
        "finished": lambda state: len(state) == len(chain_parts),
        **overrides,
    }
    return ChainTask(initial_states={item_id: () for item_id in item_ids}, parts=chain_parts, **fields)


def one_part_per_step(step: ChainStep) -> LlmRequest:
    if len(step.parts) > 1:
        raise RequestTooLarge("fixture reads one part per step")
    return render_step(step)


class Mark(BaseModel):
    id: str
    read: list[str]
    done: bool


class Marks(BaseModel):
    rows: list[Mark]


def carry_marks(response, step: ChainStep):
    return [(row.id, step.states[row.id] + tuple(row.read) + (("done",) if row.done else ())) for row in response.rows]


def marked(done: Callable[[str, str], bool]) -> Callable[[str], Marks]:
    """Answer every requested item, marking it done when ``done(item_id, prompt)`` holds."""
    def respond(prompt: str) -> Marks:
        request = prompt.split("<correction>")[0]
        return Marks(rows=[
            Mark(id=item_id, read=prompt_parts(request), done=done(item_id, prompt)) for item_id in prompt_ids(request)
        ])

    return respond


def is_done(state) -> bool:
    return bool(state) and state[-1] == "done"


def deadline() -> StructuredLlmError:
    return StructuredLlmError("deadline", terminal_category="deadline_exceeded", error_code="logical_deadline_exceeded")


SIZE_FAILURES = {
    "deadline_exceeded": deadline,
    "input_capacity_exceeded": lambda: StructuredLlmError(
        "context window", terminal_category="provider_error", error_code=INPUT_CAPACITY_EXCEEDED),
    "payload_too_large": lambda: StructuredLlmError(
        "413", terminal_category="provider_error", error_code=PAYLOAD_TOO_LARGE),
    "output_truncated": lambda: StructuredLlmError(
        "truncated", terminal_category="invalid_response", error_code=OUTPUT_TRUNCATED),
}


async def test_items_pack_into_fitting_requests_in_order_with_bounded_output():
    client = FixtureBudgetClient(respond=answer, input_tokens=5)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(10)))

    assert results == {item_id: ((),) for item_id in ids(10)}
    assert [prompt_ids(prompt) for prompt in client.prompts] == [list(ids(10)[:4]), list(ids(10)[4:8]), list(ids(10)[8:])]
    assert client.max_tokens == [client.output_tokens] * 3


@pytest.mark.parametrize("cause", sorted(SIZE_FAILURES))
async def test_size_failures_split_multi_item_requests_until_they_complete(cause):
    client = FixtureBudgetClient(respond=failing(SIZE_FAILURES[cause](), when=lambda prompt: len(prompt_ids(prompt)) > 2))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(8)))

    assert results == {item_id: ((),) for item_id in ids(8)}
    assert runner.stats.splits == 3
    assert runner.stats.calls == 7
    assert runner.stats.failed_requests == 0


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (deadline(), "deadline_exceeded"),
        (SIZE_FAILURES["payload_too_large"](), "capacity_exceeded"),
        (SIZE_FAILURES["input_capacity_exceeded"](), "capacity_exceeded"),
        (SIZE_FAILURES["output_truncated"](), "invalid_response"),
    ],
)
async def test_a_single_item_that_still_fails_returns_a_typed_failure(error, category):
    client = FixtureBudgetClient(respond=failing(error, when=lambda _prompt: True))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    [(item_id, failure)] = (await runner.run_items(item_task(client, ids(1)))).items()

    assert failure == ItemFailure(category, error.error_code, error)
    assert runner.stats.calls == 1
    assert runner.stats.failed_requests == 1


async def test_an_item_that_alone_exceeds_capacity_fails_without_a_call():
    client = FixtureBudgetClient(respond=answer, input_tokens=1)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(1)))

    assert results == {"i00": ItemFailure("capacity_exceeded", INPUT_CAPACITY_EXCEEDED)}
    assert client.prompts == []
    task = item_task(client, ids(1))
    plan = runner.plan_items(task.item_ids, task.render)
    assert plan.requests == () and plan.unfit == ("i00",)


async def test_a_renderer_that_cannot_build_a_slice_counts_as_not_fitting():
    client = FixtureBudgetClient(respond=answer)

    def render_pairs(item_ids, context):
        if len(item_ids) > 2:
            raise RequestTooLarge("catalog references exhausted")
        return render(item_ids, context)

    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    results = await runner.run_items(item_task(client, ids(5), render=render_pairs))

    assert results == {item_id: ((),) for item_id in ids(5)}
    assert [len(prompt_ids(prompt)) for prompt in client.prompts] == [2, 2, 1]


async def test_a_split_half_that_cannot_be_rendered_keeps_halving():
    client = FixtureBudgetClient(respond=failing(deadline(), when=lambda prompt: len(prompt_ids(prompt)) > 2))

    def render_after_split(item_ids, context):
        # The packed request renders; a half of three items does not.
        if len(item_ids) == 3:
            raise RequestTooLarge("catalog references exhausted")
        return render(item_ids, context)

    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    results = await runner.run_items(item_task(client, ids(6), render=render_after_split))

    assert results == {item_id: ((),) for item_id in ids(6)}
    assert sorted(len(prompt_ids(prompt)) for prompt in client.prompts) == [1, 1, 2, 2, 6]
    assert runner.stats.splits == 3


async def test_other_provider_errors_fail_every_item_without_splitting():
    error = StructuredLlmError("rate limited", terminal_category="provider_error", error_code="RateLimitError")
    client = FixtureBudgetClient(respond=failing(error, when=lambda _prompt: True))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(4)))

    assert results == dict.fromkeys(ids(4), ItemFailure("provider_error", "RateLimitError", error))
    assert runner.stats.calls == 1
    assert runner.stats.splits == 0


async def test_a_request_error_fails_every_item_without_splitting_or_counting_as_unjudgeable():
    rejected = StructuredLlmError("bad request", terminal_category="request_error", error_code="BadRequestError")
    client = FixtureBudgetClient(respond=failing(rejected, when=lambda _prompt: True))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(4)))

    assert results == dict.fromkeys(ids(4), ItemFailure("request_error", "BadRequestError", rejected))
    assert not results["i00"].unjudgeable
    assert (runner.stats.calls, runner.stats.splits) == (1, 0)


async def test_an_unexpected_exception_from_the_call_raises_without_splitting():
    def broken(prompt):
        raise KeyError("choices")

    client = FixtureBudgetClient(respond=broken)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    with pytest.raises(KeyError):
        await runner.run_items(item_task(client, ids(4)))
    assert (runner.stats.calls, runner.stats.splits) == (1, 0)


class SelectionError(ValueError):
    pass


def rejecting(item_id: str, *, until_reasked: bool = False):
    """Decode every row, rejecting the row of ``item_id``; with ``until_reasked``, only before its re-ask."""
    def decode_rows(response, item_ids, context):
        for row_id, read in decode(response, item_ids, context):
            if row_id == item_id and not (until_reasked and len(item_ids) == 1):
                yield row_id, RejectedRow(f"{row_id} names p99, which is not supplied")
            else:
                yield row_id, read

    return decode_rows


def unreadable(item_id: str):
    """Any response to a request that carries ``item_id`` cannot be read into rows."""
    def reject(response, item_ids, context):
        if item_id in item_ids:
            raise SelectionError("unknown ref")
        return decode(response, item_ids, context)

    return reject


def reasked_ids(prompt: str) -> list[str]:
    return prompt_ids(prompt.split("<correction>")[0])


@pytest.mark.parametrize(
    "first_reply",
    [
        pytest.param(lambda rows: rows[:-1], id="missing"),
        pytest.param(lambda rows: [*rows, rows[-1]], id="duplicate"),
    ],
)
async def test_a_missing_or_duplicated_row_is_re_asked_alone(first_reply):
    def respond(prompt):
        rows = answer(prompt).rows
        return Rows(rows=rows if "<correction>" in prompt else first_reply(rows))

    client = FixtureBudgetClient(respond=respond)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(3)))

    assert results == {item_id: ((),) for item_id in ids(3)}
    # The two accepted rows are never sent again; only i02 is re-asked, naming its error.
    assert [reasked_ids(prompt) for prompt in client.prompts] == [list(ids(3)), ["i02"]]
    assert "i02" in client.prompts[1].split("<correction>")[1]
    assert (runner.stats.calls, runner.stats.reasks, runner.stats.corrections) == (2, 1, 0)


async def test_a_shifted_answer_is_rejected_as_a_whole_and_no_shifted_row_is_accepted():
    """The model answers each item under its neighbour's ID, so the last answer names an ID it was never given."""
    def shifted(prompt):
        rows = answer(prompt).rows
        if "<correction>" in prompt:
            return Rows(rows=rows)
        return Rows(rows=[Row(id=f"i{int(row.id[1:]) + 1:02d}", read=["shifted"]) for row in rows])

    client = FixtureBudgetClient(respond=shifted)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(4)))

    # No row of the shifted response was accepted: every item carries the corrected answer.
    assert results == {item_id: ((),) for item_id in ids(4)}
    assert "did not supply" in client.prompts[1] and "i04" in client.prompts[1].split("<correction>")[1]
    assert (runner.stats.calls, runner.stats.corrections, runner.stats.reasks, runner.stats.splits) == (2, 1, 0, 0)


async def test_an_answer_that_stays_shifted_is_split_like_unreadable_output():
    def shifted(prompt):
        rows = answer(prompt).rows
        return Rows(rows=[Row(id=f"i{int(row.id[1:]) + 1:02d}", read=[]) for row in rows])

    client = FixtureBudgetClient(respond=shifted)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(2)))

    assert {(failure.category, failure.error_code) for failure in results.values()} == {
        ("invalid_response", OUTPUT_INVALID)
    }
    assert runner.stats.splits == 1


async def test_rejected_rows_cost_one_re_ask_whatever_their_number_and_the_request_size():
    """A request of 53 items with 2 rejected rows: the 51 accepted rows stay, the 2 are re-asked together."""
    item_count = 53
    rejected = {"i07", "i41"}

    def decode_rows(response, item_ids, context):
        for row_id, read in decode(response, item_ids, context):
            first_attempt = len(item_ids) == item_count
            yield row_id, RejectedRow(f"{row_id} is invalid") if row_id in rejected and first_attempt else read

    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(item_count), decode=decode_rows))

    assert results == {item_id: ((),) for item_id in ids(item_count)}
    assert [sorted(reasked_ids(prompt)) for prompt in client.prompts] == [list(ids(item_count)), sorted(rejected)]
    assert (runner.stats.calls, runner.stats.splits) == (2, 0)


async def test_a_row_still_rejected_after_its_re_ask_is_unjudgeable_and_costs_one_extra_call():
    item_count = 53
    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(item_count), decode=rejecting("i30")))

    failure = results.pop("i30")
    assert results == {item_id: ((),) for item_id in ids(item_count) if item_id != "i30"}
    assert (failure.category, failure.error_code, failure.unjudgeable) == ("invalid_response", OUTPUT_INVALID, True)
    assert isinstance(failure.error, RejectedRow) and "i30 names p99" in str(failure.error)
    assert [reasked_ids(prompt) for prompt in client.prompts] == [list(ids(item_count)), ["i30"]]
    assert (runner.stats.calls, runner.stats.splits) == (2, 0)


async def test_a_re_ask_that_does_not_fit_together_is_packed_by_capacity():
    # Each re-ask carries its error lines: one rejected item fits a request (41 words), two do not (52).
    client = FixtureBudgetClient(respond=answer, input_tokens=45)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    def decode_rows(response, item_ids, context):
        for row_id, read in decode(response, item_ids, context):
            first_attempt = len(item_ids) == 4
            yield row_id, RejectedRow(" ".join([row_id, "is", "invalid", *["pad"] * 6])) if first_attempt else read

    results = await runner.run_items(item_task(client, ids(4), decode=decode_rows))

    assert results == {item_id: ((),) for item_id in ids(4)}
    assert [reasked_ids(prompt) for prompt in client.prompts] == [list(ids(4)), ["i00"], ["i01"], ["i02"], ["i03"]]
    assert runner.stats.reasks == 4


async def test_output_that_cannot_be_read_into_rows_is_corrected_once_then_split():
    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(4), decode=unreadable("i02")))

    failure = results.pop("i02")
    assert results == {item_id: ((),) for item_id in ("i00", "i01", "i03")}
    assert (failure.category, failure.error_code, failure.unjudgeable) == ("invalid_response", OUTPUT_INVALID, True)
    assert isinstance(failure.error, SelectionError)
    # The failing item cannot be named, so each request that carries i02 gets its one correction and is halved.
    assert [reasked_ids(prompt) for prompt in client.prompts] == [
        list(ids(4)), list(ids(4)), ["i00", "i01"], ["i02", "i03"], ["i02", "i03"], ["i02"], ["i02"], ["i03"],
    ]
    assert (runner.stats.splits, runner.stats.corrections, runner.stats.failed_requests) == (2, 3, 1)


async def test_malformed_output_for_one_item_is_isolated_without_a_runner_correction():
    malformed = StructuredLlmError(
        "ambiguous structured JSON objects", terminal_category="invalid_response", error_code="ValueError",
    )
    client = FixtureBudgetClient(respond=failing(malformed, when=lambda prompt: "i02" in prompt_ids(prompt)))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(4)))

    assert results.pop("i02") == ItemFailure("invalid_response", "ValueError", malformed)
    assert results == {item_id: ((),) for item_id in ("i00", "i01", "i03")}
    # The client already repaired the response once, so the runner only halves.
    assert [prompt_ids(prompt) for prompt in client.prompts] == [
        list(ids(4)), ["i00", "i01"], ["i02", "i03"], ["i02"], ["i03"],
    ]


async def test_a_retried_run_reuses_accepted_rows_and_the_re_ask_without_a_call():
    store = MemoryWorkStore()
    first = FixtureBudgetClient(respond=answer)
    decode_rows = rejecting("i02", until_reasked=True)
    await LlmBatchRunner(first, model=FIXTURE_MODEL).run_items(
        item_task(first, ids(4), decode=decode_rows, journal=fixture_journal(store)))
    assert len(first.prompts) == 2

    second = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(second, model=FIXTURE_MODEL)
    results = await runner.run_items(item_task(second, ids(4), decode=decode_rows, journal=fixture_journal(store)))

    assert results == {item_id: ((),) for item_id in ids(4)}
    assert second.prompts == [] and runner.stats.reused == 2


async def test_a_re_ask_that_does_not_fit_is_not_sent():
    client = FixtureBudgetClient(respond=lambda prompt: Rows(rows=[]), input_tokens=4)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(3)))

    assert {(failure.category, failure.error_code) for failure in results.values()} == {
        ("invalid_response", OUTPUT_INVALID)
    }
    assert runner.stats.calls == 1


async def test_shared_context_is_chunked_for_an_item_that_needs_it():
    client = FixtureBudgetClient(
        respond=failing(deadline(), when=lambda prompt: "i01" in prompt and "p3" in prompt),
        input_tokens=5,
    )
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(2), context=parts(6)))

    assert results["i00"] == (("p0", "p1", "p2"), ("p3", "p4", "p5"))
    assert results["i01"].category == "deadline_exceeded"


async def test_chain_carries_state_from_step_to_step():
    client = FixtureBudgetClient(respond=answer, input_tokens=5)
    steps = []

    def record(step):
        steps.append(step)
        return render_step(step)

    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    results = await runner.run_chain(chain_task(client, ids(2), parts(4), render=record))

    assert results == dict.fromkeys(ids(2), parts(4))
    sent = [step for step in steps if step.parts in {parts(4)[:2], parts(4)[2:]}]
    assert sent[-1].states == dict.fromkeys(ids(2), ("p0", "p1"))
    assert sent[-1].position == 2


async def test_chain_splits_a_lane_that_cannot_read_one_part_together():
    client = FixtureBudgetClient(respond=answer, input_tokens=4)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_chain(chain_task(client, ids(4), parts(2)))

    assert results == dict.fromkeys(ids(4), parts(2))
    assert all(len(prompt_ids(prompt)) == 2 for prompt in client.prompts)


async def test_chain_halves_the_parts_of_a_single_item_step_before_failing():
    client = FixtureBudgetClient(respond=failing(deadline(), when=lambda prompt: len(prompt_parts(prompt)) > 1))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_chain(chain_task(client, ids(1), parts(4)))

    assert results == {"i00": parts(4)}
    assert runner.stats.splits == 2


async def test_chain_single_item_single_part_failure_leaves_other_lanes_complete():
    error = deadline()
    client = FixtureBudgetClient(respond=failing(error, when=lambda prompt: "i01" in prompt))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_chain(chain_task(client, ids(2), parts(1)))

    assert results == {"i00": ("p0",), "i01": ItemFailure("deadline_exceeded", error.error_code, error, part=0)}


async def test_chain_output_that_cannot_be_read_into_rows_splits_the_lane_until_the_item_stands_alone():
    def carry_unless_i01(response, step: ChainStep):
        if "i01" in step.item_ids:
            raise SelectionError("unknown ref")
        return carry(response, step)

    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_chain(chain_task(client, ids(3), parts(2), decode=carry_unless_i01))

    failure = results.pop("i01")
    assert results == {"i00": parts(2), "i02": parts(2)}
    assert (failure.category, failure.error_code, failure.part) == ("invalid_response", OUTPUT_INVALID, 0)
    # The whole lane, then the half with i01, then i01 alone, each with its correction; the halving reads
    # no fewer parts, so parts are never halved for invalid output.
    assert [prompt_ids(prompt.split("<correction>")[0]) for prompt in client.prompts] == [
        list(ids(3)), list(ids(3)), ["i00"], ["i01", "i02"], ["i01", "i02"], ["i01"], ["i01"], ["i02"],
    ]
    assert all(prompt_parts(prompt.split("<correction>")[0]) == list(parts(2)) for prompt in client.prompts)
    assert (runner.stats.splits, runner.stats.failed_requests) == (2, 1)


async def test_chain_re_asks_a_rejected_row_with_the_same_step_and_state_then_rejoins_the_lane():
    def carry_rows(response, step: ChainStep):
        for item_id, state in carry(response, step):
            # i01's first answer to the second step is rejected; its re-ask is accepted.
            rejected = item_id == "i01" and step.position == 1 and len(step.item_ids) > 1
            yield item_id, RejectedRow("i01 names p9, which is not supplied") if rejected else state

    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(client, ids(3), parts(3), render=one_part_per_step, decode=carry_rows)

    results = await runner.run_chain(task)

    assert results == dict.fromkeys(ids(3), parts(3))
    # The re-ask reads the same part at the same position, for i01 alone; then the lane reads on together.
    assert [(reasked_ids(prompt), prompt_parts(prompt.split("<correction>")[0])) for prompt in client.prompts] == [
        (list(ids(3)), ["p0"]), (list(ids(3)), ["p1"]), (["i01"], ["p1"]), (list(ids(3)), ["p2"]),
    ]


async def test_chain_row_still_rejected_after_its_re_ask_fails_that_item_at_its_part():
    def carry_rows(response, step: ChainStep):
        for item_id, state in carry(response, step):
            yield item_id, RejectedRow("i01 names p9, which is not supplied") if item_id == "i01" else state

    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_chain(chain_task(client, ids(2), parts(2), render=one_part_per_step, decode=carry_rows))

    failure = results.pop("i01")
    assert results == {"i00": parts(2)}
    assert (failure.category, failure.part, failure.unjudgeable) == ("invalid_response", 0, True)
    # i00 reads on alone; i01 had one re-ask.
    assert [reasked_ids(prompt) for prompt in client.prompts] == [["i00", "i01"], ["i01"], ["i00"]]


async def test_chain_item_exits_only_after_its_first_part():
    def done(item_id, prompt):
        read = prompt_parts(prompt.split("<correction>")[0])
        premature = item_id == "i01" and read == ["p0"] and "<correction>" not in prompt
        return item_id == "i00" or read == ["p3"] or premature

    client = FixtureBudgetClient(respond=marked(done))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(
        client, ids(2), parts(4), render=one_part_per_step, decode=carry_marks, finished=is_done,
        first_part_end={"i00": 1, "i01": 3},
    )

    results = await runner.run_chain(task)

    # i01 tried to finish inside its first part: i00's row was accepted, and only i01 re-read the step.
    assert reasked_ids(client.prompts[1]) == ["i01"] and prompt_parts(client.prompts[1].split("<correction>")[0]) == ["p0"]
    assert "i01 finished before reading its first part" in client.prompts[1]
    assert (runner.stats.reasks, runner.stats.corrections) == (1, 0)
    assert results == {"i00": ("p0", "done"), "i01": ("p0", "p1", "p2", "p3", "done")}
    # A finished item leaves the chain: later requests no longer carry it.
    assert [prompt_ids(prompt) for prompt in client.prompts[2:]] == [["i01"]] * 3
    assert "i00" not in "".join(client.prompts[2:])


async def test_chain_reads_the_lane_first_parts_together_before_the_rest():
    client = FixtureBudgetClient(respond=marked(lambda item_id, prompt: item_id == "i00" or "p3" in prompt_parts(prompt)))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(
        client, ids(2), parts(4), decode=carry_marks, finished=is_done, first_part_end={"i00": 1, "i01": 3},
    )

    results = await runner.run_chain(task)

    # The first step ends at the farthest first-part boundary; the rest follows once.
    assert [prompt_parts(prompt) for prompt in client.prompts] == [["p0", "p1", "p2"], ["p3"]]
    assert results == {"i00": ("p0", "p1", "p2", "done"), "i01": ("p0", "p1", "p2", "p3", "done")}


async def test_chain_items_with_different_first_parts_share_one_request_when_they_fit():
    item_ids = ids(20)
    client = FixtureBudgetClient(respond=marked(lambda _item, _prompt: True))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(
        client, item_ids, parts(24), decode=carry_marks, finished=is_done,
        first_part_end={item_id: index + 1 for index, item_id in enumerate(item_ids)},
    )

    results = await runner.run_chain(task)

    assert len(client.prompts) == 1 and prompt_parts(client.prompts[0]) == list(parts(20))
    assert all(state[-1] == "done" for state in results.values())


async def test_chain_rejects_a_first_part_outside_the_parts():
    client = FixtureBudgetClient(respond=answer)
    task = chain_task(client, ids(1), parts(2), first_part_end={"i00": 3})

    with pytest.raises(ValueError, match="outside the 2 chain parts: i00"):
        await LlmBatchRunner(client, model=FIXTURE_MODEL).run_chain(task)
    assert client.prompts == []


async def test_chain_last_step_must_finish_every_item():
    client = FixtureBudgetClient(respond=marked(lambda _item, prompt: "<correction>" in prompt))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(client, ids(1), parts(2), render=one_part_per_step, decode=carry_marks, finished=is_done)

    assert await runner.run_chain(task) == {"i00": ("p0", "p1", "done")}
    assert "<correction>" in client.prompts[-1] and "<correction>" not in client.prompts[-2]

    client = FixtureBudgetClient(respond=marked(lambda _item, _prompt: False))
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)
    task = chain_task(client, ids(1), parts(2), render=one_part_per_step, decode=carry_marks, finished=is_done)
    [failure] = (await runner.run_chain(task)).values()
    assert (failure.category, failure.error_code, failure.part) == ("invalid_response", OUTPUT_INVALID, 1)


async def test_chain_capacity_failure_names_the_part():
    padding = " pad" * 50

    def padded(step: ChainStep) -> LlmRequest:
        request = render_step(step)
        return replace(request, prompt=request.prompt + (padding if "p2" in step.parts else ""))

    client = FixtureBudgetClient(respond=answer, input_tokens=20)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    [failure] = (await runner.run_chain(chain_task(client, ids(1), parts(4), render=padded))).values()

    assert (failure.category, failure.part) == ("capacity_exceeded", 2)
    assert [prompt_parts(prompt) for prompt in client.prompts] == [["p0", "p1"]]


async def test_concurrency_stays_bounded_while_requests_split():
    client = FixtureBudgetClient(
        respond=failing(deadline(), when=lambda prompt: len(prompt_ids(prompt)) > 1),
        input_tokens=3,
        max_concurrent=2,
    )
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    results = await runner.run_items(item_task(client, ids(10)))

    assert results == {item_id: ((),) for item_id in ids(10)}
    assert client.peak_in_flight == 2


async def test_plans_match_the_requests_that_are_sent():
    client = FixtureBudgetClient(respond=answer, input_tokens=5)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    task = item_task(client, ids(6), context=parts(2))
    planned = runner.plan_items(task.item_ids, task.render, task.context)
    await runner.run_items(task)
    assert [entry.request.prompt for entry in planned.requests] == client.prompts
    assert planned.unfit == ()

    unfit = LlmBatchRunner(FixtureBudgetClient(respond=answer, input_tokens=2), model=None).plan_items(
        ids(1), render, parts(1))
    assert unfit.requests == () and unfit.unfit == ("i00",)


class MemoryWorkStore:
    def __init__(self):
        self.works = {}

    async def stage_derivation_work(self, *, derivation_id, work):
        return self.works.setdefault(work.id, work)

    async def record_derivation_work(self, *, derivation_id, work):
        self.works[work.id] = work
        return work


def fixture_journal(store: MemoryWorkStore) -> DerivationWorkJournal:
    return DerivationWorkJournal(store=store, derivation_id="derivation", kind="claim_assess",
                                 scope={"contract": "fixture"}, budget_identity="budget", model=FIXTURE_MODEL)


async def test_journal_reuses_completed_requests_and_records_failures():
    store = MemoryWorkStore()

    def journal():
        return fixture_journal(store)

    outage = StructuredLlmError("down", terminal_category="provider_error", error_code="ServiceUnavailableError")
    first = FixtureBudgetClient(respond=failing(outage, when=lambda prompt: "i03" in prompt), input_tokens=3)
    failed = await LlmBatchRunner(first, model=FIXTURE_MODEL).run_items(item_task(first, ids(4), journal=journal()))
    assert failed["i03"].category == "provider_error"
    assert sorted(work.status for work in store.works.values()) == ["completed", "retryable_failure"]

    second = FixtureBudgetClient(respond=answer, input_tokens=3)
    retry_journal = journal()
    runner = LlmBatchRunner(second, model=FIXTURE_MODEL)
    results = await runner.run_items(item_task(second, ids(4), journal=retry_journal))

    assert results == {item_id: ((),) for item_id in ids(4)}
    assert runner.stats.reused == 1
    assert [prompt_ids(prompt) for prompt in second.prompts] == [["i02", "i03"]]
    assert [work.status for work in retry_journal.works_for("i00")] == ["completed"]
    assert {work.status for work in store.works.values()} == {"completed"}


async def test_a_journaled_response_that_no_longer_decodes_is_a_typed_failure():
    class CompletedStore:
        async def stage_derivation_work(self, *, derivation_id, work):
            return replace(work, status="completed", result=Rows(rows=[]).model_dump(mode="json"))

    journal = DerivationWorkJournal(store=CompletedStore(), derivation_id="derivation", kind="claim_assess",
                                    scope={"contract": "fixture"}, budget_identity="budget", model=FIXTURE_MODEL)
    client = FixtureBudgetClient(respond=answer)

    results = await LlmBatchRunner(client, model=FIXTURE_MODEL).run_items(item_task(client, ids(2), journal=journal))

    assert {failure.error_code for failure in results.values()} == {OUTPUT_INVALID}
    assert client.prompts == []


def budget_client(monkeypatch, *, metadata, **caps) -> LiteLlmStructuredClient:
    def model_info(_model):
        if metadata is None:
            raise Exception("unknown model")
        return metadata

    monkeypatch.setattr("memforge.llm.request_budget.litellm.get_model_info", model_info)
    monkeypatch.setattr(
        "memforge.llm.structured.litellm.token_counter",
        lambda *, model, messages: len(messages[0]["content"].split()),
    )
    return LiteLlmStructuredClient(StructuredLlmConfig(
        model="provider/fixture-model", base_url=None, api_key=None, timeout_s=1.0,
        input_budget_fraction=1.0, **caps,
    ))


@pytest.mark.parametrize(
    ("metadata", "caps"),
    [
        ({"max_input_tokens": 4_000, "context_window": 8_000, "max_output_tokens": 50}, {}),
        (None, {"max_input_tokens": 4_000, "context_window_tokens": 8_000, "max_output_tokens": 50}),
    ],
    ids=["litellm-metadata", "operator-caps"],
)
def test_capacity_comes_from_litellm_metadata_or_operator_caps(monkeypatch, metadata, caps):
    runner = LlmBatchRunner(budget_client(monkeypatch, metadata=metadata, **caps), model=None)

    assert runner.fits(LlmRequest("short prompt", Rows, REQUESTED_OUTPUT))
    assert not runner.fits(LlmRequest("word " * 5_000, Rows, REQUESTED_OUTPUT))


async def test_a_route_without_capacity_metadata_or_caps_raises(monkeypatch):
    client = budget_client(monkeypatch, metadata=None)
    runner = LlmBatchRunner(client, model=None)

    with pytest.raises(ValueError, match="has no max_input_tokens metadata"):
        await runner.run_items(ItemTask(item_ids=ids(1), render=render, decode=decode, call=client.rerank_memories))


@pytest.mark.parametrize(
    ("input_tokens", "split_above"),
    [(100, 100), (3, 100), (100, 1), (6, 2)],
)
async def test_results_do_not_depend_on_packing_or_split_points(input_tokens, split_above):
    item_client = FixtureBudgetClient(
        respond=failing(deadline(), when=lambda prompt: len(prompt_ids(prompt)) > split_above),
        input_tokens=input_tokens,
    )
    items = await LlmBatchRunner(item_client, model=FIXTURE_MODEL).run_items(item_task(item_client, ids(7)))
    assert items == {item_id: ((),) for item_id in ids(7)}

    chain_client = FixtureBudgetClient(
        respond=failing(deadline(), when=lambda prompt: len(prompt_ids(prompt)) > split_above),
        input_tokens=input_tokens + 2,
    )
    chain = await LlmBatchRunner(chain_client, model=FIXTURE_MODEL).run_chain(chain_task(chain_client, ids(5), parts(4)))
    assert chain == dict.fromkeys(ids(5), parts(4))


async def test_run_one_sends_an_indivisible_request():
    client = FixtureBudgetClient(respond=answer)
    runner = LlmBatchRunner(client, model=FIXTURE_MODEL)

    ranking = await runner.run_one(render(ids(3), ()), call=client.call, decode=lambda response: len(response.rows))
    assert ranking == 3

    timeout = deadline()
    client.respond = failing(timeout, when=lambda _prompt: True)
    assert await runner.run_one(render(ids(3), ()), call=client.call) == ItemFailure(
        "deadline_exceeded", timeout.error_code, timeout)
    assert runner.stats.splits == 0
