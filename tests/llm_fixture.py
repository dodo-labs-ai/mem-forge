"""Deterministic structured-client fixture for LLM batch runner tests.

A request's token count is its word count, so tests control capacity by the
number of item and part words a renderer writes into the prompt.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from memforge.llm.request_budget import RequestBudget

# Large enough that the context window never binds before the input limit.
FIXTURE_CONTEXT_WINDOW = 1_000_000
FIXTURE_MODEL = "fixture/model"


def fixture_budget(*, input_tokens: int, output_tokens: int, correction_reserve: int) -> RequestBudget:
    return RequestBudget(
        model=FIXTURE_MODEL,
        input_limit=input_tokens,
        context_limit=FIXTURE_CONTEXT_WINDOW,
        output_limit=output_tokens,
        fraction=1.0,
        metadata_source="fixture",
        correction_reserve=correction_reserve,
    )


@dataclass
class FixtureBudgetClient:
    """Capacity by prompt word count; ``respond`` answers every call."""

    respond: Callable[[str], BaseModel]
    input_tokens: int = 10_000
    output_tokens: int = 64
    correction_reserve: int = 0
    max_concurrent: int = 1
    prompts: list[str] = field(default_factory=list)
    max_tokens: list[int] = field(default_factory=list)
    peak_in_flight: int = 0
    _in_flight: int = 0

    def request_budget(self, model: str | None = None) -> RequestBudget:
        return fixture_budget(
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            correction_reserve=self.correction_reserve,
        )

    def request_fits(self, prompt, *, response_format, max_tokens, model=None, images=(), reserve_correction=True):
        return self.request_budget(model).fits(
            len(prompt.split()), max_tokens, reserve_correction=reserve_correction,
        )

    async def call(self, prompt: str, *, max_tokens: int, model: str | None, images=()) -> BaseModel:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            # Yield so concurrent workers overlap and the peak is observable.
            await asyncio.sleep(0)
            return self.respond(prompt)
        finally:
            self._in_flight -= 1
