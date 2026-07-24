"""Bounded async collection without eager per-item task creation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar, cast


_Input = TypeVar("_Input")
_Result = TypeVar("_Result")
_MISSING = object()


async def collect_bounded(
    items: Sequence[_Input],
    worker: Callable[[_Input], Awaitable[_Result]],
    *,
    max_concurrent: int,
) -> list[_Result]:
    """Run at most ``max_concurrent`` workers and preserve input order."""

    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    if not items:
        return []

    results: list[_Result | object] = [_MISSING] * len(items)
    next_index = 0

    async def consume() -> None:
        nonlocal next_index
        while next_index < len(items):
            index = next_index
            next_index += 1
            results[index] = await worker(items[index])

    tasks = [asyncio.create_task(consume()) for _ in range(min(max_concurrent, len(items)))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return [cast(_Result, result) for result in results]
