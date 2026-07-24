import asyncio

import pytest

from memforge.pipeline.bounded_work import collect_bounded


@pytest.mark.asyncio
async def test_collect_bounded_limits_started_work_and_preserves_input_order():
    release = asyncio.Event()
    two_started = asyncio.Event()
    started: list[int] = []

    async def run_item(item: int) -> int:
        started.append(item)
        if len(started) == 2:
            two_started.set()
        await release.wait()
        return item * 10

    task = asyncio.create_task(
        collect_bounded(
            range(5),
            run_item,
            max_concurrent=2,
        )
    )

    await asyncio.wait_for(two_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert started == [0, 1]

    release.set()
    assert await task == [0, 10, 20, 30, 40]


@pytest.mark.asyncio
async def test_collect_bounded_preserves_original_failure_and_cancels_siblings():
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def run_item(item: int) -> int:
        if item == 0:
            await sibling_started.wait()
            raise RuntimeError("extraction failed")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return item

    with pytest.raises(RuntimeError, match="extraction failed"):
        await collect_bounded(
            [0, 1],
            run_item,
            max_concurrent=2,
        )

    assert sibling_cancelled.is_set()
