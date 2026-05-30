"""Retry utility for async API calls.

Provides exponential backoff retry for LLM and embedding API calls.
Used by the enricher, memory extractor, and embedding functions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["retry_async"]


async def retry_async(
    coro_factory,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    description: str = "API call",
) -> T:
    """Retry an async operation with exponential backoff.

    Parameters
    ----------
    coro_factory:
        A callable that returns a new coroutine on each call.
        Must be a factory (not a pre-built coroutine) because coroutines
        can only be awaited once.
    max_retries:
        Maximum number of retry attempts (total attempts = max_retries + 1).
    base_delay:
        Base delay in seconds. Actual delay = base_delay * 2^attempt.
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.
    description:
        Human-readable description for log messages.

    Returns
    -------
    The result of the coroutine.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    description, attempt + 1, max_retries + 1, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    description, max_retries + 1, e,
                )

    raise last_exception  # type: ignore[misc]
