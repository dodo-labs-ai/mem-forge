"""Why a Memory cannot be changed, or a vector written, at this moment.

A rebuild of the Memory vector index reads every searchable Memory and embeds
it into a new index. It is correct only if the Memories stand still while it
runs, so for its duration no Source activity is admitted and no Memory is
written. Everything refused for that reason carries one code,
``reindex_in_progress``.

An installation can also be without an index in use. A vector that is to be
written then has nowhere to go, and the write waits for an index.
"""

from __future__ import annotations

import sqlite3

__all__ = [
    "REINDEX_IN_PROGRESS",
    "REINDEX_IN_PROGRESS_MESSAGE",
    "REINDEX_NOT_REQUIRED",
    "REINDEX_NOT_REQUIRED_MESSAGE",
    "NoVectorIndex",
    "ReindexInProgress",
    "VectorReindexRefused",
    "is_reindex_refusal",
]

REINDEX_IN_PROGRESS = "reindex_in_progress"
REINDEX_IN_PROGRESS_MESSAGE = (
    "The Memory vector index is being rebuilt. Memories cannot be changed until the rebuild has finished."
)
# A rebuild is refused with this code when the index in use is already the one
# it would build.
REINDEX_NOT_REQUIRED = "reindex_not_required"
REINDEX_NOT_REQUIRED_MESSAGE = "The vector index in use was built with the embedding model in use."


class ReindexInProgress(RuntimeError):
    """A write was refused because the Memory vector index is being rebuilt."""

    code = REINDEX_IN_PROGRESS

    def __init__(self, message: str = REINDEX_IN_PROGRESS_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class VectorReindexRefused(RuntimeError):
    """A rebuild of the Memory vector index could not start, named by a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NoVectorIndex(RuntimeError):
    """A vector was to be written and no Memory vector index is in use.

    An installation has no index in use until its embedding model has answered
    once, and again while it holds vectors of another length than that
    model's, until its index is rebuilt.
    """

    code = "vector_index_missing"


def is_reindex_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` is a write refused because a rebuild is running.

    The database refuses a write to a Memory row with an error whose message is
    the code itself.
    """
    if isinstance(exc, ReindexInProgress):
        return True
    return isinstance(exc, sqlite3.DatabaseError) and str(exc).strip() == REINDEX_IN_PROGRESS
