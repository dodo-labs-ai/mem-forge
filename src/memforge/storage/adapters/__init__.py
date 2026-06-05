"""Pluggable storage adapters: caller context, protocols, and concrete impls.

Each adapter is bound to one datastore at construction. This package is the
boundary a different storage backend plugs in behind.
"""

from __future__ import annotations

from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import (
    KeywordSearch,
    RelationalStore,
    VectorStore,
)

__all__ = [
    "AccessScope",
    "LOCAL_DEV_USER_ID",
    "KeywordSearch",
    "RelationalStore",
    "VectorStore",
]
