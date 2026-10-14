"""The Memory vector index: the Chroma collection that holds the vectors in use.

The database records each physical index once, with the fingerprint of the
Embedding Profile it was built with, and marks one of them as in use.
``open_memory_vector_index`` is the one way to that collection: it returns the
index in use together with its fingerprint, so whoever embeds for it can tell
whether its vectors belong there.

An installation starts without a recorded index. Its first one is settled the
first time the Embedding Profile in use is known: the collection of an
installation that already has vectors is adopted when those vectors have the
profile's length, and an installation with no vectors gets a new, empty index.
Vectors of another length cannot be adopted; that installation has no index
in use until one is rebuilt, and they are deleted once it has one.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
import numpy as np
from chromadb.errors import NotFoundError

from memforge.llm.embedding_profile import EmbeddingProfile

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "STORED_VECTORS_COLLECTION",
    "MemoryVectorIndex",
    "collection_name_for",
    "count_valid_vectors",
    "create_index_collection",
    "delete_index_collection",
    "first_index_record",
    "forget_index_collection",
    "open_memory_vector_index",
    "settle_memory_vector_index",
    "stored_vector_dimension",
]

# The collection the vectors of an installation with no recorded index are in.
STORED_VECTORS_COLLECTION = "memories"
_COLLECTION_METADATA = {"hnsw:space": "cosine"}
_COLLECTION_PREFIX = "memories-"
# Collection names are limited in length; this much of a fingerprint is unique.
_FINGERPRINT_NAME_LENGTH = 32
_VERIFY_PAGE_SIZE = 500

# One client per path: several PersistentClient instances on one path contend
# for the SQLite file inside ChromaDB.
_chroma_clients: dict[str, chromadb.ClientAPI] = {}
_chroma_lock = threading.Lock()
# The open collection of each recorded index, by path and index ID. An index
# ID names one physical collection for as long as that collection exists.
_open_collections: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class MemoryVectorIndex:
    """The Memory vector index in use: its collection and the fingerprint it was built with."""

    collection: Any
    # ``None`` for a collection that records none; such an index is bound to no
    # embedding model.
    fingerprint: str | None = None
    id: str | None = None
    name: str | None = None
    dimension: int | None = None
    # Whether the vectors were in place before the index was recorded, so that
    # the model they came from is assumed, not known.
    adopted: bool = False


async def open_memory_vector_index(db: "Database", chroma_path: str) -> MemoryVectorIndex | None:
    """Open the Memory vector index in use, or return ``None`` when there is none.

    Every reader and writer of Memory vectors comes through here, each time it
    needs the index, so all of them move to a rebuilt index the moment it is
    put into use.
    """
    record = await db.get_active_vector_index()
    if record is None:
        return None
    key = (chroma_path, str(record["id"]))
    collection = _open_collections.get(key)
    if collection is None:
        collection = await asyncio.to_thread(_get_or_create, chroma_path, str(record["collection_name"]))
        _open_collections[key] = collection
    return MemoryVectorIndex(
        collection=collection,
        fingerprint=str(record["fingerprint"]),
        id=str(record["id"]),
        name=str(record["collection_name"]),
        dimension=int(record["dimension"]),
        adopted=bool(record["adopted"]),
    )


async def settle_memory_vector_index(
    db: "Database",
    chroma_path: str,
    profile: EmbeddingProfile,
) -> dict[str, Any] | None:
    """Give an installation with no index in use its first one, for ``profile``.

    Returns the record of the index in use afterwards. ``None`` means the
    installation holds vectors of another length than the profile's and has to
    rebuild its index.
    """
    in_use = await db.get_active_vector_index()
    if in_use is not None:
        return in_use
    record = await asyncio.to_thread(first_index_record, chroma_path, profile)
    if record is None:
        return None
    await db.record_active_vector_index(record)
    in_use = await db.get_active_vector_index()
    if in_use is not None and in_use["id"] == record["id"]:
        logger.info(
            "The Memory vector index in use is %s (%s)",
            record["collection_name"],
            "adopted" if record["adopted"] else "new",
        )
    return in_use


def first_index_record(chroma_path: str, profile: EmbeddingProfile) -> dict[str, Any] | None:
    """Describe the first index of an installation, for ``profile``.

    Stored vectors of the profile's length are adopted where they are, and the
    model they came from is taken to be the profile's. With no stored vectors
    the index is a new, empty collection named from the fingerprint. ``None``
    means stored vectors of another length: nothing can be adopted.
    """
    stored_dimension = stored_vector_dimension(chroma_path)
    if stored_dimension is None:
        delete_index_collection(chroma_path, STORED_VECTORS_COLLECTION)
        name = collection_name_for(profile.fingerprint)
        _get_or_create(chroma_path, name)
        return _index_record(profile, name, adopted=False, vector_count=0)
    if stored_dimension != profile.dimension:
        return None
    stored = _existing(chroma_path, STORED_VECTORS_COLLECTION)
    return _index_record(
        profile,
        STORED_VECTORS_COLLECTION,
        adopted=True,
        vector_count=0 if stored is None else int(stored.count()),
    )


def stored_vector_dimension(chroma_path: str) -> int | None:
    """The length of one vector an installation with no recorded index holds, if it holds any."""
    stored = _existing(chroma_path, STORED_VECTORS_COLLECTION)
    if stored is None:
        return None
    embeddings = stored.get(limit=1, include=["embeddings"]).get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return len(embeddings[0])


def collection_name_for(fingerprint: str) -> str:
    """The name of the collection of a new index built with ``fingerprint``."""
    return f"{_COLLECTION_PREFIX}{fingerprint[:_FINGERPRINT_NAME_LENGTH]}"


def create_index_collection(chroma_path: str, name: str) -> Any:
    """Create the empty collection ``name``, replacing whatever an unfinished build left under it."""
    delete_index_collection(chroma_path, name)
    return _get_or_create(chroma_path, name)


def delete_index_collection(chroma_path: str, name: str) -> None:
    """Delete the collection ``name`` and its vectors, if it exists.

    ChromaDB deletes the record of a collection and leaves the files of its
    vector segment where they are, so those are removed here.
    """
    collection = _existing(chroma_path, name)
    if collection is None:
        return
    segment_directories = _segment_directories(chroma_path, str(collection.id))
    try:
        _client(chroma_path).delete_collection(name)
    except (NotFoundError, ValueError):
        return
    for directory in segment_directories:
        shutil.rmtree(directory, ignore_errors=True)


def forget_index_collection(chroma_path: str, index_id: str) -> None:
    """Drop the open collection of an index that is no longer in use."""
    _open_collections.pop((chroma_path, index_id), None)


def count_valid_vectors(collection: Any, *, dimension: int) -> tuple[int, int]:
    """Count the vectors in ``collection`` and those that are valid.

    A vector is valid when it has ``dimension`` numbers and every one of them
    is finite.
    """
    total = 0
    valid = 0
    while True:
        page = collection.get(include=["embeddings"], limit=_VERIFY_PAGE_SIZE, offset=total)
        ids = page.get("ids") or []
        if not ids:
            return total, valid
        embeddings = page.get("embeddings")
        for position in range(len(ids)):
            vector = None if embeddings is None or position >= len(embeddings) else embeddings[position]
            if vector is not None and len(vector) == dimension and bool(np.isfinite(vector).all()):
                valid += 1
        total += len(ids)


def _segment_directories(chroma_path: str, collection_id: str) -> list[Path]:
    """The directories that hold the vector files of one collection.

    ChromaDB keeps the files of a segment in a directory named by the
    segment's ID, beside its own database, which lists the segments of each
    collection. A database that cannot be read this way names no directory,
    and the files stay.
    """
    root = Path(chroma_path)
    try:
        with closing(sqlite3.connect(f"{(root / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True)) as chroma:
            rows = chroma.execute("SELECT id FROM segments WHERE collection = ?", (collection_id,)).fetchall()
    except sqlite3.Error:
        logger.debug("Could not list the segments of collection %s", collection_id, exc_info=True)
        return []
    directories: list[Path] = []
    for (segment_id,) in rows:
        try:
            directory = root / str(uuid.UUID(str(segment_id)))
        except ValueError:
            continue
        if directory.is_dir():
            directories.append(directory)
    return directories


def _index_record(
    profile: EmbeddingProfile,
    collection_name: str,
    *,
    adopted: bool,
    vector_count: int,
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "fingerprint": profile.fingerprint,
        "collection_name": collection_name,
        "adopted": adopted,
        "dimension": profile.dimension,
        "profile": profile.fields(),
        "vector_count": vector_count,
    }


def _client(chroma_path: str) -> chromadb.ClientAPI:
    with _chroma_lock:
        client = _chroma_clients.get(chroma_path)
        if client is None:
            client = chromadb.PersistentClient(path=chroma_path)
            _chroma_clients[chroma_path] = client
        return client


def _get_or_create(chroma_path: str, name: str) -> Any:
    return _client(chroma_path).get_or_create_collection(name=name, metadata=_COLLECTION_METADATA)


def _existing(chroma_path: str, name: str) -> Any | None:
    try:
        return _client(chroma_path).get_collection(name)
    except (NotFoundError, ValueError):
        return None
