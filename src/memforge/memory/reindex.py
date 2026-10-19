"""Rebuild the Memory vector index for another embedding model.

Changing the embedding model changes the vector space, so the vectors of every
searchable Memory are made again. A rebuild embeds them from the relational
rows into a new index, verifies that index, and only then puts it into use, in
the same transaction that makes the embedding binding it was built for the one
in use. Search reads the previous index until that moment. The previous index
is deleted afterwards.

A rebuild that fails at any point before the switch changes nothing that is in
use: the index it was building is dropped, the reason is recorded as a code,
and the rebuild can be started again.

A running rebuild holds a lease, which it renews on a timer for as long as its
process lives, and it records which process it runs in. A rebuild whose process
was killed therefore ends in one of two ways: a process on the same host that
cleans up finds the runner gone and records the rebuild as interrupted at
once, and anyone else sees the lease run out within ``REINDEX_LEASE_SECONDS``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from memforge.llm.capability import embedding_error_code
from memforge.llm.embedding import (
    EmbeddingError,
    EmbeddingIndexMismatch,
    EmbeddingRuntime,
    EmbeddingSettings,
    EmbeddingTransport,
    embed_documents_for_index,
)
from memforge.llm.embedding_profile import EmbeddingProfile
from memforge.llm.providers import ResolvedInvocation
from memforge.memory.lifecycle_plan import LifecycleVectorOperation
from memforge.memory.store import MemoryStore
from memforge.retrieval import vector_index
from memforge.retrieval.vector_index import (
    STORED_VECTORS_COLLECTION,
    MemoryVectorIndex,
    collection_name_for,
    count_valid_vectors,
    create_index_collection,
    delete_index_collection,
    first_index_record,
    forget_index_collection,
    settle_memory_vector_index,
    stored_vector_dimension,
)
from memforge.storage.adapters.sqlite import SqliteVectorStore, build_sqlite_adapters
from memforge.vector_index_errors import REINDEX_NOT_REQUIRED

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "REINDEX_LEASE_SECONDS",
    "MemoryVectorIndexes",
    "ReindexTarget",
]

# How long a rebuild holds the installation after its last renewal: a rebuild
# whose process stopped without recording it counts as interrupted once this
# has passed.
REINDEX_LEASE_SECONDS = 60
# How often a running rebuild renews its lease, whatever else it is doing.
_LEASE_RENEWAL_SECONDS = 15
_BATCH_SIZE = 64
# Tells this process from an earlier one that had the same process ID, as the
# first process of a restarted container has.
_PROCESS_TOKEN = uuid.uuid4().hex

ReindexProgress = Callable[[int, int], Awaitable[None] | None]


@dataclass(frozen=True)
class ReindexTarget:
    """The embedding binding an index is built for."""

    profile: EmbeddingProfile
    invocation: ResolvedInvocation
    # Which binding it is: the one in use, or the candidate waiting for the index.
    binding_slot: Literal["active", "candidate"]
    # The candidate to make the binding in use at the switch, as the database
    # promotes it: ``updated_at``, ``state`` and ``validation``.
    promotion: Mapping[str, Any] | None = None


class _RebuildFailed(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _this_process() -> dict[str, Any]:
    """Name the process a rebuild started here runs in."""
    return {"host": socket.gethostname(), "pid": os.getpid(), "token": _PROCESS_TOKEN}


def _runner_is_gone(runner: Mapping[str, Any] | None) -> bool:
    """Whether the process that ran a rebuild is known to have ended.

    That can be known only of a process on this host. A rebuild that runs
    anywhere else is given up when its lease runs out.
    """
    if not runner or runner.get("host") != socket.gethostname():
        return False
    pid = runner.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    if pid == os.getpid():
        return runner.get("token") != _PROCESS_TOKEN
    return not _process_exists(pid)


def _process_exists(pid: int) -> bool:
    """Whether a process with ``pid`` runs on this host; True where that cannot be asked."""
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # The process exists and belongs to another user.
        return True
    return True


class MemoryVectorIndexes:
    """The Memory vector indexes of one database: the one in use, and how another is built."""

    def __init__(
        self,
        db: "Database",
        chroma_path: str,
        *,
        embedding_transport: EmbeddingTransport | None = None,
    ) -> None:
        self._db = db
        self._chroma_path = chroma_path
        self._embedding_transport = embedding_transport

    async def in_use(self) -> MemoryVectorIndex | None:
        """Open the index in use."""
        return await vector_index.open_memory_vector_index(self._db, self._chroma_path)

    async def settle(self, profile: EmbeddingProfile) -> dict[str, Any] | None:
        """Give an installation with no index in use its first one, for ``profile``."""
        return await settle_memory_vector_index(self._db, self._chroma_path, profile)

    async def first_index_record(self, profile: EmbeddingProfile) -> dict[str, Any] | None:
        """Describe the first index of an installation, to be recorded by the caller."""
        holds_memories = await self._db.count_memories_kept_without_vectors() > 0
        return await asyncio.to_thread(first_index_record, self._chroma_path, profile, holds_memories=holds_memories)

    async def holds_stored_vectors(self) -> bool:
        """Whether the installation holds vectors from before it recorded any index."""
        return await asyncio.to_thread(stored_vector_dimension, self._chroma_path) is not None

    async def start_rebuild(self, target: ReindexTarget) -> dict[str, Any]:
        """Admit a rebuild for ``target`` and return the record of the index it builds.

        Raises ``VectorReindexRefused`` while a Source is syncing or another
        rebuild is running, and when the index in use is already the one the
        rebuild would build.
        """
        await self.clean_up()
        record = await self._db.start_vector_reindex(
            {
                "id": uuid.uuid4().hex,
                "fingerprint": target.profile.fingerprint,
                "collection_name": collection_name_for(target.profile.fingerprint),
                "dimension": target.profile.dimension,
                "profile": target.profile.fields(),
                "binding_slot": target.binding_slot,
                "binding_saved_at": None if target.promotion is None else target.promotion["updated_at"],
                "runner": _this_process(),
            },
            lease_seconds=REINDEX_LEASE_SECONDS,
        )
        # The new rebuild is the latest one; what failed before it is forgotten.
        await self.clean_up()
        for failed in await self._db.list_vector_indexes("failed"):
            await self._db.delete_vector_index(str(failed["id"]))
        return record

    async def rebuild(
        self,
        record: Mapping[str, Any],
        target: ReindexTarget,
        *,
        progress: ReindexProgress | None = None,
        on_switch: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Build, verify and switch to the index ``start_rebuild`` recorded.

        ``on_switch`` is called as soon as the rebuilt index is in use, before
        the replaced index is deleted. Returns the record as the rebuild left
        it: in use, or failed with the code of the reason.
        """
        index_id = str(record["id"])
        lease = asyncio.create_task(self._keep_lease(index_id))
        try:
            vector_count = await self._build(record, target, progress)
            refusal = await self._db.activate_vector_index(
                index_id,
                vector_count=vector_count,
                promote_candidate=target.promotion,
            )
            if refusal is not None:
                raise _RebuildFailed(refusal)
        except asyncio.CancelledError:
            await self._abandon(record, "interrupted")
            raise
        except _RebuildFailed as exc:
            await self._abandon(record, exc.code)
        except EmbeddingIndexMismatch:
            await self._abandon(record, "embedding_dimension_mismatch")
        except EmbeddingError as exc:
            await self._abandon(record, embedding_error_code(exc))
        except Exception:
            logger.exception("The rebuild of the Memory vector index failed")
            await self._abandon(record, "index_error")
        else:
            if on_switch is not None:
                on_switch()
            logger.info(
                "The Memory vector index was rebuilt: %d vectors in %s",
                vector_count,
                record["collection_name"],
            )
            await self._delete_replaced_index()
        finally:
            lease.cancel()
        finished = await self._db.get_vector_index(index_id)
        assert finished is not None
        return finished

    async def clean_up(self) -> None:
        """Delete what indexes no longer in use have left behind.

        A rebuild whose lease has run out, or whose process on this host has
        ended, is recorded as interrupted. The vectors of failed and replaced
        indexes are deleted, and a replaced index is forgotten once its
        vectors are gone. Once an index is in use, vectors the installation
        stored before it recorded any index, and which no index adopted, are
        deleted as well.
        """
        await self._db.fail_interrupted_vector_reindex()
        for building in await self._db.list_vector_indexes("building"):
            if _runner_is_gone(building["runner"]):
                await self._db.fail_vector_reindex(str(building["id"]), error_code="interrupted")
        in_use = await self._db.list_vector_indexes("active")
        kept = await self._held_collections()
        for state in ("failed", "retired"):
            for index in await self._db.list_vector_indexes(state):
                if index["collection_name"] not in kept:
                    await asyncio.to_thread(
                        delete_index_collection,
                        self._chroma_path,
                        str(index["collection_name"]),
                    )
                forget_index_collection(self._chroma_path, str(index["id"]))
                if state == "retired":
                    await self._db.delete_vector_index(str(index["id"]))
        # With no index in use the stored vectors may still be adopted.
        if in_use and STORED_VECTORS_COLLECTION not in kept:
            await asyncio.to_thread(delete_index_collection, self._chroma_path, STORED_VECTORS_COLLECTION)

    async def _held_collections(self) -> set[str]:
        """The collections that the index in use or a running rebuild holds, which stay."""
        return {
            str(index["collection_name"])
            for state in ("active", "building")
            for index in await self._db.list_vector_indexes(state)
        }

    async def _keep_lease(self, index_id: str) -> None:
        """Renew the lease of a running rebuild until the rebuild ends or loses it.

        The renewal does not wait for the rebuild to get anywhere: a batch
        that takes long keeps the lease, and a process that has stopped lets
        it run out.
        """
        while True:
            await asyncio.sleep(_LEASE_RENEWAL_SECONDS)
            try:
                if not await self._db.renew_vector_reindex(index_id, lease_seconds=REINDEX_LEASE_SECONDS):
                    return
            except Exception:
                logger.warning("Could not renew the lease of the Memory vector index rebuild", exc_info=True)

    async def _delete_replaced_index(self) -> None:
        """Clean up after a switch, which stands whether or not the clean-up succeeds.

        Whatever cannot be deleted now is deleted by the next clean-up: when
        the server starts, and before the next rebuild.
        """
        try:
            await self.clean_up()
        except Exception:
            logger.warning(
                "Could not delete the Memory vector index that was replaced; the next clean-up deletes it",
                exc_info=True,
            )

    async def _build(
        self,
        record: Mapping[str, Any],
        target: ReindexTarget,
        progress: ReindexProgress | None,
    ) -> int:
        """Embed every searchable Memory into the new index and return its verified vector count."""
        index_id = str(record["id"])
        dimension = target.profile.dimension
        in_use = await self._db.get_active_vector_index()
        if in_use is not None and in_use["collection_name"] == record["collection_name"]:
            # Creating the collection empties it, and it holds the index in use.
            raise _RebuildFailed(REINDEX_NOT_REQUIRED)
        collection = await asyncio.to_thread(
            create_index_collection,
            self._chroma_path,
            str(record["collection_name"]),
        )
        store = SqliteVectorStore(collection, index_fingerprint=target.profile.fingerprint)
        embedding = EmbeddingRuntime(
            EmbeddingSettings(
                invocation=target.invocation,
                identity=target.profile.identity,
                cache_size=0,
            ),
            transport=self._embedding_transport,
        )
        adapters = build_sqlite_adapters(self._db, store)
        projections = MemoryStore(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=store,
            embedding=embedding,
        )

        total = await self._db.count_searchable_memories()
        indexed = 0
        await self._report(index_id, total, indexed, progress)
        after = ""
        while page := await self._db.list_searchable_memory_ids(after=after, limit=_BATCH_SIZE):
            after = page[-1]
            ids: list[str] = []
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for memory_id in page:
                operation, text, metadata = await projections.vector_projection(memory_id)
                if operation != LifecycleVectorOperation.UPSERT.value:
                    continue
                assert text is not None and metadata is not None
                ids.append(memory_id)
                texts.append(text)
                metadatas.append(metadata)
            if ids:
                vectors = await embed_documents_for_index(embedding, store, texts)
                await store.upsert(ids, vectors, metadatas)
            indexed += len(ids)
            await self._report(index_id, total, indexed, progress)

        searchable = await self._db.count_searchable_memories()
        vector_count, valid = await asyncio.to_thread(count_valid_vectors, collection, dimension=dimension)
        if vector_count != searchable:
            raise _RebuildFailed("vector_count_mismatch")
        if valid != vector_count:
            raise _RebuildFailed("vector_invalid")
        await self._report(index_id, searchable, vector_count, progress)
        return vector_count

    async def _report(
        self,
        index_id: str,
        total: int,
        indexed: int,
        progress: ReindexProgress | None,
    ) -> None:
        """Record how far the rebuild is, and stop it when it no longer holds its lease."""
        renewed = await self._db.renew_vector_reindex(
            index_id,
            lease_seconds=REINDEX_LEASE_SECONDS,
            memory_count=total,
            vector_count=indexed,
        )
        if not renewed:
            raise _RebuildFailed("reindex_lease_lost")
        if progress is not None:
            reported = progress(indexed, total)
            if reported is not None:
                await reported

    async def _abandon(self, record: Mapping[str, Any], error_code: str) -> None:
        """Record the rebuild as failed and drop the index it was building."""
        logger.warning("The rebuild of the Memory vector index failed (%s)", error_code)
        await self._db.fail_vector_reindex(str(record["id"]), error_code=error_code)
        if record["collection_name"] not in await self._held_collections():
            await asyncio.to_thread(
                delete_index_collection,
                self._chroma_path,
                str(record["collection_name"]),
            )
