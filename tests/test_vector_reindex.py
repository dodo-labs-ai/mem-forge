"""Changing the embedding model rebuilds the Memory vector index, and nothing moves under a rebuild."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import chromadb
import httpx
import pytest
from click.testing import CliRunner

from memforge import runtime
from memforge.config import AppConfig
from memforge.llm.bindings import embedding_profile, resolve_active_binding
from memforge.llm.providers import ModelConfigError
from memforge.memory import reindex
from memforge.memory.repair import MemoryIndexRepairer
from memforge.model_settings import CandidateDraft, ModelSettings
from memforge.retrieval.vector_index import STORED_VECTORS_COLLECTION, collection_name_for
from memforge.runtime import DefaultRuntimeProvider, SourceSyncActivityConflict, SourceSyncWorker, SyncService
from memforge.scheduler import SyncScheduler
from memforge.server.admin_api import create_admin_app
from memforge.source_activity import SourceActivityConflict, SourceActivityKind
from memforge.storage.database import Database
from memforge.vector_index_errors import ReindexInProgress, is_reindex_refusal
from tests.test_vector_index import (
    ENDPOINT,
    IDENTITY_A,
    IDENTITY_B,
    _collection_names,
    _config as _environment_config,
    _memory,
    _stored_vectors,
    _vector_hits,
    _vectors_from_before_indexes_were_recorded,
    _write_memory,
)
from tests.vector_index_fixture import EmbeddingModels

SECRET_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
KEY_A = "embedding-secret-key-aaaa"
KEY_B = "embedding-secret-key-bbbb"
API = "/api/v1/model-settings"
PROFILE_A = IDENTITY_A.profile(3)
PROFILE_B = IDENTITY_B.profile(4)
CONTENTS = {
    "mem-1": "Deploy through the release pipeline",
    "mem-2": "Rotate the signing key every quarter",
    "mem-3": "The staging cluster lives in the east region",
}


@pytest.fixture(autouse=True)
def secret_key(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", SECRET_KEY)


class Rejection(Exception):
    """A provider answer no retry changes."""

    status_code = 401


@pytest.fixture
async def db(tmp_path):
    """The database of the installation at ``tmp_path / "mem"``, which the command line opens too."""
    database = Database(AppConfig(base_dir=tmp_path / "mem").storage.db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def models(monkeypatch) -> EmbeddingModels:
    return EmbeddingModels(monkeypatch, **{"embed-a": 3, "embed-b": 4, "embed-c": 3})


@pytest.fixture
def config(tmp_path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = ""
    config.llm.embedding_model = ""
    config.llm.embedding_api_key = ""
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


@pytest.fixture
def settings(config) -> ModelSettings:
    return ModelSettings(config, DefaultRuntimeProvider())


def _draft(model: str, key: str | None) -> CandidateDraft:
    return CandidateDraft(provider="openai_compatible", endpoint=ENDPOINT, model_id=model, api_key=key)


async def _installation(db: Database, config: AppConfig, settings: ModelSettings) -> None:
    """An installation whose embedding model is embed-a, with three Memories in its index."""
    await settings.save_candidate(db, "embedding", _draft("embed-a", KEY_A))
    assert (await settings.validate_candidate(db, "embedding")).promoted is True
    for memory_id, content in CONTENTS.items():
        await _write_memory(db, config, memory_id, content)


async def _candidate_b(db: Database, settings: ModelSettings) -> None:
    """Save and validate embed-b, which waits for a rebuild."""
    await settings.save_candidate(db, "embedding", _draft("embed-b", KEY_B))
    outcome = await settings.validate_candidate(db, "embedding")
    assert (outcome.promoted, outcome.reindex_required) == (False, True)


async def _active_route(db: Database, config: AppConfig) -> str:
    return (await resolve_active_binding(db, config, "embedding")).route


def _expected_vectors(models: EmbeddingModels, route: str) -> dict[str, list[float]]:
    return {
        memory_id: models.vector(route, f"Fact: {content}\nEntities: ")
        for memory_id, content in CONTENTS.items()
    }


async def _hold_rebuild(settings: ModelSettings, db: Database):
    """Admit a rebuild and leave it running, as it is between its first and its last step."""
    indexes, target = await settings._reindex_target(db)
    return indexes, target, await indexes.start_rebuild(target)


def _expire_lease(db: Database) -> None:
    with sqlite3.connect(db.db_path) as raw:
        raw.execute("UPDATE vector_indexes SET lease_expires_epoch = ? WHERE state = 'building'", (int(time.time()) - 1,))


def _record_runner(db: Database, **changes) -> None:
    """Record the running rebuild as one that another process runs."""
    with sqlite3.connect(db.db_path) as raw:
        (stored,) = raw.execute("SELECT runner_json FROM vector_indexes WHERE state = 'building'").fetchone()
        raw.execute(
            "UPDATE vector_indexes SET runner_json = ? WHERE state = 'building'",
            (json.dumps({**json.loads(stored), **changes}),),
        )


def _segment_directories(chroma_path: str) -> list[str]:
    """The directories ChromaDB keeps vector files in, one for each collection that holds vectors."""
    return sorted(entry.name for entry in Path(chroma_path).iterdir() if entry.is_dir())


def _pid_of_an_ended_process() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def _reindex_ended(client: httpx.AsyncClient) -> dict:
    for _ in range(1000):
        status = (await client.get(f"{API}/vector-index")).json()
        if status["reindex"] is not None and status["reindex"]["state"] != "running":
            return status
        await asyncio.sleep(0.005)
    raise AssertionError("the rebuild did not end")


# -- Validate, then reindex ---------------------------------------------------------


async def test_the_first_embedding_model_of_an_installation_gets_its_index_with_its_promotion(
    db, config, settings, models
) -> None:
    await settings.save_candidate(db, "embedding", _draft("embed-a", KEY_A))

    outcome = await settings.validate_candidate(db, "embedding")

    in_use = await db.get_active_vector_index()
    assert (outcome.promoted, outcome.reindex_required) == (True, False)
    assert in_use["fingerprint"] == PROFILE_A.fingerprint
    assert (in_use["collection_name"], in_use["adopted"]) == (collection_name_for(PROFILE_A.fingerprint), False)
    assert (await settings.vector_index(db)).status == "ready"


async def test_a_candidate_in_the_vector_space_of_the_index_becomes_active_at_once(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    app = create_admin_app(db=db, config=config)

    async with _client(app) as client:
        await client.put(
            f"{API}/embedding/candidate",
            json={"provider": "openai_compatible", "endpoint": ENDPOINT, "model_id": "embed-a", "api_key": KEY_B},
        )
        validated = (await client.post(f"{API}/embedding/candidate/validate")).json()
        read = (await client.get(API)).json()

    assert (validated["passed"], validated["promoted"], validated["reindex_required"]) == (True, True, False)
    assert read["embedding"]["candidate"] is None
    assert read["embedding"]["active"]["api_key"]["last4"] == "bbbb"
    assert read["vector_index"]["status"] == "ready"
    assert read["vector_index"]["reindex"] is None
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint


async def test_a_candidate_that_writes_the_endpoint_another_way_needs_no_reindex(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    written = ENDPOINT.replace("embed.example.test", "Embed.Example.Test")

    await settings.save_candidate(
        db,
        "embedding",
        CandidateDraft(provider="openai_compatible", endpoint=written, model_id="embed-a"),
    )
    outcome = await settings.validate_candidate(db, "embedding")

    assert (outcome.promoted, outcome.reindex_required) == (True, False)
    assert (await resolve_active_binding(db, config, "embedding")).endpoint == written
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)


async def test_a_candidate_of_another_vector_space_stays_validated_and_waits_for_a_reindex(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    app = create_admin_app(db=db, config=config)

    async with _client(app) as client:
        await client.put(
            f"{API}/embedding/candidate",
            json={"provider": "openai_compatible", "endpoint": ENDPOINT, "model_id": "embed-b", "api_key": KEY_B},
        )
        validated = (await client.post(f"{API}/embedding/candidate/validate")).json()
        read = (await client.get(API)).json()
        health = (await client.get("/api/v1/health")).json()["models"]["embedding"]

    assert (validated["passed"], validated["promoted"], validated["reindex_required"]) == (True, False, True)
    candidate = read["embedding"]["candidate"]
    assert (candidate["state"], candidate["waits_for_reindex"]) == ("ready", True)
    assert candidate["profile"] == {
        "provider": "openai_compatible",
        "route": "openai/embed-b",
        "endpoint": ENDPOINT,
        "model_id": "embed-b",
        "requested_dimensions": None,
        "dimension": 4,
        "fingerprint": PROFILE_B.fingerprint,
        "thresholds_calibrated": False,
    }
    active = read["embedding"]["active"]
    assert (active["route"], active["waits_for_reindex"]) == ("openai/embed-a", False)
    assert active["profile"]["fingerprint"] == PROFILE_A.fingerprint
    assert read["vector_index"] == {
        "status": "ready",
        "reindex_required": False,
        "serves_search": True,
        "reindex_target": "candidate",
        "index": {
            "fingerprint": PROFILE_A.fingerprint,
            "collection": collection_name_for(PROFILE_A.fingerprint),
            "dimension": 3,
            "adopted": False,
            "vector_count": 3,
            "activated_at": read["vector_index"]["index"]["activated_at"],
            "profile": active["profile"],
        },
        "reindex": None,
    }
    assert health == {"configured": True, "state": "ready", "vector_index": "ready", "reindex_required": False}
    assert await _vector_hits(db, config) != []


async def test_a_reindex_builds_verifies_switches_and_deletes_the_previous_index(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config)
    app.state.memory_search_engine = stale_engine = object()

    async with _client(app) as client:
        started = await client.post(f"{API}/vector-index/reindex")
        ended = await _reindex_ended(client)
        read = (await client.get(API)).json()

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "reindexing"
    assert started.json()["reindex"]["state"] == "running"
    assert started.json()["reindex"]["target"] == "candidate"
    assert started.json()["reindex_target"] is None
    assert ended["status"] == "ready"
    assert ended["reindex"]["state"] == "succeeded"
    assert ended["reindex"]["error_code"] is None
    assert (ended["reindex"]["memories_total"], ended["reindex"]["memories_indexed"]) == (3, 3)
    assert ended["reindex"]["fingerprint"] == ended["index"]["fingerprint"] == PROFILE_B.fingerprint
    assert (ended["index"]["collection"], ended["index"]["adopted"]) == (collection_name_for(PROFILE_B.fingerprint), False)
    assert ended["index"]["vector_count"] == 3
    assert read["embedding"]["candidate"] is None
    assert read["embedding"]["active"]["route"] == "openai/embed-b"
    assert read["embedding"]["active"]["state"] == "ready"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_B.fingerprint)]
    assert _stored_vectors(config.storage.chroma_path, collection_name_for(PROFILE_B.fingerprint)) == (
        _expected_vectors(models, "openai/embed-b")
    )
    assert app.state.memory_search_engine is not stale_engine
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT state FROM vector_indexes").fetchall() == [("active",)]


async def test_a_switch_stands_when_the_previous_index_cannot_be_deleted(
    db, config, settings, models, monkeypatch
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    indexes, target, record = await _hold_rebuild(settings, db)
    previous = collection_name_for(PROFILE_A.fingerprint)
    delete = reindex.delete_index_collection
    switched: list[bool] = []

    def busy(chroma_path: str, name: str) -> None:
        if name == previous:
            raise RuntimeError("the vector store is busy")
        delete(chroma_path, name)

    monkeypatch.setattr(reindex, "delete_index_collection", busy)
    await settings._run_reindex(indexes, record, target, on_switch=lambda: switched.append(True))

    assert switched == [True]
    assert (await db.get_vector_index(record["id"]))["state"] == "active"
    assert await _active_route(db, config) == "openai/embed-b"
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)
    assert previous in _collection_names(config.storage.chroma_path)

    monkeypatch.setattr(reindex, "delete_index_collection", delete)
    await settings.prepare_vector_index(db)

    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_B.fingerprint)]
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT state FROM vector_indexes").fetchall() == [("active",)]


async def test_search_reads_the_previous_index_until_the_switch(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    engine = await runtime.build_search_engine(db, config)
    during: dict[str, object] = {}

    async def search_while_building(request: dict) -> None:
        if request["model"] != "openai/embed-b" or during:
            return
        hits = await engine.search("unrelated words only", top_k=5)
        during["hits"] = sorted(hit.memory_id for hit in hits["results"])
        during["status"] = (await settings.vector_index(db)).status
        during["in_use"] = (await db.get_active_vector_index())["fingerprint"]
        during["route"] = await _active_route(db, config)

    models.before_answer = search_while_building

    finished = await settings.run_reindex(db)

    assert finished["state"] == "active"
    assert during == {
        "hits": sorted(CONTENTS),
        "status": "reindexing",
        "in_use": PROFILE_A.fingerprint,
        "route": "openai/embed-a",
    }
    assert (await engine.search("unrelated words only", top_k=5))["results"] == []
    assert engine.serves_another_index is True
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)


async def test_the_binding_in_use_and_the_index_in_use_change_in_one_transaction(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    observed: list[tuple[str, str]] = []
    commit = db.db.commit

    async def observing_commit() -> None:
        await commit()
        with sqlite3.connect(db.db_path) as raw:
            route = raw.execute("SELECT route FROM model_bindings WHERE kind = 'embedding' AND slot = 'active'").fetchone()
            index = raw.execute("SELECT fingerprint FROM vector_indexes WHERE state = 'active'").fetchall()
        assert len(index) == 1
        observed.append((route[0], index[0][0]))

    db.db.commit = observing_commit
    try:
        await settings.run_reindex(db)
    finally:
        db.db.commit = commit

    assert set(observed) == {
        ("openai/embed-a", PROFILE_A.fingerprint),
        ("openai/embed-b", PROFILE_B.fingerprint),
    }
    active = await resolve_active_binding(db, config, "embedding")
    assert embedding_profile(active).fingerprint == (await db.get_active_vector_index())["fingerprint"]


async def test_a_switch_that_fails_inside_its_transaction_changes_nothing(db, config, settings, models, monkeypatch) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    promote = Database._promote_model_candidate_unlocked

    async def promote_then_fail(self, *args, **kwargs):
        await promote(self, *args, **kwargs)
        raise RuntimeError("the database stopped answering")

    monkeypatch.setattr(Database, "_promote_model_candidate_unlocked", promote_then_fail)
    failed = await settings.run_reindex(db)
    monkeypatch.setattr(Database, "_promote_model_candidate_unlocked", promote)

    assert (failed["state"], failed["error_code"]) == ("failed", "index_error")
    assert await _active_route(db, config) == "openai/embed-a"
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint
    assert (await settings.view(db, "embedding")).candidate.waits_for_reindex is True
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)
    assert (await settings.run_reindex(db))["state"] == "active"
    assert await _active_route(db, config) == "openai/embed-b"


@pytest.mark.parametrize(
    ("vector_count", "promotion", "refusal"),
    [
        (2, None, "memories_changed"),
        (3, {"updated_at": "2020-01-01T00:00:00+00:00", "state": "ready", "validation": {}}, "candidate_changed"),
    ],
    ids=["memories_changed", "candidate_saved_again"],
)
async def test_a_switch_is_refused_when_what_the_index_was_built_for_has_changed(
    db, config, settings, models, vector_count, promotion, refusal
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    _indexes, target, record = await _hold_rebuild(settings, db)

    refused = await db.activate_vector_index(
        record["id"],
        vector_count=vector_count,
        promote_candidate=promotion or target.promotion,
    )

    assert refused == refusal
    assert await _active_route(db, config) == "openai/embed-a"
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint
    assert (await db.get_vector_index(record["id"]))["state"] == "building"


async def test_replacing_the_key_of_a_validated_candidate_makes_it_need_revalidation(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config)
    candidate_b = {"provider": "openai_compatible", "endpoint": ENDPOINT, "model_id": "embed-b"}

    async with _client(app) as client:
        kept = (await client.put(f"{API}/embedding/candidate", json=candidate_b)).json()["candidate"]
        replaced = (
            await client.put(f"{API}/embedding/candidate", json={**candidate_b, "api_key": KEY_A})
        ).json()["candidate"]
        status = (await client.get(f"{API}/vector-index")).json()
        reindex_refused = await client.post(f"{API}/vector-index/reindex")
        validated = (await client.post(f"{API}/embedding/candidate/validate")).json()

    assert (kept["state"], kept["validation"]["current"], kept["waits_for_reindex"]) == ("ready", True, True)
    assert (replaced["state"], replaced["validation"]["current"]) == ("needs_revalidation", False)
    assert (replaced["waits_for_reindex"], replaced["profile"]) == (False, None)
    assert replaced["api_key"]["last4"] == "aaaa"
    assert status["reindex_target"] is None
    assert reindex_refused.status_code == 409
    assert reindex_refused.json()["detail"]["error"] == "reindex_not_required"
    assert (validated["passed"], validated["promoted"], validated["reindex_required"]) == (True, False, True)
    assert models.requests[-1]["api_key"] == KEY_A


async def test_removing_the_key_of_a_validated_candidate_makes_it_need_revalidation(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)

    view = await settings.save_candidate(
        db,
        "embedding",
        CandidateDraft(provider="openai_compatible", endpoint=ENDPOINT, model_id="embed-b", auth_strategy="none"),
    )

    assert view.candidate.binding.state == "needs_revalidation"
    assert view.candidate.key.is_set is False
    assert view.candidate.waits_for_reindex is False
    assert view.active.binding.state == "ready"


# -- A failed or interrupted reindex ------------------------------------------------


async def test_a_reindex_that_fails_midway_changes_nothing_in_use_and_can_be_run_again(
    db, config, settings, models, monkeypatch
) -> None:
    monkeypatch.setattr(reindex, "_BATCH_SIZE", 2)
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    engine = await runtime.build_search_engine(db, config)
    rebuild_requests: list[dict] = []
    files_before = _segment_directories(config.storage.chroma_path)

    async def fail_the_second_batch(request: dict) -> None:
        if request["model"] != "openai/embed-b":
            return
        rebuild_requests.append(request)
        if len(rebuild_requests) == 2:
            raise Rejection("the key was revoked")

    models.before_answer = fail_the_second_batch

    failed = await settings.run_reindex(db)
    view = await settings.vector_index(db)

    assert len(rebuild_requests) == 2
    assert (failed["state"], failed["error_code"]) == ("failed", "authentication_failed")
    assert (failed["memory_count"], failed["vector_count"]) == (3, 2)
    assert (view.status, view.reindex_target, view.reindex_running) == ("ready", "candidate", False)
    assert view.reindex["id"] == failed["id"]
    assert view.in_use["fingerprint"] == PROFILE_A.fingerprint
    assert await _active_route(db, config) == "openai/embed-a"
    assert (await settings.view(db, "embedding")).candidate.binding.route == "openai/embed-b"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]
    # The failed rebuild leaves no vector files of its own on disk either.
    assert _segment_directories(config.storage.chroma_path) == files_before != []
    assert _stored_vectors(config.storage.chroma_path, collection_name_for(PROFILE_A.fingerprint)) == (
        _expected_vectors(models, "openai/embed-a")
    )
    hits = await engine.search("unrelated words only", top_k=5)
    assert sorted(hit.memory_id for hit in hits["results"]) == sorted(CONTENTS)
    await _write_memory(db, config, "mem-after-failure", "A fact written after the failed rebuild")

    models.before_answer = None
    rerun = await settings.run_reindex(db)

    assert rerun["state"] == "active"
    assert rerun["id"] != failed["id"]
    assert rerun["vector_count"] == 4
    assert await _active_route(db, config) == "openai/embed-b"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_B.fingerprint)]
    # The index that was replaced went with its files: one index, one directory of vector files.
    files_after = _segment_directories(config.storage.chroma_path)
    assert len(files_after) == 1 and files_after != files_before
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT state FROM vector_indexes").fetchall() == [("active",)]


async def test_a_reindex_whose_process_died_is_interrupted_and_can_be_run_again(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    _indexes, _target, record = await _hold_rebuild(settings, db)
    app = create_admin_app(db=db, config=config)

    _expire_lease(db)
    async with _client(app) as client:
        status = (await client.get(f"{API}/vector-index")).json()

    assert (status["status"], status["reindex_target"]) == ("ready", "candidate")
    assert (status["reindex"]["state"], status["reindex"]["error_code"]) == ("failed", "interrupted")
    assert await db.vector_reindex_running() is False
    await _write_memory(db, config, "mem-after-crash", "A fact written after the crash")
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint

    rerun = await settings.run_reindex(db)

    assert rerun["state"] == "active"
    assert rerun["vector_count"] == 4
    assert (await db.get_vector_index(record["id"])) is None
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_B.fingerprint)]


@pytest.mark.skipif(os.name != "posix", reason="whether a process runs can be asked on POSIX only")
@pytest.mark.parametrize("runner", ["an_ended_process", "this_process_id_reused"])
async def test_a_starting_server_ends_the_reindex_of_a_process_that_was_killed_on_its_host(
    db, config, settings, models, runner
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    _indexes, _target, record = await _hold_rebuild(settings, db)
    if runner == "an_ended_process":
        _record_runner(db, pid=_pid_of_an_ended_process())
    else:
        _record_runner(db, token="the-token-of-an-earlier-process")
    restarted = ModelSettings(config, DefaultRuntimeProvider())

    assert await db.vector_reindex_running() is True
    await restarted.prepare_vector_index(db)
    view = await restarted.vector_index(db)

    assert await db.vector_reindex_running() is False
    assert (view.status, view.reindex_target, view.reindex_running) == ("ready", "candidate", False)
    assert (view.reindex["id"], view.reindex["state"], view.reindex["error_code"]) == (
        record["id"],
        "failed",
        "interrupted",
    )
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]
    await _write_memory(db, config, "mem-after-kill", "A fact written after the server was killed")
    assert (await restarted.run_reindex(db))["state"] == "active"


@pytest.mark.parametrize(
    "runner",
    [{}, {"host": "another-host", "pid": 1}],
    ids=["this_process", "a_process_on_another_host"],
)
async def test_a_reindex_whose_process_may_be_running_holds_until_its_lease_runs_out(
    db, config, settings, models, runner
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    indexes, _target, record = await _hold_rebuild(settings, db)
    _record_runner(db, **runner)

    await indexes.clean_up()
    held = await db.vector_reindex_running()
    _expire_lease(db)
    await indexes.clean_up()

    assert held is True
    assert record["runner"] == {"host": socket.gethostname(), "pid": os.getpid(), "token": record["runner"]["token"]}
    assert ((await db.get_vector_index(record["id"]))["state"], await db.vector_reindex_running()) == ("failed", False)


async def test_a_reindex_keeps_its_lease_while_one_batch_takes_long(db, config, settings, models, monkeypatch) -> None:
    monkeypatch.setattr(reindex, "_LEASE_RENEWAL_SECONDS", 0.001)
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    reached = asyncio.Event()
    renewed_in_the_batch = asyncio.Event()
    renewals: list[dict] = []
    renew = db.renew_vector_reindex

    async def record_renewal(index_id: str, **reported) -> bool:
        renewals.append(reported)
        if reached.is_set() and len(renewals) >= at_the_batch + 3:
            renewed_in_the_batch.set()
        return await renew(index_id, **reported)

    async def wait_for_renewals(request: dict) -> None:
        nonlocal at_the_batch
        if request["model"] == "openai/embed-b" and not reached.is_set():
            at_the_batch = len(renewals)
            reached.set()
            await renewed_in_the_batch.wait()

    at_the_batch = 0
    monkeypatch.setattr(db, "renew_vector_reindex", record_renewal)
    models.before_answer = wait_for_renewals

    finished = await asyncio.wait_for(settings.run_reindex(db), timeout=30)

    in_the_batch = renewals[at_the_batch : at_the_batch + 3]
    assert finished["state"] == "active"
    # The timer renews the lease and reports no progress, which only a finished batch does.
    assert all(set(renewal) == {"lease_seconds"} for renewal in in_the_batch)
    assert reindex.REINDEX_LEASE_SECONDS <= 60


@pytest.mark.skipif(
    os.name != "posix" or threading.current_thread() is not threading.main_thread(),
    reason="a signal reaches the main thread of a POSIX process only",
)
async def test_a_reindex_command_asked_to_terminate_records_the_reindex_as_interrupted(
    db, config, settings, models
) -> None:
    from memforge.main import _cancelling_on_sigterm

    await _installation(db, config, settings)
    await _candidate_b(db, settings)

    async def terminate_in_the_rebuild(request: dict) -> None:
        if request["model"] == "openai/embed-b":
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(30)

    models.before_answer = terminate_in_the_rebuild

    async def command() -> bool:
        with _cancelling_on_sigterm() as termination:
            try:
                await settings.run_reindex(db)
            except asyncio.CancelledError:
                return termination.requested
        return False

    terminated = await asyncio.wait_for(asyncio.create_task(command()), timeout=20)

    latest = await db.get_latest_vector_reindex()
    assert terminated is True
    assert (latest["state"], latest["error_code"]) == ("failed", "interrupted")
    assert await db.vector_reindex_running() is False
    assert await _active_route(db, config) == "openai/embed-a"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


async def test_a_server_that_stops_records_its_running_reindex_as_interrupted(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    reached = asyncio.Event()
    release = asyncio.Event()

    async def wait_in_the_rebuild(request: dict) -> None:
        if request["model"] == "openai/embed-b":
            reached.set()
            await release.wait()

    models.before_answer = wait_in_the_rebuild

    started = await settings.start_reindex(db)
    await reached.wait()
    await settings.shutdown()

    latest = await db.get_latest_vector_reindex()
    assert started.status == "reindexing"
    assert (latest["state"], latest["error_code"]) == ("failed", "interrupted")
    assert await db.vector_reindex_running() is False
    assert await _active_route(db, config) == "openai/embed-a"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]


@pytest.mark.parametrize(
    ("counted", "error_code"),
    [
        ((2, 2), "vector_count_mismatch"),
        ((4, 4), "vector_count_mismatch"),
        ((3, 2), "vector_invalid"),
    ],
    ids=["missing_vector", "extra_vector", "invalid_vector"],
)
async def test_an_index_that_does_not_verify_is_not_put_into_use(
    db, config, settings, models, monkeypatch, counted, error_code
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    monkeypatch.setattr(reindex, "count_valid_vectors", lambda _collection, *, dimension: counted)

    failed = await settings.run_reindex(db)

    assert (failed["state"], failed["error_code"]) == ("failed", error_code)
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint
    assert await _active_route(db, config) == "openai/embed-a"
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]


async def test_a_model_that_answers_with_another_length_than_it_was_validated_with_fails_the_reindex(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    models.dimensions["embed-b"] = 5

    failed = await settings.run_reindex(db)

    assert (failed["state"], failed["error_code"]) == ("failed", "embedding_dimension_mismatch")
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint


# -- The guard: nothing moves under a reindex ----------------------------------------


async def test_a_memory_write_during_a_reindex_is_refused_with_a_stable_code(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config, principal_resolver=lambda _request: "author@example.test")
    indexes, target, record = await _hold_rebuild(settings, db)
    created = {
        "content": "A fact written while the index is rebuilt.",
        "provenance": "Confirmed by the release owner during the rollout review.",
        "memory_type": "fact",
        "client": "codex",
    }

    async with _client(app) as client:
        create = await client.post("/api/v1/memories/create", json=created)
        retire = await client.post("/api/v1/memories/mem-1/retire", json={"reason": "obsolete"})
        decide = await client.post("/api/v1/memory-reviews/review-1/approve", json={})
        search = await client.post("/api/v1/memories/search", json={"query": "release pipeline"})
    with pytest.raises(sqlite3.IntegrityError) as insert:
        await db.insert_memory(_memory("mem-during", "A row written past the API"))
    await db.db.rollback()
    with pytest.raises(sqlite3.IntegrityError) as retired:
        await db.db.execute("UPDATE memories SET status = 'retired' WHERE id = 'mem-1'")
    await db.db.rollback()
    with pytest.raises(sqlite3.IntegrityError) as deleted:
        await db.db.execute("DELETE FROM memories WHERE id = 'mem-1'")
    await db.db.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        await _write_memory(db, config, "mem-sync-like", "A write through the Memory store")

    for refused in (create, retire, decide):
        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"]["error"] == "reindex_in_progress"
    assert search.status_code == 200, search.text
    assert all(is_reindex_refusal(error.value) for error in (insert, retired, deleted))
    assert str(insert.value) == "reindex_in_progress"
    assert await db.count_searchable_memories() == 3

    finished = await indexes.rebuild(record, target)
    async with _client(app) as client:
        after = await client.post("/api/v1/memories/create", json=created)

    assert finished["state"] == "active"
    assert after.status_code == 200, after.text
    assert await db.count_searchable_memories() == 4


async def test_a_write_the_database_refuses_past_the_request_check_is_answered_with_the_code(
    db, config, settings, models, monkeypatch
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config, principal_resolver=lambda _request: "author@example.test")
    await _hold_rebuild(settings, db)

    async def not_running(_db) -> bool:
        return False

    monkeypatch.setattr(app.state.model_settings, "reindex_running", not_running)
    async with _client(app) as client:
        response = await client.post(
            "/api/v1/memories/create",
            json={
                "content": "A fact that slips past the request check.",
                "provenance": "Confirmed by the release owner during the rollout review.",
                "memory_type": "fact",
                "client": "codex",
            },
        )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error"] == "reindex_in_progress"
    assert await db.count_searchable_memories() == 3


async def _refused_writes(db: Database) -> dict:
    """Every Memory write that keeps the write lock for one transaction of its own."""
    held = await db.get_memory("mem-held")
    return {
        "content": lambda: db.update_memory_content("mem-held", "A fact that was rewritten"),
        "status": lambda: db.update_memory_status("mem-held", "retired", reason="obsolete"),
        "purge": lambda: db.purge_memory("mem-1"),
        "snapshot": lambda: db.restore_memory_snapshot(held, search_visible_statuses={"active"}),
        "promotion": lambda: db.promote_quarantined_challenger(
            incumbent_id="mem-held",
            challenger=held,
            replacement_kind="revision",
        ),
        "expiry": db.retire_expired_memories,
        "support": lambda: db.refresh_memory_support_state("mem-held"),
    }


@pytest.mark.parametrize("write", ["content", "status", "purge", "snapshot", "promotion", "expiry", "support"])
async def test_a_refused_memory_write_leaves_nothing_behind_on_its_connection(
    db, config, settings, models, write
) -> None:
    await _installation(db, config, settings)
    await db.insert_memory(_memory("mem-held", "A fact no Source supports"))
    with sqlite3.connect(db.db_path) as raw:
        raw.execute("UPDATE memories SET valid_until = '2020-01-01' WHERE id = 'mem-held'")
        before = raw.execute("SELECT COUNT(*) FROM memory_sources").fetchone()
    await _candidate_b(db, settings)
    indexes, target, record = await _hold_rebuild(settings, db)

    with pytest.raises(sqlite3.IntegrityError) as refused:
        await (await _refused_writes(db))[write]()

    assert is_reindex_refusal(refused.value)
    assert db.db.in_transaction is False
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM memory_sources").fetchone() == before
        assert raw.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone() == (4,)
    assert (await indexes.rebuild(record, target))["state"] == "active"


async def test_a_write_refused_on_another_connection_does_not_hold_the_reindex_up(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await db.insert_memory(_memory("mem-held", "A fact no Source supports"))
    await _candidate_b(db, settings)
    server = Database(db.db_path)
    await server.connect()
    try:
        indexes, target, record = await _hold_rebuild(settings, db)
        with pytest.raises(sqlite3.IntegrityError):
            await server.update_memory_status("mem-held", "retired", reason="expired")
        finished = await indexes.rebuild(record, target)
    finally:
        await server.close()

    assert (finished["state"], finished["vector_count"]) == ("active", 4)


async def test_the_expiry_job_leaves_expired_memories_to_its_next_run_during_a_reindex(
    db, config, settings, models, caplog
) -> None:
    await _installation(db, config, settings)
    await db.insert_memory(_memory("mem-held", "A fact no Source supports"))
    with sqlite3.connect(db.db_path) as raw:
        raw.execute("UPDATE memories SET valid_until = '2020-01-01' WHERE id = 'mem-held'")
    await _candidate_b(db, settings)
    scheduler = SyncScheduler(db, SyncService(db, config))
    indexes, target, record = await _hold_rebuild(settings, db)

    with caplog.at_level("INFO", logger="memforge.scheduler"):
        await scheduler._retire_expired_memories()
    finished = await indexes.rebuild(record, target)

    assert "Expired memories are retired by the next run" in caplog.text
    assert (await db.get_memory("mem-held")).status == "active"
    assert finished["state"] == "active"

    await scheduler._retire_expired_memories()

    assert (await db.get_memory("mem-held")).status == "retired"


async def test_the_expiry_job_fails_for_any_other_reason(db, config, monkeypatch) -> None:
    service = SyncService(db, config)

    async def broken() -> int:
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(service, "retire_expired_memories", broken)

    with pytest.raises(sqlite3.IntegrityError):
        await SyncScheduler(db, service)._retire_expired_memories()


async def test_a_write_that_does_not_touch_what_vectors_are_made_from_is_not_refused(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    await _hold_rebuild(settings, db)

    await db.db.execute("UPDATE memories SET corroboration_count = corroboration_count + 1 WHERE id = 'mem-1'")
    await db.db.commit()

    assert (await db.get_memory("mem-1")).corroboration_count == 2


async def test_a_source_sync_does_not_start_during_a_reindex(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    run = await db.enqueue_source_sync_run(source_id="src-index", trigger="manual")
    indexes, target, record = await _hold_rebuild(settings, db)

    leased_during = await db.lease_next_source_sync_run(worker_id="worker")
    with pytest.raises(SourceActivityConflict, match="reindex_in_progress"):
        await db.acquire_source_activity(
            activity_id="sync-during-reindex",
            source_id="src-index",
            kind=SourceActivityKind.SYNC,
        )
    with pytest.raises(SourceSyncActivityConflict, match="reindex_in_progress"):
        await runtime.run_source_sync(db, config, await db.get_source("src-index"))
    with sqlite3.connect(db.db_path) as raw:
        pending = raw.execute("SELECT status FROM source_sync_runs WHERE run_id = ?", (run.run_id,)).fetchone()
        leases = raw.execute("SELECT COUNT(*) FROM source_activity_leases").fetchone()

    assert leased_during is None
    assert pending == ("pending",)
    assert leases == (0,)

    await indexes.rebuild(record, target)
    leased_after = await db.lease_next_source_sync_run(worker_id="worker")

    assert leased_after is not None and leased_after.run_id == run.run_id


@pytest.mark.parametrize("kind", [SourceActivityKind.SYNC, SourceActivityKind.AGENT_PATCH, SourceActivityKind.MAINTENANCE])
async def test_a_reindex_does_not_start_while_a_source_is_syncing(db, config, settings, models, kind) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config)
    await db.acquire_source_activity(activity_id="running-sync", source_id="src-index", kind=kind)

    with pytest.raises(ModelConfigError) as refused:
        await settings.run_reindex(db)
    async with _client(app) as client:
        response = await client.post(f"{API}/vector-index/reindex")

    assert refused.value.code == "sync_in_progress"
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "sync_in_progress"
    assert await db.get_latest_vector_reindex() is None
    assert await db.vector_reindex_running() is False

    await db.release_source_activity(activity_id="running-sync")
    assert (await settings.run_reindex(db))["state"] == "active"


async def test_a_sync_whose_lease_has_run_out_does_not_hold_a_reindex_back(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    await db.acquire_source_activity(
        activity_id="abandoned-sync",
        source_id="src-index",
        kind=SourceActivityKind.SYNC,
        lease_seconds=1,
    )
    with sqlite3.connect(db.db_path) as raw:
        raw.execute("UPDATE source_activity_leases SET lease_until = '2020-01-01T00:00:00+00:00'")

    assert (await settings.run_reindex(db))["state"] == "active"


async def test_only_one_reindex_runs_at_a_time(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config)
    _indexes, _target, record = await _hold_rebuild(settings, db)

    async with _client(app) as client:
        second = await client.post(f"{API}/vector-index/reindex")

    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "reindex_in_progress"
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT id FROM vector_indexes WHERE state = 'building'").fetchall() == [(record["id"],)]


async def test_the_embedding_candidate_cannot_change_during_a_reindex(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    app = create_admin_app(db=db, config=config)
    await _hold_rebuild(settings, db)

    async with _client(app) as client:
        saved = await client.put(
            f"{API}/embedding/candidate",
            json={"provider": "openai_compatible", "endpoint": ENDPOINT, "model_id": "embed-c", "api_key": KEY_B},
        )
        validated = await client.post(f"{API}/embedding/candidate/validate")
        discarded = await client.delete(f"{API}/embedding/candidate")
        generation = await client.delete(f"{API}/generation/candidate")

    for refused in (saved, validated, discarded):
        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"]["error"] == "reindex_in_progress"
    assert generation.status_code == 200
    assert (await settings.view(db, "embedding")).candidate.binding.route == "openai/embed-b"


async def test_index_repair_does_not_run_during_a_reindex(db, config, settings, models, monkeypatch) -> None:
    from memforge.main import cli

    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    indexes, _target, _record = await _hold_rebuild(settings, db)
    index = await indexes.in_use()
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(config.base_dir))
    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "")

    with pytest.raises(ReindexInProgress):
        await MemoryIndexRepairer(
            db=db,
            memory_collection=index.collection,
            embedding=await runtime.build_active_embedding_runtime(db, config, DefaultRuntimeProvider()),
            index_fingerprint=index.fingerprint,
        ).repair()
    result = await asyncio.to_thread(CliRunner().invoke, cli, ["maintenance", "repair-indexes"])

    assert result.exit_code == 1, result.output
    assert "being rebuilt" in result.output


async def test_the_workers_idle_work_waits_during_a_reindex(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    worker = SourceSyncWorker(db, config)
    slices: list[str] = []

    async def lifecycle() -> bool:
        slices.append("lifecycle")
        return False

    async def review() -> bool:
        slices.append("review")
        return False

    async def relations() -> bool:
        slices.append("relations")
        return False

    worker.process_lifecycle_vector_delivery_once = lifecycle
    worker.process_review_vector_delivery_once = review
    worker._process_relation_discovery_once = relations
    indexes, target, record = await _hold_rebuild(settings, db)

    await worker.process_bounded_maintenance_once()
    during = list(slices)
    await indexes.rebuild(record, target)
    await worker.process_bounded_maintenance_once()

    assert during == []
    assert slices == ["lifecycle", "review", "relations"]


# -- Which binding a reindex is for ---------------------------------------------------


async def test_a_reindex_is_refused_when_the_index_already_fits_the_model(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    app = create_admin_app(db=db, config=config)

    async with _client(app) as client:
        response = await client.post(f"{API}/vector-index/reindex")
        status = (await client.get(f"{API}/vector-index")).json()

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "reindex_not_required"
    assert (status["status"], status["reindex_target"], status["reindex"]) == ("ready", None, None)


async def test_a_reindex_decided_on_before_another_one_switched_is_refused(db, config, settings, models) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    indexes, target = await settings._reindex_target(db)
    assert (await settings.run_reindex(db))["state"] == "active"
    in_use = await db.get_active_vector_index()

    with pytest.raises(ModelConfigError) as refused:
        await settings._admit_reindex(indexes, target)

    assert refused.value.code == "reindex_not_required"
    assert await db.get_active_vector_index() == in_use
    assert await db.list_vector_indexes("building") == []
    assert _stored_vectors(config.storage.chroma_path, in_use["collection_name"]) == (
        _expected_vectors(models, "openai/embed-b")
    )
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)


async def test_a_reindex_never_empties_the_collection_of_the_index_in_use(db, tmp_path, models) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=8)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())
    indexes, target, record = await _hold_rebuild(settings, db)
    # Another process records its index in the collection this rebuild is about to fill.
    held = chromadb.PersistentClient(path=config.storage.chroma_path).get_or_create_collection(
        name=record["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )
    held.upsert(ids=["mem-in-use"], embeddings=[[1.0, 2.0, 3.0]])
    assert await db.record_active_vector_index(
        {
            "id": "index-in-use",
            "fingerprint": PROFILE_A.fingerprint,
            "collection_name": record["collection_name"],
            "adopted": False,
            "dimension": 3,
            "profile": PROFILE_A.fields(),
            "vector_count": 1,
        }
    )

    finished = await indexes.rebuild(record, target)

    assert (finished["state"], finished["error_code"]) == ("failed", "reindex_not_required")
    assert (await db.get_active_vector_index())["id"] == "index-in-use"
    assert _stored_vectors(config.storage.chroma_path, record["collection_name"]) == {"mem-in-use": [1.0, 2.0, 3.0]}


async def test_a_reindex_without_an_embedding_model_is_refused(db, config, settings, models) -> None:
    app = create_admin_app(db=db, config=config)

    async with _client(app) as client:
        response = await client.post(f"{API}/vector-index/reindex")
        status = (await client.get(f"{API}/vector-index")).json()

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "embedding_not_validated"
    assert status == {
        "status": "not_configured",
        "reindex_required": False,
        "serves_search": False,
        "reindex_target": None,
        "index": None,
        "reindex": None,
    }


async def test_a_changed_environment_model_is_reindexed_as_the_model_in_use(db, tmp_path, models) -> None:
    config = _environment_config(tmp_path)
    settings = ModelSettings(config, DefaultRuntimeProvider())
    for memory_id, content in CONTENTS.items():
        await _write_memory(db, config, memory_id, content)
    config.llm.embedding_model = "embed-b"
    assert await _vector_hits(db, config) == []

    finished = await settings.run_reindex(db)
    view = await settings.vector_index(db)

    assert (finished["state"], finished["binding_slot"]) == ("active", "active")
    assert (view.status, view.reindex_target) == ("ready", None)
    assert view.in_use["fingerprint"] == PROFILE_B.fingerprint
    assert sorted(await _vector_hits(db, config)) == sorted(CONTENTS)
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM model_bindings").fetchone() == (0,)


async def test_a_reindex_started_while_vector_search_is_off_leaves_it_off_until_the_switch(
    db, tmp_path, models
) -> None:
    config = _environment_config(tmp_path)
    settings = ModelSettings(config, DefaultRuntimeProvider())
    for memory_id, content in CONTENTS.items():
        await _write_memory(db, config, memory_id, content)
    config.llm.embedding_model = "embed-b"
    required = await settings.vector_index(db)
    during: list = []

    async def read_the_index(request: dict) -> None:
        if request["model"] == "openai/embed-b" and len(request["input"]) == len(CONTENTS):
            during.append(await settings.vector_index(db))
            during.append(await _vector_hits(db, config))

    models.before_answer = read_the_index
    finished = await settings.run_reindex(db)
    after = await settings.vector_index(db)

    assert (required.status, required.serves_search) == ("reindex_required", False)
    assert (during[0].status, during[0].serves_search, during[0].reindex_target) == ("reindexing", False, None)
    assert during[1] == []
    assert finished["state"] == "active"
    assert (after.status, after.serves_search) == ("ready", True)


async def test_an_installation_whose_stored_vectors_have_another_length_is_reindexed_from_its_memories(
    db, tmp_path, models
) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=8)
    await db.upsert_source("src-index", "confluence", "Index", "{}", access_policy="workspace", owner_user_id="dev")
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())

    finished = await settings.run_reindex(db)

    assert (finished["state"], finished["adopted"], finished["vector_count"]) == ("active", False, 1)
    assert (await settings.vector_index(db)).status == "ready"
    assert await _vector_hits(db, config) == ["mem-stored"]
    # The vectors no index adopted went with the rebuild that replaced them.
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]


async def test_stored_vectors_stay_while_no_index_is_in_use(db, tmp_path, models) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=8)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())
    await settings.validate_active(db, "embedding")
    models.failure = Rejection("the key was revoked")

    failed = await settings.run_reindex(db)
    await settings.prepare_vector_index(db)

    assert (failed["state"], failed["error_code"]) == ("failed", "authentication_failed")
    assert await db.get_active_vector_index() is None
    assert _collection_names(config.storage.chroma_path) == [STORED_VECTORS_COLLECTION]
    assert _stored_vectors(config.storage.chroma_path, STORED_VECTORS_COLLECTION) == {"mem-stored": [1.0] * 8}


async def test_an_adopted_index_can_be_rebuilt_for_the_model_in_use(db, tmp_path, models) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=3)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))
    settings = ModelSettings(config, DefaultRuntimeProvider())
    await settings.validate_active(db, "embedding")
    adopted = await settings.vector_index(db)

    finished = await settings.run_reindex(db)
    rebuilt = await settings.vector_index(db)

    assert (adopted.status, adopted.reindex_target, adopted.in_use["adopted"]) == ("ready", "active", True)
    assert adopted.in_use["collection_name"] == STORED_VECTORS_COLLECTION
    assert finished["state"] == "active"
    assert (rebuilt.status, rebuilt.reindex_target, rebuilt.in_use["adopted"]) == ("ready", None, False)
    assert rebuilt.in_use["fingerprint"] == adopted.in_use["fingerprint"] == PROFILE_A.fingerprint
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_A.fingerprint)]
    assert _stored_vectors(config.storage.chroma_path, collection_name_for(PROFILE_A.fingerprint)) == {
        "mem-stored": models.vector("openai/embed-a", "Fact: A fact from before the index was recorded\nEntities: "),
    }


async def test_stored_vectors_are_not_adopted_for_a_candidate_of_another_model(db, tmp_path, models) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=3)
    settings = ModelSettings(config, DefaultRuntimeProvider())

    await settings.save_candidate(db, "embedding", _draft("embed-c", KEY_B))
    another_model = await settings.validate_candidate(db, "embedding")
    await settings.save_candidate(db, "embedding", _draft("embed-a", KEY_A))
    same_model = await settings.validate_candidate(db, "embedding")

    assert (another_model.promoted, another_model.reindex_required) == (False, True)
    assert (same_model.promoted, same_model.reindex_required) == (True, False)
    in_use = await db.get_active_vector_index()
    assert (in_use["collection_name"], in_use["adopted"]) == (STORED_VECTORS_COLLECTION, True)
    assert in_use["fingerprint"] == PROFILE_A.fingerprint


class _ProviderWithoutIndexManagement(DefaultRuntimeProvider):
    memory_vector_indexes = None


class _ProviderThatKeepsThisIndexItself(DefaultRuntimeProvider):
    def memory_vector_indexes(self, db, config):
        return None


@pytest.mark.parametrize(
    "provider",
    [_ProviderWithoutIndexManagement(), _ProviderThatKeepsThisIndexItself()],
    ids=["no_index_management", "none_for_this_database"],
)
async def test_a_deployment_that_keeps_its_own_index_is_not_reindexed_from_here(
    db, config, models, provider
) -> None:
    app = create_admin_app(db=db, config=config, runtime_provider=provider)

    async with _client(app) as client:
        status = (await client.get(f"{API}/vector-index")).json()
        response = await client.post(f"{API}/vector-index/reindex")
        health = (await client.get("/api/v1/health")).json()["models"]["embedding"]

    assert status["status"] == "not_managed"
    assert response.status_code == 405
    assert response.json()["detail"]["error"] == "reindex_not_available"
    assert (health["vector_index"], health["reindex_required"]) == ("not_managed", False)
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM vector_indexes").fetchone() == (0,)


def test_the_default_provider_manages_the_index_of_a_sqlite_database_only(db, config) -> None:
    provider = DefaultRuntimeProvider()

    class AnotherStore:
        """A workspace store that is not the SQLite database."""

    assert provider.memory_vector_indexes(db, config) is not None
    assert provider.memory_vector_indexes(AnotherStore(), config) is None
    assert runtime.memory_vector_indexes(provider, AnotherStore(), config) is None
    assert runtime.first_index_settler(provider, AnotherStore(), config) is None


# -- A server that learns of a reindex another process made ---------------------------


async def test_the_workers_vector_runtime_is_built_again_after_another_process_reindexed(
    db, config, settings, models
) -> None:
    await _installation(db, config, settings)
    worker = SourceSyncWorker(db, config)
    worker._vector_runtime = stale = await worker._build_runtime()
    await _candidate_b(db, settings)
    await settings.run_reindex(db)

    class Waiting:
        attempted_tasks = 0
        delivered_tasks = 0
        failed_tasks = 0
        pending = True
        error_types = ("EmbeddingIndexMismatch",)

    worker._discard_runtime_of_another_index(Waiting())
    rebuilt = await worker._build_runtime()

    assert stale.memory_store.embedding.settings.invocation.route == "openai/embed-a"
    assert worker._vector_runtime is None
    assert rebuilt.memory_store.embedding.settings.invocation.route == "openai/embed-b"


async def test_the_servers_search_engine_is_built_again_after_another_process_reindexed(
    db, config, settings, models, monkeypatch
) -> None:
    from memforge.server import admin_api

    monkeypatch.setattr(admin_api, "_SEARCH_ENGINE_REBUILD_S", 0.0)
    await _installation(db, config, settings)
    app = create_admin_app(db=db, config=config)

    async with _client(app) as client:
        before = (await client.post("/api/v1/memories/search", json={"query": "unrelated words only"})).json()
        await _candidate_b(db, settings)
        await ModelSettings(config, DefaultRuntimeProvider()).run_reindex(db)
        stale = (await client.post("/api/v1/memories/search", json={"query": "unrelated words only"})).json()
        rebuilt = (await client.post("/api/v1/memories/search", json={"query": "unrelated words only"})).json()

    assert len(before["results"]) == 3
    assert stale["results"] == []
    assert len(rebuilt["results"]) == 3
    assert models.models()[-1] == "openai/embed-b"


# -- The command line ------------------------------------------------------------------


def _cli(config: AppConfig, db: Database, monkeypatch, *arguments: str, embedding_model: str = ""):
    from memforge.main import cli

    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(config.base_dir))
    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", embedding_model)
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", ENDPOINT if embedding_model else "")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_API_KEY", "embedding-key" if embedding_model else "")
    monkeypatch.setenv("COLUMNS", "200")
    return CliRunner().invoke(cli, ["maintenance", "reindex", *arguments])


async def test_the_reindex_command_offers_the_rebuild_for_a_changed_environment_model(
    db, tmp_path, models, monkeypatch
) -> None:
    config = _environment_config(tmp_path)
    for memory_id, content in CONTENTS.items():
        await _write_memory(db, config, memory_id, content)
    asked = len(models.requests)

    check = await asyncio.to_thread(_cli, config, db, monkeypatch, "--check", embedding_model="embed-b")
    asked_by_the_check = len(models.requests) - asked
    applied = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply", embedding_model="embed-b")

    assert check.exit_code == 0, check.output
    assert "reindex required" in check.output
    assert "the embedding model in use" in check.output
    assert "no rebuild can start" not in check.output
    assert "maintenance reindex --apply" in check.output
    assert asked_by_the_check == 0
    assert applied.exit_code == 0, applied.output
    assert f"3 vectors in {collection_name_for(PROFILE_B.fingerprint)}" in applied.output


async def test_the_reindex_command_offers_the_rebuild_of_stored_vectors_before_the_model_is_checked(
    db, tmp_path, models, monkeypatch
) -> None:
    config = _environment_config(tmp_path)
    _vectors_from_before_indexes_were_recorded(config.storage.chroma_path, dimension=8)
    await db.insert_memory(_memory("mem-stored", "A fact from before the index was recorded"))

    check = await asyncio.to_thread(_cli, config, db, monkeypatch, "--check", embedding_model="embed-a")
    applied = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply", embedding_model="embed-a")

    assert check.exit_code == 0, check.output
    assert "the embedding model has not passed a check yet" in check.output
    assert "the embedding model in use" in check.output
    assert "maintenance reindex --apply" in check.output
    assert "--apply checks the embedding model first" in check.output
    assert applied.exit_code == 0, applied.output
    assert f"1 vectors in {collection_name_for(PROFILE_A.fingerprint)}" in applied.output


async def test_the_reindex_command_checks_without_changing_anything(db, config, settings, models, monkeypatch) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)

    default = await asyncio.to_thread(_cli, config, db, monkeypatch)
    check = await asyncio.to_thread(_cli, config, db, monkeypatch, "--check")

    for result in (default, check):
        assert result.exit_code == 0, result.output
        assert "ready" in result.output
        assert collection_name_for(PROFILE_A.fingerprint) in result.output
        assert "embed-a, 3 dimensions" in result.output
        assert "the embedding candidate that waits for the index" in result.output
        assert "maintenance reindex --apply" in result.output
    assert await db.get_latest_vector_reindex() is None
    assert await _active_route(db, config) == "openai/embed-a"
    assert KEY_A not in default.output and KEY_B not in default.output


async def test_the_reindex_command_applies_the_same_rebuild_as_the_server(db, config, settings, models, monkeypatch) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)

    applied = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply")
    again = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply")

    assert applied.exit_code == 0, applied.output
    assert "3/3 memories embedded" in applied.output
    assert f"3 vectors in {collection_name_for(PROFILE_B.fingerprint)}" in applied.output
    assert await _active_route(db, config) == "openai/embed-b"
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_B.fingerprint
    assert _collection_names(config.storage.chroma_path) == [collection_name_for(PROFILE_B.fingerprint)]
    assert again.exit_code == 1, again.output
    assert "reindex_not_required" in again.output


async def test_the_reindex_command_reports_a_failed_rebuild_and_leaves_the_index_in_use(
    db, config, settings, models, monkeypatch
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)

    async def fail_the_rebuild(request: dict) -> None:
        if request["model"] == "openai/embed-b":
            raise Rejection("the key was revoked")

    models.before_answer = fail_the_rebuild

    failed = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply")

    assert failed.exit_code == 1, failed.output
    assert "The rebuild failed (authentication_failed)" in failed.output
    assert "run the command again" in failed.output
    assert (await db.get_active_vector_index())["fingerprint"] == PROFILE_A.fingerprint


async def test_the_reindex_command_does_not_start_while_a_source_is_syncing(
    db, config, settings, models, monkeypatch
) -> None:
    await _installation(db, config, settings)
    await _candidate_b(db, settings)
    await db.acquire_source_activity(activity_id="running-sync", source_id="src-index", kind=SourceActivityKind.SYNC)

    refused = await asyncio.to_thread(_cli, config, db, monkeypatch, "--apply")

    assert refused.exit_code == 1, refused.output
    assert "sync_in_progress" in refused.output
    assert await db.get_latest_vector_reindex() is None
