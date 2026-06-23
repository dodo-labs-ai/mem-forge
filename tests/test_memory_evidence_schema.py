from __future__ import annotations

import pytest
import pytest_asyncio

from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "evidence.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _columns(database: Database, table: str) -> set[str]:
    async with database.db.execute(f"PRAGMA table_info({table})") as cur:
        return {row[1] async for row in cur}


async def _primary_key(database: Database, table: str) -> tuple[str, ...]:
    rows = []
    async with database.db.execute(f"PRAGMA table_info({table})") as cur:
        async for row in cur:
            rows.append(row)
    return tuple(row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5])


async def _table_sql(database: Database, table: str) -> str:
    async with database.db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.asyncio
async def test_evidence_units_schema_captures_source_scope_and_provenance(db: Database) -> None:
    assert await _primary_key(db, "evidence_units") == ("id",)
    assert await _columns(db, "evidence_units") >= {
        "id",
        "source_id",
        "doc_id",
        "doc_revision_id",
        "source_type",
        "client",
        "repo_identifier",
        "source_anchor",
        "source_lineage_id",
        "source_metadata_json",
        "project_key",
        "visibility",
        "owner_user_id",
        "observed_at",
        "extractor_run_id",
        "access_context_hash",
        "content",
        "excerpt",
        "evidence_provenance",
        "created_at",
        "updated_at",
    }


@pytest.mark.asyncio
async def test_evidence_relations_schema_has_current_pair_key_and_no_lifecycle_relation(db: Database) -> None:
    assert await _primary_key(db, "evidence_relations") == ("evidence_unit_id", "memory_id")
    sql = await _table_sql(db, "evidence_relations")
    assert "superseded_by" not in sql.lower()
    assert "no_relation" not in sql.lower()
    assert await _columns(db, "evidence_relations") >= {
        "evidence_unit_id",
        "memory_id",
        "relation_type",
        "authority_case",
        "is_authoritative_support",
        "source_lineage_id",
        "confidence",
        "reason",
        "proposed_memory_content",
        "excerpt",
        "classifier_version",
        "relation_run_id",
        "created_at",
    }


@pytest.mark.asyncio
async def test_relation_runs_schema_records_candidate_completeness_and_apply_outcome(db: Database) -> None:
    assert await _primary_key(db, "relation_runs") == ("id",)
    assert await _columns(db, "relation_runs") >= {
        "id",
        "evidence_unit_id",
        "access_context_hash",
        "candidate_count",
        "mandatory_candidate_count",
        "checked_candidate_count",
        "incomplete_mandatory_buckets_json",
        "classifier_version",
        "lifecycle_action",
        "review_case",
        "status",
        "result_memory_id",
        "audit_json",
        "started_at",
        "completed_at",
    }


@pytest.mark.asyncio
async def test_relation_candidates_schema_audits_checked_candidate_universe(
    db: Database,
) -> None:
    assert await _primary_key(db, "relation_candidates") == (
        "relation_run_id",
        "bucket",
        "memory_id",
    )
    assert await _columns(db, "relation_candidates") >= {
        "relation_run_id",
        "evidence_unit_id",
        "memory_id",
        "bucket",
        "bucket_rank",
        "candidate_rank",
        "score",
        "is_mandatory",
        "bucket_complete",
        "was_checked",
        "reason",
        "created_at",
    }
