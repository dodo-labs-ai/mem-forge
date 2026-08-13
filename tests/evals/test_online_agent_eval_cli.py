from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from click.testing import CliRunner

from memforge.evals.agent_evaluation import QualitySignal, bind_quality_signals
from memforge.main import cli
from memforge.storage.database import Database


def test_online_agent_evaluation_report_is_bounded_and_content_free(tmp_path) -> None:
    base_dir = tmp_path / "memforge"
    db_path = base_dir / "db" / "memforge.db"

    async def seed() -> None:
        db = Database(str(db_path))
        await db.connect()
        await db.db.execute(
            """INSERT INTO sources (
                   id, type, name, config, owner_user_id, access_policy
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("src-teams", "teams", "Teams", "{}", "user-1", "private"),
        )
        await db.record_agent_runtime_events(
            bind_quality_signals(
                (
                    QualitySignal(
                        event_name="evidence_admission_outcome",
                        outcome="rejected",
                        reason_code="unknown_evidence_block_id",
                        candidate_hash="a" * 64,
                    ),
                ),
                source_id="src-teams",
                source_type="teams",
                doc_id="doc-1",
                source_unit_id="unit-1",
                target_unit_revision_id="sur-1",
                projection_run_id="spr-1",
                derivation_id="sda-1",
                batch_id="batch-1",
                batch_attempt=1,
                extraction_contract_version="projection-extraction-v8",
                occurred_at=datetime(2026, 8, 13, 6, tzinfo=timezone.utc),
            )
        )
        await db.close()

    asyncio.run(seed())
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "online-report",
            "--from",
            "2026-08-13T05:00:00Z",
            "--to",
            "2026-08-13T07:00:00Z",
            "--source-type",
            "teams",
            "--event-id",
            "are-does-not-match",
            "--user-id",
            "user-1",
            "--include-private",
        ],
        env={"MEMFORGE_BASE_DIR": str(base_dir)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cohort"]["event_name_counts"] == {}
    assert payload["events"] == []

    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "online-report",
            "--from",
            "2026-08-13T05:00:00Z",
            "--to",
            "2026-08-13T07:00:00Z",
            "--source-type",
            "teams",
            "--user-id",
            "user-1",
            "--include-private",
        ],
        env={"MEMFORGE_BASE_DIR": str(base_dir)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cohort"]["event_name_counts"] == {
        "evidence_admission_outcome": 1,
    }
    [event] = payload["events"]
    assert event["source_id"] == "src-teams"
    assert event["candidate_hash"] == "a" * 64
    assert not {
        "prompt",
        "quote",
        "source_content",
        "memory_content",
        "provider_error_body",
    }.intersection(event)


def test_online_agent_evaluation_purge_cli_applies_configured_retention(tmp_path) -> None:
    base_dir = tmp_path / "memforge"
    db_path = base_dir / "db" / "memforge.db"

    async def seed() -> None:
        db = Database(str(db_path))
        await db.connect()
        await db.db.execute(
            """INSERT INTO sources (
                   id, type, name, config, owner_user_id, access_policy
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("src-teams", "teams", "Teams", "{}", "user-1", "private"),
        )
        await db.record_agent_runtime_events(
            bind_quality_signals(
                (QualitySignal("extraction_batch_outcome", "expected", "zero_candidates"),),
                source_id="src-teams",
                source_type="teams",
                doc_id="doc-1",
                source_unit_id="unit-1",
                target_unit_revision_id="sur-1",
                projection_run_id="spr-1",
                derivation_id="sda-1",
                batch_id="batch-1",
                batch_attempt=1,
                extraction_contract_version="projection-extraction-v8",
                occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        await db.close()

    asyncio.run(seed())
    result = CliRunner().invoke(
        cli,
        ["eval", "purge-runtime-events", "--before", "2026-02-01T00:00:00Z"],
        env={"MEMFORGE_BASE_DIR": str(base_dir)},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted"] == 1
