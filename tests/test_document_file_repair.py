"""Explicit repair preserves byte provenance, original files and stale guards."""

import hashlib
import sqlite3

import pytest

from memforge.storage.document_file_repair import (
    materialize_document_file_repair,
    plan_document_file_repair,
    record_document_file_repair,
)
from memforge.storage.document_store import LocalDocumentStore


def recorded(tmp_path):
    store = LocalDocumentStore(str(tmp_path / "files"))
    raw = store.store_raw("old", "other", "Page", b'[{"doc_id":"doc"}]', "application/json")
    normalized = store.store_normalized("old", "other", "Page", "Exact revision content")
    row = dict(
        source_unit_id="unit",
        unit_revision_id="revision",
        source_id="source",
        document_id="doc",
        recorded_at="watermark",
        raw_content_uri=raw,
        raw_content_type="application/json",
        raw_content_sha256=None,
        normalized_content_uri=normalized,
        normalized_content_hash=hashlib.sha256(b"Exact revision content").hexdigest(),
        pdf_content_uri=raw,
    )
    return store, row


def test_only_recorded_hashes_prove_a_revision_and_original_files_remain(tmp_path):
    store, row = recorded(tmp_path)
    plan = plan_document_file_repair(row, store)
    updates = materialize_document_file_repair(plan, store)
    assert updates["raw_content_uri"] is None  # Matching Document ID is not a revision fingerprint.
    assert updates["pdf_content_uri"] is None
    assert store.read_artifact(updates["normalized_content_uri"]) == b"Exact revision content"
    assert store.belongs_to_document(updates["normalized_content_uri"], source_id="source", doc_id="doc")
    assert store.read_artifact(row["raw_content_uri"])
    assert store.read_artifact(row["normalized_content_uri"])
    assert materialize_document_file_repair(plan, store) == updates


def test_changed_bytes_reject_a_planned_copy(tmp_path):
    store, row = recorded(tmp_path)
    plan = plan_document_file_repair(row, store)
    from pathlib import Path

    Path(row["normalized_content_uri"]).write_text("Different revision")
    with pytest.raises(RuntimeError, match="changed after"):
        materialize_document_file_repair(plan, store)


@pytest.mark.parametrize(
    "stale_field", [None, "recorded_at", "raw_content_uri", "normalized_content_hash", "current_revision"]
)
def test_guarded_record_rejects_stale_plan_and_rolls_back(tmp_path, stale_field):
    store, row = recorded(tmp_path)
    plan = plan_document_file_repair(row, store)
    updates = materialize_document_file_repair(plan, store)
    connection = sqlite3.connect(":memory:")
    fields = list(row)
    connection.execute("CREATE TABLE source_unit_inputs (" + ", ".join(f"{f} TEXT" for f in fields) + ")")
    connection.execute("CREATE TABLE source_units (id TEXT, current_revision_id TEXT)")
    connection.execute(
        "INSERT INTO source_unit_inputs VALUES (" + ",".join("?" for _ in fields) + ")", tuple(row.values())
    )
    connection.execute("INSERT INTO source_units VALUES (?, ?)", ("unit", "revision"))
    if stale_field == "current_revision":
        connection.execute("UPDATE source_units SET current_revision_id = 'new-revision'")
    elif stale_field:
        connection.execute(f"UPDATE source_unit_inputs SET {stale_field} = 'changed'")
    connection.commit()
    if stale_field:
        with pytest.raises(RuntimeError, match="stale"):
            with connection:
                record_document_file_repair(connection.cursor(), plan, updates)
    else:
        with connection:
            record_document_file_repair(connection.cursor(), plan, updates)
        assert (
            connection.execute("SELECT normalized_content_hash FROM source_unit_inputs").fetchone()[0]
            == row["normalized_content_hash"]
        )
        assert connection.execute("SELECT raw_content_uri FROM source_unit_inputs").fetchone()[0] is None


@pytest.mark.parametrize("matches", [True, False])
def test_raw_copy_requires_the_recorded_byte_hash(tmp_path, matches):
    store, row = recorded(tmp_path)
    original = store.read_artifact(row["raw_content_uri"])
    row["raw_content_sha256"] = hashlib.sha256(original).hexdigest() if matches else "0" * 64
    updates = materialize_document_file_repair(plan_document_file_repair(row, store), store)
    assert bool(updates["raw_content_uri"]) is matches
    if matches:
        assert store.read_artifact(updates["raw_content_uri"]) == original
        assert store.belongs_to_document(updates["raw_content_uri"], source_id="source", doc_id="doc")


def test_provider_read_failure_does_not_turn_into_permission_to_clear(tmp_path, monkeypatch):
    store, row = recorded(tmp_path)

    def unavailable(uri):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr(store, "read_artifact", unavailable)
    with pytest.raises(ConnectionError):
        plan_document_file_repair(row, store)


def test_a_stale_later_row_rolls_back_the_entire_repair(tmp_path):
    store, row = recorded(tmp_path)
    plan = plan_document_file_repair(row, store)
    updates = materialize_document_file_repair(plan, store)
    connection = sqlite3.connect(":memory:")
    fields = list(row)
    connection.execute("CREATE TABLE source_unit_inputs (" + ", ".join(f"{f} TEXT" for f in fields) + ")")
    connection.execute("CREATE TABLE source_units (id TEXT, current_revision_id TEXT)")
    connection.execute(
        "INSERT INTO source_unit_inputs VALUES (" + ",".join("?" for _ in fields) + ")", tuple(row.values())
    )
    connection.execute("INSERT INTO source_units VALUES (?, ?)", ("unit", "revision"))
    connection.commit()
    with pytest.raises(RuntimeError, match="stale"):
        with connection:
            record_document_file_repair(connection.cursor(), plan, updates)
            record_document_file_repair(connection.cursor(), plan, updates)
    assert connection.execute("SELECT raw_content_uri FROM source_unit_inputs").fetchone()[0] == row["raw_content_uri"]
    assert (
        connection.execute("SELECT normalized_content_uri FROM source_unit_inputs").fetchone()[0]
        == row["normalized_content_uri"]
    )
