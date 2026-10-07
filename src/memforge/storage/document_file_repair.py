"""Explicit repair primitives for Document files; never invoked by database startup.

The operator owns the bounded input snapshot, recovery export and transaction.
Only recorded byte fingerprints authorize a copy. Originals are never deleted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from memforge.storage.document_store import DocumentStore


@dataclass(frozen=True)
class DocumentFileRepair:
    recorded: Mapping[str, Any]
    # A recorded field maps to the proven bytes' digest, or None to clear it.
    objects: Mapping[str, str | None]


def plan_document_file_repair(row: Mapping[str, Any], store: DocumentStore) -> DocumentFileRepair:
    """Read one recorded input and classify its foreign files without writing."""
    objects: dict[str, str | None] = {}
    for field, hash_field in (
        ("raw_content_uri", "raw_content_sha256"),
        ("normalized_content_uri", "normalized_content_hash"),
        ("pdf_content_uri", None),
    ):
        uri = row.get(field)
        if not uri or store.belongs_to_document(uri, source_id=row["source_id"], doc_id=row["document_id"]):
            continue
        digest = None
        expected = row.get(hash_field) if hash_field else None
        if expected:
            try:
                body = store.read_artifact(uri)
            except FileNotFoundError:
                pass
            else:
                observed = hashlib.sha256(body).hexdigest()
                if observed == expected:
                    digest = observed
        objects[field] = digest
    return DocumentFileRepair(recorded=dict(row), objects=objects)


def materialize_document_file_repair(plan: DocumentFileRepair, store: DocumentStore) -> dict[str, str | None]:
    """Copy proven bytes under owned keys; return file references for the transaction."""
    row = plan.recorded
    updates = {field: row.get(field) for field in ("raw_content_uri", "normalized_content_uri", "pdf_content_uri")}
    for field, digest in plan.objects.items():
        uri = None
        if digest is not None:
            body = store.read_artifact(row[field])
            if hashlib.sha256(body).hexdigest() != digest:
                raise RuntimeError("Document file changed after repair planning")
            if field == "normalized_content_uri":
                uri = store.store_normalized(row["source_id"], row["document_id"], digest, body.decode("utf-8"))
            else:
                uri = store.store_raw(row["source_id"], row["document_id"], digest, body, row["raw_content_type"])
        updates[field] = uri
    return updates


# DB-API SQL shared by SQLite and HANA. The caller holds one transaction for the
# whole reviewed plan and must roll it back when any update raises.
_REPAIR_GUARDS = (
    "source_unit_id",
    "unit_revision_id",
    "source_id",
    "document_id",
    "recorded_at",
    "raw_content_uri",
    "raw_content_type",
    "raw_content_sha256",
    "normalized_content_uri",
    "normalized_content_hash",
    "pdf_content_uri",
)
REPAIR_DOCUMENT_FILES_SQL = (
    "UPDATE source_unit_inputs SET raw_content_uri = ?, normalized_content_uri = ?, pdf_content_uri = ? WHERE "
    + " AND ".join(f"COALESCE({field}, '') = ?" for field in _REPAIR_GUARDS)
    + " AND EXISTS (SELECT 1 FROM source_units u WHERE u.id = source_unit_inputs.source_unit_id "
    "AND u.current_revision_id = source_unit_inputs.unit_revision_id)"
)


def record_document_file_repair(cursor: Any, plan: DocumentFileRepair, updates: Mapping[str, str | None]) -> None:
    """Record one exactly matched current input; stale or missing rows reject the plan."""
    params = tuple(updates[field] for field in ("raw_content_uri", "normalized_content_uri", "pdf_content_uri"))
    params += tuple(plan.recorded.get(field) or "" for field in _REPAIR_GUARDS)
    cursor.execute(REPAIR_DOCUMENT_FILES_SQL, params)
    if cursor.rowcount != 1:
        raise RuntimeError("Document file repair plan is stale")
