from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import SupportScopeVersion
from memforge.models import (
    ContentItem,
    DocumentRecord,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
)
from memforge.pipeline import extraction_contract
from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_V8,
    PROJECTION_EXTRACTION_V9,
    ProjectionExtractionContract,
    active_projection_extraction_contract,
    projection_extraction_contract,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    SourceUnitDerivationRequest,
    SourceUnitDeriver,
)
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "extraction-contract.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.parametrize(
    ("support_scope_version", "expected_version", "uses_fragment_catalog"),
    (
        (SupportScopeVersion.REFERENCE_SET_V1, PROJECTION_EXTRACTION_V8, False),
        (SupportScopeVersion.EVIDENCE_UNIT_SET_V2, PROJECTION_EXTRACTION_V9, True),
    ),
)
def test_active_projection_extraction_contract_is_scope_driven(
    support_scope_version: SupportScopeVersion,
    expected_version: str,
    uses_fragment_catalog: bool,
) -> None:
    contract = active_projection_extraction_contract(support_scope_version)

    assert contract.version == expected_version
    assert contract.uses_fragment_catalog is uses_fragment_catalog
    assert projection_extraction_contract(expected_version) is contract


def test_future_fragment_contract_promotion_changes_only_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v10 = ProjectionExtractionContract(
        version="projection-extraction-v10",
        uses_fragment_catalog=True,
    )
    monkeypatch.setitem(
        extraction_contract._PROJECTION_EXTRACTION_CONTRACTS,
        v10.version,
        v10,
    )
    monkeypatch.setitem(
        extraction_contract._ACTIVE_CONTRACT_VERSION_BY_SUPPORT_SCOPE,
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        v10.version,
    )

    assert active_projection_extraction_contract(
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2
    ) is v10
    assert projection_extraction_contract(v10.version) is v10


@pytest.mark.asyncio
async def test_future_fragment_contract_runs_through_the_deriver(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v10 = ProjectionExtractionContract(
        version="projection-extraction-v10",
        uses_fragment_catalog=True,
    )
    monkeypatch.setitem(
        extraction_contract._PROJECTION_EXTRACTION_CONTRACTS,
        v10.version,
        v10,
    )
    monkeypatch.setitem(
        extraction_contract._ACTIVE_CONTRACT_VERSION_BY_SUPPORT_SCOPE,
        SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        v10.version,
    )
    source_id = "source-v10-contract"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="V10 contract",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    now = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
    body = "# Rule\n\nA future contract must keep compiler-backed planning.\n"
    item = ContentItem(
        item_id="doc-v10-contract",
        title="V10 contract",
        source_url="https://example.test/repo/v10.md",
        last_modified=now,
        content_type="text/markdown",
        version="1",
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="github_repo",
        run_id="run-v10-contract",
        item=item,
        raw=RawContent(
            item=item,
            body=body.encode(),
            content_type="text/markdown",
        ),
        normalized=NormalizedContent(item=item, markdown_body=body),
        scope={},
        access_context={"visibility": "workspace"},
    )
    document = DocumentRecord(
        doc_id=item.item_id,
        source=source_id,
        source_url=item.source_url,
        title=item.title,
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        token_count=12,
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    seen_batches = []

    async def extract(batch):
        seen_batches.append(batch)
        return MemoryExtractionResult(memories=[])

    result = await SourceUnitDeriver(db).derive(
        SourceUnitDerivationRequest(
            projection=projection,
            context=SourceUnitDerivationContext(
                document=document,
                doc_type="document",
                project_key=None,
                repo_identifier=None,
                document_content=body,
                update_mode="full_document",
                changed_hunks=None,
                update_plan_stats=None,
                source_updated_at=now.isoformat(),
                user_id=None,
                source_activity_epoch=None,
            ),
            extract_batch=extract,
            max_concurrent=1,
            extraction_contract_version=(
                active_projection_extraction_contract(
                    SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                ).version
            ),
        )
    )

    assert result.derivation.extraction_contract_version == v10.version
    assert seen_batches
    assert {batch.__class__.__name__ for batch in seen_batches} == {
        "ProjectionExtractionBatch"
    }


def test_unknown_projection_extraction_contract_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="unknown projection extraction contract",
    ):
        projection_extraction_contract("projection-extraction-v999")
