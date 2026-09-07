from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.derivation_work import DerivationWork, payload_hash
from memforge.models import DocumentRecord
from memforge.source_derivation import SourceUnitDerivationContext, source_derivation_manifest
from memforge.storage.database import Database
from tests.test_revision_assessment import revisions


def staged_fixture():
    projection, _ = revisions('Two reviewers approve US releases.\n', 'unused')
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    document = DocumentRecord(doc_id='doc', source=projection.source_id,
        source_url='https://example.test/doc', title='Stage fixture', space_or_project='TEST',
        author=None, last_modified=now, labels=[], version='1', content_hash='hash', token_count=20,
        raw_content_uri=None, raw_content_type='text/markdown', normalized_content_uri=None,
        pdf_content_uri=None, last_synced=now)
    context = SourceUnitDerivationContext(document=document, doc_type='document', project_key=None,
        repo_identifier=None, document_content=projection.observation_revisions[0].content,
        update_mode='new', changed_hunks=None, update_plan_stats=None, source_updated_at=now.isoformat(),
        user_id=None, source_activity_epoch=None)
    return projection, source_derivation_manifest(projection, (), context=context)


async def prepare_database(path):
    db = Database(str(path))
    await db.connect()
    projection, manifest = staged_fixture()
    await db.upsert_source(id=projection.source_id, type='confluence', name='Stage fixture', config_json='{}', access_policy='workspace', owner_user_id='fixture-user')
    await db.stage_source_derivation(manifest)
    return db, manifest


@pytest.mark.asyncio
async def test_durable_work_reuses_completed_result_after_reopen_and_checks_dependencies(tmp_path):
    path = tmp_path / 'stages.db'
    db, root = await prepare_database(path)
    scan = DerivationWork.create('support_scan', {'target': root.target_unit_revision_id, 'dependencies': []})
    child = DerivationWork.create('support_finalize', {'dependencies': [[scan.id, payload_hash({'results': []})]]})
    try:
        with pytest.raises(ValueError, match='dependency missing'):
            await db.stage_derivation_work(derivation_id=root.id, work=child)
        assert await db.stage_derivation_work(derivation_id=root.id, work=scan) == scan
        with pytest.raises(ValueError, match='dependency incomplete'):
            await db.stage_derivation_work(derivation_id=root.id, work=child)
        result = {'results': []}
        completed = replace(scan, status='completed', result=result, result_hash=payload_hash(result))
        await db.record_derivation_work(derivation_id=root.id, work=completed)
    finally:
        await db.close()
    db = Database(str(path)); await db.connect()
    try:
        assert await db.stage_derivation_work(derivation_id=root.id, work=scan) == completed
        late_failure = replace(scan, status='retryable_failure', error_code='TimeoutError')
        assert await db.record_derivation_work(derivation_id=root.id, work=late_failure) == completed
        await db.stage_derivation_work(derivation_id=root.id, work=child)
        with pytest.raises(ValueError, match='live root'):
            await db.stage_derivation_work(derivation_id='another-root', work=child)
        changed_model = DerivationWork.create('support_scan', {**scan.manifest, 'model': 'different-model'})
        assert changed_model.id != scan.id
        assert (await db.stage_derivation_work(derivation_id=root.id, work=changed_model)).status == 'pending'
    finally:
        await db.close()


def test_corrupt_completed_result_and_manifest_are_rejected():
    work = DerivationWork.create('support_scan', {})
    with pytest.raises(ValueError, match='result mismatch'):
        DerivationWork.from_payload(replace(work, status='completed', result={'x': 1}, result_hash='wrong').to_payload())
    with pytest.raises(ValueError, match='identity mismatch'):
        DerivationWork.from_payload(replace(work, manifest={'kind':'support_scan','model':'changed'}).to_payload())
