"""Private Agent Knowledge Bundle patching.

Agent-session clients upload evidence windows. This module owns the service-side
patch boundary that turns a structured patch proposal into private, stable
concept claims and then into searchable memories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memforge.memory.evidence import (
    AccessContext,
    AuthorityCase,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    MemoryRelationApplyService,
    RelationCandidateRecord,
    RelationDecision,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    build_candidate_universe,
    build_mandatory_candidate_bucket_results,
    relation_bundle_snapshot_audit,
)
from memforge.models import (
    DocumentRecord,
    Memory,
    ReplacementKind,
    Visibility,
    content_hash,
    slugify,
)


PatchAction = Literal[
    "create_new_concept",
    "update_existing_claim",
    "supersede_existing_claim",
    "add_new_claim",
    "no_output",
]
PatchOutcome = Literal[
    "applied",
    "skipped_not_memory",
    "skipped_ambiguous",
    "skipped_conflict",
    "rejected_scope",
    "parse_failed",
]
PatchResultBucket = Literal["applied", "failed", "no_output"]


class DurableClaim(BaseModel):
    """Durable memory shape produced by the LLM and rendered by the service."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    rationale: str | None = None

    @field_validator("rule", "scope")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be blank")
        return text

    @field_validator("rationale")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class AgentKnowledgePatchProposal(BaseModel):
    """Validated LLM proposal. The service validates scope before applying it."""

    model_config = ConfigDict(extra="forbid")

    action: PatchAction
    concept_id: str | None = None
    claim_id: str | None = None
    concept_type: (
        Literal[
            "preference",
            "convention",
            "procedure",
            "debugging_takeaway",
            "decision",
            "pitfall",
        ]
        | None
    ) = None
    title: str | None = None
    claim_text: str = ""
    durable_claim: DurableClaim | None = None
    memory_type: Literal["fact", "decision", "convention", "procedure"] = "fact"
    tags: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentKnowledgePatchResult:
    outcome: PatchOutcome
    result_bucket: PatchResultBucket
    concept_id: str | None = None
    claim_id: str | None = None
    memory_id: str | None = None
    reason: str | None = None


class AgentKnowledgeBundleService:
    """Apply private agent-session concept patches and reconcile memories."""

    def __init__(self, *, db, memory_store) -> None:
        self.db = db
        self.memory_store = memory_store

    async def apply_patch_proposal(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        owner_user_id: str,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        repo_identifier: str | None,
        project_key: str | None,
        submitted_at: datetime | None = None,
        source_updated_at: datetime | None,
    ) -> AgentKnowledgePatchResult:
        """Apply one structured patch proposal.

        V1 is private-only. Existing concept/claim writes must belong to
        ``owner_user_id`` and the same ``repo_identifier``.
        """

        submitted_at = _utc(submitted_at)
        source_updated_at = _utc(source_updated_at) if source_updated_at is not None else None
        if proposal.action == "no_output":
            return AgentKnowledgePatchResult(
                outcome="skipped_not_memory",
                result_bucket="no_output",
                reason=proposal.reason or "proposal returned no_output",
            )
        if not proposal.claim_text.strip():
            return AgentKnowledgePatchResult(
                outcome="parse_failed",
                result_bucket="failed",
                reason="claim_text is required",
            )
        text_error = _validate_memory_content(proposal)
        if text_error:
            return AgentKnowledgePatchResult(outcome="parse_failed", result_bucket="failed", reason=text_error)
        memory_content = _memory_content_for(proposal)

        if proposal.action == "create_new_concept":
            if not proposal.title or not proposal.concept_type:
                return AgentKnowledgePatchResult(
                    outcome="parse_failed",
                    result_bucket="failed",
                    reason="create_new_concept requires title and concept_type",
                )
            concept_id = proposal.concept_id or _stable_concept_id(
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                concept_type=proposal.concept_type,
                title=proposal.title,
            )
            claim_id = proposal.claim_id or _stable_claim_id(
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                concept_id=concept_id,
                claim_text=proposal.claim_text,
                citations=proposal.citations,
            )
            markdown_body = _render_markdown(
                title=proposal.title,
                concept_type=proposal.concept_type,
                repo_identifier=repo_identifier,
                claim_id=claim_id,
                claim_text=proposal.claim_text,
                citations=proposal.citations,
            )
            await self._write_concept_document(
                concept_id=concept_id,
                source_id=source_id,
                client=client,
                title=proposal.title,
                concept_type=proposal.concept_type,
                owner_user_id=owner_user_id,
                workspace=workspace,
                repo_identifier=repo_identifier,
                project_key=project_key,
                submitted_at=submitted_at,
                markdown_body=markdown_body,
            )
            memory_id = await self._insert_claim_memory(
                proposal=proposal,
                concept_id=concept_id,
                claim_id=claim_id,
                display_anchor=slugify(proposal.title),
                source_id=source_id,
                client=client,
                session_id=session_id,
                workspace=workspace,
                claim_text=proposal.claim_text,
                memory_content=memory_content,
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
                source_type="agent_session",
                submitted_at=submitted_at,
                observed_at=submitted_at,
                source_updated_at=source_updated_at,
                citations=proposal.citations,
                concept_projection={
                    "concept_id": concept_id,
                    "source_id": source_id,
                    "owner_user_id": owner_user_id,
                    "workspace": workspace,
                    "repo_identifier": repo_identifier,
                    "concept_type": proposal.concept_type,
                    "concept_path": _concept_path(
                        owner_user_id, repo_identifier, proposal.concept_type, proposal.title
                    ),
                    "title": proposal.title,
                    "markdown_body": markdown_body,
                    "frontmatter": {
                        "visibility": Visibility.PRIVATE.value,
                        "owner_user_id": owner_user_id,
                        "repo_identifier": repo_identifier,
                        "source_id": source_id,
                        "source_type": "agent_session",
                    },
                },
            )
            return AgentKnowledgePatchResult(
                outcome="applied",
                result_bucket="applied",
                concept_id=concept_id,
                claim_id=claim_id,
                memory_id=memory_id,
            )

        if proposal.action not in {"update_existing_claim", "supersede_existing_claim", "add_new_claim"}:
            return AgentKnowledgePatchResult(
                outcome="parse_failed",
                result_bucket="failed",
                reason="unsupported action",
            )

        concept = await self.db.get_agent_concept(proposal.concept_id or "")
        if not self._can_patch_concept(concept, owner_user_id, repo_identifier):
            return AgentKnowledgePatchResult(
                outcome="rejected_scope",
                result_bucket="failed",
                concept_id=proposal.concept_id,
                claim_id=proposal.claim_id,
                reason="private concept is outside the caller scope",
            )

        if proposal.action == "add_new_claim":
            claim_id = proposal.claim_id or _stable_claim_id(
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                concept_id=concept["id"],
                claim_text=proposal.claim_text,
                citations=proposal.citations,
            )
            concept_markdown_body = await self._render_concept_markdown_with_patch(
                concept,
                claim_id=claim_id,
                claim_text=proposal.claim_text,
                citations=proposal.citations,
            )
            memory_id = await self._insert_claim_memory(
                proposal=proposal,
                concept_id=concept["id"],
                claim_id=claim_id,
                display_anchor=slugify(proposal.claim_text)[:80],
                source_id=source_id,
                client=client,
                session_id=session_id,
                workspace=workspace,
                claim_text=proposal.claim_text,
                memory_content=memory_content,
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
                source_type="agent_session",
                submitted_at=submitted_at,
                observed_at=submitted_at,
                source_updated_at=source_updated_at,
                citations=proposal.citations,
                concept_markdown_body=concept_markdown_body,
            )
            return AgentKnowledgePatchResult(
                outcome="applied",
                result_bucket="applied",
                concept_id=concept["id"],
                claim_id=claim_id,
                memory_id=memory_id,
            )

        claim = await self.db.get_agent_claim(proposal.claim_id or "")
        if not claim or claim["concept_id"] != concept["id"]:
            return AgentKnowledgePatchResult(
                outcome="rejected_scope",
                result_bucket="failed",
                concept_id=concept["id"],
                claim_id=proposal.claim_id,
                reason="claim is outside the target concept",
            )

        concept_markdown_body = await self._render_concept_markdown_with_patch(
            concept,
            claim_id=claim["id"],
            claim_text=proposal.claim_text,
            citations=proposal.citations,
        )
        memory_id = await self._supersede_claim_memory(
            proposal=proposal,
            old_memory_id=claim["memory_id"],
            concept_id=concept["id"],
            claim_id=claim["id"],
            display_anchor=claim["display_anchor"],
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            claim_text=proposal.claim_text,
            memory_content=memory_content,
            memory_type=proposal.memory_type,
            tags=proposal.tags,
            confidence=proposal.confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            source_type="agent_session",
            replacement_reason=proposal.reason or "agent claim updated",
            replacement_kind=_replacement_kind_for_action(proposal.action),
            submitted_at=submitted_at,
            observed_at=submitted_at,
            source_updated_at=source_updated_at,
            citations=proposal.citations,
            concept_markdown_body=concept_markdown_body,
        )
        return AgentKnowledgePatchResult(
            outcome="applied",
            result_bucket="applied",
            concept_id=concept["id"],
            claim_id=claim["id"],
            memory_id=memory_id,
        )

    def _can_patch_concept(
        self,
        concept: dict | None,
        owner_user_id: str,
        repo_identifier: str | None,
    ) -> bool:
        return bool(
            concept
            and concept.get("visibility") == Visibility.PRIVATE.value
            and concept.get("owner_user_id") == owner_user_id
            and (concept.get("repo_identifier") or None) == (repo_identifier or None)
        )

    async def _insert_claim_memory(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        concept_id: str,
        claim_id: str,
        display_anchor: str,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        claim_text: str,
        memory_content: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        source_type: str,
        submitted_at: datetime,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_projection: dict[str, object] | None = None,
        concept_markdown_body: str | None = None,
    ) -> str:
        unit = _agent_evidence_unit(
            proposal=proposal,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            concept_id=concept_id,
            claim_id=claim_id,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            submitted_at=submitted_at,
        )
        lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [])
        if lifecycle.action is not LifecycleAction.CREATE_MEMORY or not lifecycle.created_memory_id:
            raise RuntimeError(f"unexpected agent claim create lifecycle action: {lifecycle.action}")
        memory_id = lifecycle.created_memory_id
        memory = self._build_claim_memory(
            memory_id=memory_id,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
            tags=tags,
            confidence=confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
        )
        existing = await self.db.get_memory(memory_id)
        relation_outcome = self._relation_outcome_bundle(
            unit=unit,
            relation_run_id=_relation_run_id(unit.id, session_id, proposal.action),
            lifecycle_action=lifecycle.action,
            review_case=None,
            memory_id=memory_id,
            candidates=[],
            confidence=confidence,
            reason=proposal.reason,
            submitted_at=submitted_at,
        )
        if existing is None:
            await self.memory_store.insert_agent_claim_memory(
                memory=memory,
                doc_id=concept_id,
                source_type=source_type,
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text.strip(),
                memory_type=memory_type,
                tags=tags,
                confidence=confidence,
                observed_at=observed_at,
                source_updated_at=source_updated_at,
                citations=citations,
                concept_projection=concept_projection,
                concept_markdown_body=concept_markdown_body,
                excerpt=claim_text.strip(),
                relation_outcome=relation_outcome,
            )
        else:
            await self.db.record_relation_outcome_bundle(relation_outcome)
        return memory_id

    async def _supersede_claim_memory(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        old_memory_id: str,
        concept_id: str,
        claim_id: str,
        display_anchor: str,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        claim_text: str,
        memory_content: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        source_type: str,
        replacement_reason: str,
        replacement_kind: ReplacementKind,
        submitted_at: datetime,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_markdown_body: str | None = None,
    ) -> str:
        unit = _agent_evidence_unit(
            proposal=proposal,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            concept_id=concept_id,
            claim_id=claim_id,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            submitted_at=submitted_at,
        )
        relation_run_id = _relation_run_id(unit.id, session_id, proposal.action)
        new_memory_id = _replacement_memory_id(unit, replacement_kind)
        memory = self._build_claim_memory(
            memory_id=new_memory_id,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
            tags=tags,
            confidence=confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
        )
        if old_memory_id == new_memory_id:
            existing_run = await self.db.get_relation_run(relation_run_id)
            if existing_run is None:
                raise RuntimeError("agent claim replacement retry is missing its relation run")
            if existing_run.result_memory_id != new_memory_id:
                raise RuntimeError("agent claim replacement retry result memory does not match current claim")
            committed_candidates = tuple(await self.db.get_relation_candidates(relation_run_id))
            relation_outcome = self._relation_outcome_bundle(
                unit=unit,
                relation_run_id=relation_run_id,
                lifecycle_action=LifecycleAction.SUPERSEDE_MEMORY,
                review_case=None,
                memory_id=new_memory_id,
                candidates=committed_candidates,
                incomplete_mandatory_buckets=existing_run.incomplete_mandatory_buckets,
                candidate_count=existing_run.candidate_count,
                confidence=confidence,
                reason=replacement_reason,
                submitted_at=submitted_at,
            )
            await self.db.record_relation_outcome_bundle(relation_outcome)
            await self.memory_store.ensure_agent_claim_memory_projection(
                memory,
                doc_id=concept_id,
                source_type=source_type,
                excerpt=claim_text.strip(),
                source_updated_at=source_updated_at,
            )
            return new_memory_id
        universe = await self._mandatory_candidate_universe(
            unit=unit,
            relation_run_id=relation_run_id,
            owner_user_id=owner_user_id,
            source_id=source_id,
            repo_identifier=repo_identifier,
        )
        target_candidate = next(
            (candidate for candidate in universe.candidates if candidate.memory_id == old_memory_id),
            None,
        )
        if target_candidate is None:
            raise RuntimeError("agent claim replacement target missing from mandatory candidate universe")
        decision = RelationDecision(
            candidate_memory_id=old_memory_id,
            relation_type=(RelationType.REFINES if replacement_kind == "revision" else RelationType.CONTRADICTS),
            authority_case=AuthorityCase.SAME_AGENT_CLAIM,
            confidence=confidence,
            reason=replacement_reason,
            proposed_memory_content=memory_content,
            evidence_excerpt=claim_text.strip(),
            matched_bucket=target_candidate.bucket,
            matched_bucket_complete=target_candidate.bucket_complete,
            classifier_batch_key=relation_run_id,
        )
        lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [decision])
        if lifecycle.action is not LifecycleAction.SUPERSEDE_MEMORY:
            raise RuntimeError(f"unexpected agent claim replace lifecycle action: {lifecycle.action}")
        relation_outcome = self._relation_outcome_bundle(
            unit=unit,
            relation_run_id=relation_run_id,
            lifecycle_action=lifecycle.action,
            review_case=None,
            memory_id=new_memory_id,
            candidates=universe.candidates,
            incomplete_mandatory_buckets=universe.incomplete_mandatory_buckets,
            candidate_count=universe.total_unique_candidates,
            confidence=confidence,
            reason=replacement_reason,
            submitted_at=submitted_at,
        )
        await self.memory_store.supersede_agent_claim_memory(
            old_memory_id,
            memory,
            doc_id=concept_id,
            source_type=source_type,
            excerpt=claim_text.strip(),
            replacement_reason=replacement_reason,
            replacement_kind=replacement_kind,
            claim_id=claim_id,
            concept_id=concept_id,
            display_anchor=display_anchor,
            claim_text=claim_text.strip(),
            memory_type=memory_type,
            tags=tags,
            confidence=confidence,
            observed_at=observed_at,
            source_updated_at=source_updated_at,
            relation_outcome=relation_outcome,
            citations=citations,
            concept_markdown_body=concept_markdown_body,
        )
        return new_memory_id

    async def _mandatory_candidate_universe(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        owner_user_id: str,
        source_id: str,
        repo_identifier: str | None,
    ):
        buckets = await build_mandatory_candidate_bucket_results(
            store=self.db,
            unit=unit,
            access_context=AccessContext(
                actor_user_id=owner_user_id,
                source_subscriptions=(source_id,),
                repo_identifier=repo_identifier,
                operation_type="agent_session_patch",
            ),
        )
        return build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            bucket_results=buckets,
        )

    def _relation_outcome_bundle(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        lifecycle_action: LifecycleAction,
        review_case,
        memory_id: str,
        candidates: tuple[RelationCandidateRecord, ...] | list[RelationCandidateRecord],
        incomplete_mandatory_buckets: tuple[str, ...] = (),
        confidence: float,
        reason: str,
        submitted_at: datetime,
        candidate_count: int | None = None,
    ) -> RelationOutcomeBundle:
        now = submitted_at.isoformat()
        relations = (
            EvidenceRelationRecord(
                evidence_unit_id=unit.id,
                memory_id=memory_id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.SAME_AGENT_CLAIM,
                is_authoritative_support=True,
                source_lineage_id=unit.source_lineage_id,
                confidence=confidence,
                reason=reason,
                excerpt=unit.excerpt,
                classifier_version="agent_session_intent_v1",
                relation_run_id=relation_run_id,
                created_at=now,
            ),
        )
        audit = {
            "source_patch_intent": unit.source_metadata.get("source_patch_intent"),
            **relation_bundle_snapshot_audit(candidates=candidates, relations=relations),
        }
        return RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=RelationRunRecord(
                id=relation_run_id,
                evidence_unit_id=unit.id,
                access_context_hash=unit.access_context_hash,
                candidate_count=len(candidates) if candidate_count is None else candidate_count,
                mandatory_candidate_count=sum(1 for candidate in candidates if candidate.is_mandatory),
                checked_candidate_count=sum(1 for candidate in candidates if candidate.was_checked),
                incomplete_mandatory_buckets=incomplete_mandatory_buckets,
                classifier_version="agent_session_intent_v1",
                lifecycle_action=lifecycle_action,
                review_case=review_case,
                status="applied",
                result_memory_id=memory_id,
                audit=audit,
                started_at=now,
                completed_at=now,
            ),
            candidates=tuple(candidates),
            relations=relations,
        )

    def _build_claim_memory(
        self,
        *,
        memory_id: str,
        claim_text: str,
        memory_content: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
    ) -> Memory:
        return Memory(
            id=memory_id,
            memory_type=memory_type,
            content=memory_content.strip(),
            content_hash=content_hash(memory_content.strip()),
            visibility=Visibility.PRIVATE.value,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
            tags=tags,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=claim_text.strip(),
        )

    async def _write_concept_document(
        self,
        *,
        concept_id: str,
        source_id: str,
        client: str,
        title: str,
        concept_type: str,
        owner_user_id: str,
        workspace: str,
        repo_identifier: str | None,
        project_key: str | None,
        submitted_at: datetime,
        markdown_body: str,
    ) -> None:
        await self.db.upsert_document(
            DocumentRecord(
                doc_id=concept_id,
                source=source_id,
                source_url=f"agent-knowledge://{slugify(owner_user_id)}/{concept_id}",
                title=title,
                space_or_project=project_key or "UNSORTED",
                author=client,
                last_modified=submitted_at,
                labels=[concept_type],
                version=content_hash(markdown_body),
                content_hash=content_hash(markdown_body),
                token_count=None,
                raw_content_uri=None,
                raw_content_type="text/markdown",
                normalized_content_uri=None,
                pdf_content_uri=None,
                last_synced=submitted_at,
                client=client,
            )
        )

    async def _render_concept_markdown_with_patch(
        self,
        concept: dict,
        *,
        claim_id: str,
        claim_text: str,
        citations: list[str],
    ) -> str:
        claims = await self.db.list_agent_claims(concept["id"])
        citations_by_claim = {claim["id"]: await self.db.list_agent_claim_citations(claim["id"]) for claim in claims}
        patched_claims = []
        claim_seen = False
        for claim in claims:
            if claim["id"] == claim_id:
                patched_claims.append({**claim, "claim_text": claim_text})
                claim_seen = True
            else:
                patched_claims.append(claim)
        if not claim_seen:
            patched_claims.append({"id": claim_id, "claim_text": claim_text})

        merged_citations: dict[str, list[str]] = {}
        for claim in patched_claims:
            existing = [
                citation["citation_url"]
                for citation in citations_by_claim.get(claim["id"], [])
                if citation["citation_url"].strip()
            ]
            if claim["id"] == claim_id:
                existing.extend(citation.strip() for citation in citations if citation.strip())
            merged_citations[claim["id"]] = list(dict.fromkeys(existing))

        return _render_markdown(
            title=concept["title"],
            concept_type=concept["concept_type"],
            repo_identifier=concept.get("repo_identifier"),
            claim_id=patched_claims[0]["id"] if patched_claims else claim_id,
            claim_text="\n\n".join(claim["claim_text"] for claim in patched_claims),
            citations=[citation for claim in patched_claims for citation in merged_citations[claim["id"]]],
        )


async def render_agent_knowledge_patch_prompt(
    *,
    db,
    owner_user_id: str,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    repo_identifier: str | None,
    branch: str | None,
    history_window: dict,
    events: list[dict],
    transcript_markdown: str,
) -> str:
    """Build the semantic patch prompt for one private agent-session window."""

    concepts = await db.list_agent_concepts(
        owner_user_id=owner_user_id,
        repo_identifier=repo_identifier,
        limit=20,
    )
    concept_lines = []
    for concept in concepts:
        claims = await db.list_agent_claims(concept["id"])
        claim_lines = [f"    - claim_id={claim['id']} :: {claim['claim_text']}" for claim in claims[:5]]
        concept_lines.append(
            "\n".join(
                [
                    f"- concept_id={concept['id']}",
                    f"  title={concept['title']}",
                    f"  type={concept['concept_type']}",
                    *claim_lines,
                ]
            )
        )

    existing = "\n".join(concept_lines) if concept_lines else "- none"
    evidence_lines = []
    for event in events:
        name = event.get("name")
        label = event.get("kind", "event")
        if name:
            label = f"{label}:{name}"
        text = event.get("text") or event.get("summary") or json.dumps(event, ensure_ascii=False)
        evidence_lines.append(f"[{label}] {text}")
    evidence = "\n\n".join(evidence_lines)
    if transcript_markdown.strip():
        evidence = f"{evidence}\n\nFull transcript evidence:\n{transcript_markdown.strip()}".strip()

    return f"""You are updating a private MemForge agent-memory bundle for one user.

Decision boundary:
- Write only durable preferences, conventions, procedures, decisions, pitfalls, or debugging takeaways.
- Do not summarize ordinary progress, transient status, or facts a future agent can rediscover from the current repo.
- Agent-session memories are private-only in this version.
- If the evidence updates an existing claim, use update_existing_claim and copy the exact concept_id and claim_id.
- If the evidence replaces or invalidates an existing claim, use supersede_existing_claim and copy the exact concept_id and claim_id. Lifecycle is represented only by action, not by claim_text or durable_claim.
- If it belongs in an existing concept but is a distinct durable claim, use add_new_claim and copy the exact concept_id.
- If it is a new durable concept, use create_new_concept with a concise title and concept_type.
- If nothing durable should be kept, use no_output.
- claim_text is the detailed atomic evidence statement. It may contain the full corrected rule or runbook step.
- claim_text may keep evidence details such as branch names, exact test names, run-log fragments, implementation checklists, timestamps, and deployment or verification notes when they are useful provenance.
- durable_claim is required for all non-no_output actions. It is the durable memory record to keep, and the service will render Memory.content from it.
- durable_claim.rule states the durable rule, decision, invariant, pitfall, or reusable takeaway in present tense.
- durable_claim.scope states where or when the rule applies.
- durable_claim.rationale may explain why the rule matters, but only when the reason is durable.
- Do not prefix durable_claim.scope with "Applies:"; the service adds that label when rendering.
- durable_claim must omit evidence-only details unless a specific detail is itself the durable rule.
- If the evidence only supports implementation narration, verification status, branch/test/deploy notes, or other provenance details, keep those details in claim_text and return no_output unless a clean durable_claim can be filled.
- If the durable takeaway cannot be separated from evidence details, return no_output instead of copying claim_text.

<operational_context>
- owner_user_id: {owner_user_id}
- client: {client}
- session_id: {session_id}
- trigger: {trigger}
- workspace: {workspace}
- repo_identifier: {repo_identifier or "none"}
- branch: {branch or "none"}
</operational_context>

<comparison_context>
Existing private concepts for this user and repo. Use these only to choose create/update/supersede/add action and IDs; do not extract new memory from this section alone.
{existing}
</comparison_context>

<candidate_evidence>
Canonical event evidence for this window. Create or update memory only when this evidence supports a clean durable_claim.
{evidence or "- no evidence"}
</candidate_evidence>
"""


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must include timezone information")
    return value.astimezone(timezone.utc)


def _agent_evidence_unit(
    *,
    proposal: AgentKnowledgePatchProposal,
    source_id: str,
    client: str,
    session_id: str,
    workspace: str,
    concept_id: str,
    claim_id: str,
    owner_user_id: str,
    repo_identifier: str | None,
    project_key: str | None,
    submitted_at: datetime,
) -> EvidenceUnit:
    claim_anchor = _claim_anchor(owner_user_id, repo_identifier, concept_id, claim_id)
    source_metadata = {
        "concept_id": concept_id,
        "claim_id": claim_id,
        "claim_anchor": claim_anchor,
        "source_patch_intent": proposal.action,
        "session_id": session_id,
        "workspace": workspace,
        "reason": proposal.reason,
        "citations": [citation for citation in proposal.citations if citation.strip()],
    }
    content = _memory_content_for(proposal)
    unit_id = _stable_id(
        "eunit",
        source_id,
        session_id,
        proposal.action,
        claim_anchor,
        content_hash(content),
    )
    return EvidenceUnit(
        id=unit_id,
        source_id=source_id,
        doc_id=concept_id,
        doc_revision_id=content_hash(proposal.claim_text.strip()),
        source_type="agent_session",
        client=client,
        repo_identifier=repo_identifier,
        source_anchor=claim_anchor,
        source_lineage_id=claim_anchor,
        source_metadata=source_metadata,
        project_key=project_key,
        visibility=Visibility.PRIVATE.value,
        owner_user_id=owner_user_id,
        observed_at=submitted_at.isoformat(),
        extractor_run_id=session_id,
        access_context_hash=_stable_id("access", owner_user_id, repo_identifier or "", source_id),
        content=content,
        excerpt=proposal.claim_text.strip(),
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )


def _claim_anchor(
    owner_user_id: str,
    repo_identifier: str | None,
    concept_id: str,
    claim_id: str,
) -> str:
    return f"{owner_user_id}:{repo_identifier or 'none'}:{concept_id}:{claim_id}"


def _relation_run_id(evidence_unit_id: str, session_id: str, action: str) -> str:
    return _stable_id("relrun", evidence_unit_id, session_id, action)


def _replacement_memory_id(unit: EvidenceUnit, replacement_kind: ReplacementKind) -> str:
    digest = sha256(f"{unit.id}\x1f{replacement_kind}".encode("utf-8")).hexdigest()[:8]
    return f"mem-{digest}"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _memory_content_for(proposal: AgentKnowledgePatchProposal) -> str:
    if proposal.durable_claim is None:
        return ""
    claim = proposal.durable_claim
    parts = [
        _ensure_sentence(claim.rule),
        f"Applies: {_ensure_sentence(claim.scope)}",
    ]
    if claim.rationale and claim.rationale.strip():
        parts.append(f"Why: {_ensure_sentence(claim.rationale)}")
    return "\n".join(parts).strip()


def _ensure_sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _validate_memory_content(proposal: AgentKnowledgePatchProposal) -> str | None:
    memory_content = _memory_content_for(proposal)
    if not memory_content:
        return "durable_claim is required"
    return None


def _replacement_kind_for_action(action: PatchAction) -> ReplacementKind:
    if action == "update_existing_claim":
        return "revision"
    if action == "supersede_existing_claim":
        return "supersession"
    raise ValueError(f"action does not replace an existing claim: {action}")


def _stable_concept_id(
    *,
    owner_user_id: str,
    repo_identifier: str | None,
    concept_type: str,
    title: str,
) -> str:
    return _stable_id("akb_concept", owner_user_id, repo_identifier or "", concept_type, title.strip())


def _stable_claim_id(
    *,
    owner_user_id: str,
    repo_identifier: str | None,
    concept_id: str,
    claim_text: str,
    citations: list[str],
) -> str:
    citation_identity = "\x1e".join(citation.strip() for citation in citations if citation.strip())
    return _stable_id(
        "akb_claim",
        owner_user_id,
        repo_identifier or "",
        concept_id,
        claim_text.strip(),
        citation_identity,
    )


def _concept_path(
    owner_user_id: str,
    repo_identifier: str | None,
    concept_type: str,
    title: str,
) -> str:
    repo = slugify(repo_identifier or "no-repo")
    return f"users/{slugify(owner_user_id)}/repos/{repo}/{concept_type}/{slugify(title)}.md"


def _render_markdown(
    *,
    title: str,
    concept_type: str,
    repo_identifier: str | None,
    claim_id: str,
    claim_text: str,
    citations: list[str],
) -> str:
    frontmatter = {
        "type": concept_type,
        "title": title,
        "visibility": Visibility.PRIVATE.value,
        "repo_identifier": repo_identifier,
    }
    citation_lines = "\n".join(f"- {citation}" for citation in citations if citation.strip())
    return (
        "---\n"
        f"{json.dumps(frontmatter, indent=2, sort_keys=True)}\n"
        "---\n\n"
        f"# {title}\n\n"
        "<!--\n"
        "mf:claim\n"
        f'id="{claim_id}"\n'
        "-->\n"
        f"{claim_text.strip()}\n\n"
        "# Citations\n\n"
        f"{citation_lines or '- none'}\n"
    )
