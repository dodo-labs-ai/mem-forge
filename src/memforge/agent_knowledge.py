"""Private Agent Knowledge Bundle patching.

Agent-session clients upload evidence windows. This module owns the service-side
patch boundary that turns a structured patch proposal into private, stable
concept claims and then into searchable memories.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memforge.agent_knowledge_markdown import (
    render_agent_concept_markdown,
    render_agent_concept_markdown_with_patch,
    render_agent_concept_markdown_without_claim,
)
from memforge.memory.evidence import (
    AccessContext,
    AuthorityCase,
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
from memforge.memory.lifecycle_plan import LifecycleMutationType, LifecyclePlan, ReconciliationScope
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_access_context_hash,
    lifecycle_plan_id,
)
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    ReplacementKind,
    Visibility,
    content_hash,
    slugify,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import project_source_item


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
    reason: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    primary_evidence_ids: list[str] = Field(default_factory=list)
    covered_concept_id: str | None = None
    covered_claim_id: str | None = None


@dataclass(frozen=True)
class AgentKnowledgePatchResult:
    outcome: PatchOutcome
    result_bucket: PatchResultBucket
    concept_id: str | None = None
    claim_id: str | None = None
    memory_id: str | None = None
    covered_concept_id: str | None = None
    covered_claim_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _AgentRelationIntent:
    client: str
    source_metadata: Mapping[str, object]


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
            covered_concept_id, covered_claim_id = await self._validated_covered_ids(
                proposal=proposal,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
            )
            return AgentKnowledgePatchResult(
                outcome="skipped_not_memory",
                result_bucket="no_output",
                covered_concept_id=covered_concept_id,
                covered_claim_id=covered_claim_id,
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

        resolved_claim: dict | None = None
        if proposal.action in {"create_new_concept", "add_new_claim"}:
            targets = await self._claim_targets_from_memory_candidates(
                proposal=proposal,
                source_id=source_id,
                client=client,
                session_id=session_id,
                workspace=workspace,
                memory_content=memory_content,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
            )
            if len(targets) == 1:
                resolved_claim, _matched_memory = targets[0]
                proposal.action = "update_existing_claim"
                proposal.concept_id = resolved_claim["concept_id"]
                proposal.claim_id = resolved_claim["id"]
            elif len(targets) > 1:
                return AgentKnowledgePatchResult(
                    outcome="skipped_ambiguous",
                    result_bucket="failed",
                    reason="create/add proposal matched multiple current claim memory targets",
                )

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
            markdown_body = render_agent_concept_markdown(
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

        if proposal.action in {"update_existing_claim", "supersede_existing_claim"} and not proposal.claim_id:
            resolution = await self._resolve_claim_target_from_memory_candidate(
                proposal=proposal,
                source_id=source_id,
                client=client,
                session_id=session_id,
                workspace=workspace,
                memory_content=memory_content,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
            )
            if isinstance(resolution, AgentKnowledgePatchResult):
                return resolution
            resolved_claim = resolution
            proposal.concept_id = resolved_claim["concept_id"]
            proposal.claim_id = resolved_claim["id"]

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

        claim = resolved_claim or await self.db.get_agent_claim(proposal.claim_id or "")
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

    async def replace_claim_from_user_correction(
        self,
        *,
        old_memory_id: str,
        replacement_content: str,
        provenance: str,
        reason: str,
        replacement_kind: ReplacementKind,
        observed_at: datetime,
    ) -> str:
        """Apply an explicit user correction through Agent Source Projection."""

        claim = await self.db.get_agent_claim_by_memory_id(old_memory_id)
        concept = (
            await self.db.get_agent_concept(claim["concept_id"])
            if claim is not None
            else None
        )
        old_memory = await self.db.get_memory(old_memory_id)
        if claim is None or concept is None or old_memory is None:
            raise ValueError("corrected agent claim lineage is incomplete")
        source_id = str(concept["source_id"])
        owner_user_id = str(concept["owner_user_id"])
        session_id = "correction-" + sha256(
            "\x1f".join((old_memory_id, replacement_content, reason)).encode("utf-8")
        ).hexdigest()[:20]
        action: PatchAction = (
            "update_existing_claim"
            if replacement_kind == "revision"
            else "supersede_existing_claim"
        )
        proposal = AgentKnowledgePatchProposal(
            action=action,
            concept_id=str(concept["id"]),
            claim_id=str(claim["id"]),
            claim_text=replacement_content,
            durable_claim=DurableClaim(rule=replacement_content, scope=provenance),
            memory_type=str(claim["memory_type"]),
            reason=reason,
            confidence=float(claim["confidence"]),
        )
        markdown_body = await self._render_concept_markdown_with_patch(
            concept,
            claim_id=str(claim["id"]),
            claim_text=replacement_content,
            citations=[],
        )
        return await self._supersede_claim_memory(
            proposal=proposal,
            old_memory_id=old_memory_id,
            concept_id=str(concept["id"]),
            claim_id=str(claim["id"]),
            display_anchor=str(claim["display_anchor"]),
            source_id=source_id,
            client="user_correction",
            session_id=session_id,
            workspace=str(concept["workspace"]),
            claim_text=replacement_content,
            memory_content=replacement_content,
            memory_type=str(claim["memory_type"]),
            confidence=float(claim["confidence"]),
            owner_user_id=owner_user_id,
            repo_identifier=concept.get("repo_identifier"),
            project_key=old_memory.project_key,
            source_type="agent_session",
            replacement_reason=reason,
            replacement_kind=replacement_kind,
            memory_extraction_context=provenance,
            submitted_at=observed_at,
            observed_at=observed_at,
            source_updated_at=observed_at,
            concept_markdown_body=markdown_body,
        )

    async def retire_claim_from_user_request(
        self,
        *,
        old_memory_id: str,
        reason: str,
        observed_at: datetime,
    ) -> str:
        """Retire one managed Agent claim through its Source Projection."""

        claim = await self.db.get_agent_claim_by_memory_id(old_memory_id)
        concept = (
            await self.db.get_agent_concept(claim["concept_id"])
            if claim is not None
            else None
        )
        old_memory = await self.db.get_memory(old_memory_id)
        if claim is None or concept is None or old_memory is None:
            raise ValueError("retired agent claim lineage is incomplete")
        markdown_body = await render_agent_concept_markdown_without_claim(
            self.db,
            concept,
            claim_id=str(claim["id"]),
        )
        session_id = "retirement-" + sha256(
            "\x1f".join((old_memory_id, reason)).encode("utf-8")
        ).hexdigest()[:20]
        projection, plan, target_memory_id = await self._build_agent_claim_lifecycle(
            concept_id=str(concept["id"]),
            source_id=str(concept["source_id"]),
            client="user_retirement",
            session_id=session_id,
            workspace=str(concept["workspace"]),
            claim_text=str(claim["claim_text"]),
            memory_content=None,
            memory_type=str(claim["memory_type"]),
            confidence=float(claim["confidence"]),
            owner_user_id=str(concept["owner_user_id"]),
            repo_identifier=concept.get("repo_identifier"),
            project_key=old_memory.project_key,
            submitted_at=observed_at,
            source_updated_at=observed_at,
            concept_projection=None,
            concept_markdown_body=markdown_body,
            incumbent_memory_id=old_memory_id,
            reconcile_action=ReconcileAction.DELETE,
            reconciliation_reason=reason,
        )
        if not any(
            mutation.mutation_type is LifecycleMutationType.RETIRE_MEMORY
            and mutation.memory_id == old_memory_id
            for mutation in plan.mutations
        ):
            raise ValueError("agent claim retirement requires an enabled lifecycle gate")
        await self.memory_store.retire_agent_claim_memory(
            old_memory_id=old_memory_id,
            projection=projection,
            plan=plan,
            claim_id=str(claim["id"]),
            concept_id=str(concept["id"]),
            display_anchor=str(claim["display_anchor"]),
            claim_text=str(claim["claim_text"]),
            memory_type=str(claim["memory_type"]),
            confidence=float(claim["confidence"]),
            observed_at=observed_at,
            concept_markdown_body=markdown_body,
        )
        return target_memory_id

    async def _resolve_claim_target_from_memory_candidate(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        memory_content: str | None,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
    ) -> dict | AgentKnowledgePatchResult:
        scoped_claims = await self._claim_targets_from_memory_candidates(
            proposal=proposal,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            memory_content=memory_content,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
        )
        if not scoped_claims:
            return AgentKnowledgePatchResult(
                outcome="rejected_scope",
                result_bucket="failed",
                reason="update/supersede proposal did not resolve a current claim memory target",
            )
        if len(scoped_claims) > 1:
            return AgentKnowledgePatchResult(
                outcome="skipped_ambiguous",
                result_bucket="failed",
                reason="update/supersede proposal matched multiple current claim memory targets",
            )
        return scoped_claims[0][0]

    async def _claim_targets_from_memory_candidates(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        memory_content: str,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
    ) -> list[tuple[dict, Memory]]:
        del client, workspace
        candidate_memory = self._build_claim_memory(
            memory_id=_stable_id("agent_candidate", source_id, session_id, content_hash(memory_content)),
            claim_text=proposal.claim_text,
            memory_content=memory_content,
            memory_type=proposal.memory_type,
            confidence=proposal.confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
        )
        matches = await self.memory_store.find_agent_claim_memory_candidates(
            candidate_memory,
            source_id=source_id,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
        )
        scoped_claims: list[tuple[dict, Memory]] = []
        seen_claim_ids: set[str] = set()
        for matched_memory, _score in matches:
            claim = await self.db.get_agent_claim_by_memory_id(matched_memory.id)
            if claim is None or claim["id"] in seen_claim_ids:
                continue
            concept = await self.db.get_agent_concept(claim["concept_id"])
            if not self._can_patch_concept(concept, owner_user_id, repo_identifier):
                continue
            scoped_claims.append((claim, matched_memory))
            seen_claim_ids.add(claim["id"])
        return scoped_claims

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
        intent = _agent_relation_intent(
            proposal=proposal,
            client=client,
            session_id=session_id,
            workspace=workspace,
            concept_id=concept_id,
            claim_id=claim_id,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
        )
        projection, plan, memory_id = await self._build_agent_claim_lifecycle(
            concept_id=concept_id,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
            confidence=confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            submitted_at=submitted_at,
            source_updated_at=source_updated_at,
            concept_projection=concept_projection,
            concept_markdown_body=concept_markdown_body,
            incumbent_memory_id=None,
            reconcile_action=ReconcileAction.ADD,
            reconciliation_reason=proposal.reason or "agent claim created",
        )
        plan, unit = self._bind_relation_evidence_to_plan(
            plan=plan,
            target_memory_id=memory_id,
            intent=intent,
        )
        lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [])
        if lifecycle.action is not LifecycleAction.CREATE_MEMORY or not lifecycle.created_memory_id:
            raise RuntimeError(f"unexpected agent claim create lifecycle action: {lifecycle.action}")
        memory = self._build_claim_memory(
            memory_id=memory_id,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
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
                projection=projection,
                lifecycle_plan=plan,
                doc_id=concept_id,
                source_type=source_type,
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text.strip(),
                memory_type=memory_type,
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

    async def _build_agent_claim_lifecycle(
        self,
        *,
        concept_id: str,
        source_id: str,
        client: str,
        session_id: str,
        workspace: str,
        claim_text: str,
        memory_content: str,
        memory_type: str,
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        submitted_at: datetime,
        source_updated_at: datetime | None,
        concept_projection: dict[str, object] | None,
        concept_markdown_body: str | None,
        incumbent_memory_id: str | None,
        reconcile_action: ReconcileAction,
        reconciliation_reason: str,
        memory_extraction_context: str | None = None,
    ):
        """Build the provider-neutral projection and complete claim plan.

        Agent Knowledge is command-originated source content, but its durable
        Memory follows the same Source Projection and Lifecycle Plan seam as
        every Gene-backed source.
        """

        existing_concept = await self.db.get_agent_concept(concept_id)
        markdown_body = (
            concept_markdown_body
            or str((concept_projection or {}).get("markdown_body") or "")
            or str((existing_concept or {}).get("markdown_body") or "")
        )
        if not markdown_body or (
            reconcile_action is not ReconcileAction.DELETE
            and claim_text.strip() not in markdown_body
        ):
            raise ValueError("agent claim projection must contain the exact claim evidence")
        title = str(
            (concept_projection or {}).get("title")
            or (existing_concept or {}).get("title")
            or concept_id
        )
        observed_at = source_updated_at or submitted_at
        item = ContentItem(
            item_id=concept_id,
            title=title,
            source_url=f"agent-knowledge://{slugify(owner_user_id)}/{concept_id}",
            last_modified=observed_at,
            content_type="text/markdown",
            space_or_project=project_key or "UNSORTED",
            version=content_hash(markdown_body),
            author=client,
        )
        native = {
            "doc_id": concept_id,
            "markdown": markdown_body,
            "receipt": {
                "client": client,
                "session_id": session_id,
                "history_window_kind": "agent_knowledge_patch",
                "workspace": workspace,
            },
        }
        current_unit = await self.db.find_source_unit_by_document_id(source_id, concept_id)
        prior_unit_revision = (
            await self.db.get_current_source_unit_revision(current_unit.id)
            if current_unit is not None
            else None
        )
        prior_observation_revisions = (
            await self.db.get_current_source_observation_revisions(current_unit.id)
            if current_unit is not None
            else {}
        )
        run_digest = sha256(
            "\x1f".join(
                (source_id, concept_id, client, session_id, content_hash(markdown_body))
            ).encode("utf-8")
        ).hexdigest()[:20]
        raw = RawContent(
            item=item,
            body=json.dumps(native, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )
        projection = project_source_item(
            source_id=source_id,
            source_type="agent_session",
            run_id=f"agent-projection-{run_digest}",
            item=item,
            raw=raw,
            normalized=NormalizedContent(item=item, markdown_body=markdown_body),
            scope={"managed_capture_source": source_id},
            access_context={
                "visibility": Visibility.PRIVATE.value,
                "owner_user_id": owner_user_id,
            },
            prior_unit_revision=prior_unit_revision,
            prior_observation_revisions=prior_observation_revisions,
        )
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=f"scope:{projection.run_id}",
            source_id=source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        raw_memory = (
            RawMemory(
                content=memory_content,
                memory_type=memory_type,
                confidence=confidence,
                extraction_context=(memory_extraction_context or claim_text).strip(),
                evidence_quote=claim_text.strip(),
            )
            if memory_content is not None
            else None
        )
        access_hash = lifecycle_access_context_hash(
            visibility=Visibility.PRIVATE.value,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
        )
        source_support = await self.db.get_source_unit_support_reference_ids(
            scope.source_unit_id
        )
        incumbents: dict[str, Memory] = {}
        incumbent_candidates: dict[str, RawMemory] = {}
        for memory_id in sorted(source_support):
            current = await self.db.get_memory(memory_id)
            if current is not None and current.status == "active":
                incumbents[memory_id] = current
                claim = await self.db.get_agent_claim_by_memory_id(memory_id)
                if claim is None:
                    raise ValueError("agent Source Unit support points to a non-claim Memory")
                incumbent_candidates[memory_id] = RawMemory(
                    content=current.content,
                    memory_type=current.memory_type,
                    confidence=current.confidence,
                    extraction_context=str(claim["claim_text"]).strip(),
                    evidence_quote=str(claim["claim_text"]).strip(),
                )
        if incumbent_memory_id is not None and incumbent_memory_id not in incumbents:
            raise ValueError("agent claim replacement lacks current Source Unit support")
        evidence_candidates = [
            candidate
            for memory_id, candidate in sorted(incumbent_candidates.items())
            if memory_id != incumbent_memory_id
        ]
        if raw_memory is not None:
            evidence_candidates.append(raw_memory)
        evidence = build_projected_claim_evidence(
            projection=projection,
            raw_memories=tuple(evidence_candidates),
            doc_id=concept_id,
            source_type="agent_session",
            project_key=project_key,
            visibility=Visibility.PRIVATE.value,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            access_context_hash=access_hash,
            extractor_run_id=projection.run_id,
            observed_at=observed_at.isoformat(),
        )
        operations = [
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=memory_id,
                memory=incumbent_candidates[memory_id],
                reason="unchanged agent claim",
            )
            for memory_id in sorted(incumbents)
            if memory_id != incumbent_memory_id
        ]
        operations.append(
            ReconcileOperation(
                action=reconcile_action,
                memory_id=incumbent_memory_id,
                memory=raw_memory,
                reason=reconciliation_reason,
            )
        )
        all_active_support = {
            memory_id: await self.db.get_active_memory_support_reference_ids(memory_id)
            for memory_id in incumbents
        }
        support_hashes = {
            memory_id: await self.db.get_memory_support_set_hash(memory_id)
            for memory_id in incumbents
        }
        gate = await self.db.get_lifecycle_gate(source_id)
        plan = build_lifecycle_plan(
            plan_id=lifecycle_plan_id(scope),
            scope=scope,
            gate_state=gate.state,
            operations=tuple(operations),
            incumbents=incumbents,
            source_support_reference_ids=source_support,
            all_active_support_reference_ids=all_active_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=tuple(
                revision.id for revision in projection.observation_revisions
            ),
            new_evidence_reference_ids=(),
            evidence_reference_ids_by_claim_hash=evidence.reference_ids_by_claim_hash,
            defaults=NewMemoryDefaults(
                visibility=Visibility.PRIVATE.value,
                owner_user_id=owner_user_id,
                project_key=project_key,
                repo_identifier=repo_identifier,
                doc_id=concept_id,
                source_type="agent_session",
                access_context_hash=access_hash,
                source_updated_at=(
                    source_updated_at.isoformat() if source_updated_at is not None else None
                ),
            ),
            evidence_units=evidence.units,
            evidence_references=evidence.references,
        )
        memory_id = next(
            (
                mutation.memory_id
                for mutation in plan.mutations
                if mutation.mutation_type is LifecycleMutationType.CREATE_MEMORY
            ),
            incumbent_memory_id,
        )
        if memory_id is None:
            raise ValueError("agent claim lifecycle produced no target Memory")
        return projection, plan, memory_id

    @staticmethod
    def _bind_relation_evidence_to_plan(
        *,
        plan: LifecyclePlan,
        target_memory_id: str,
        intent: _AgentRelationIntent,
    ) -> tuple[LifecyclePlan, EvidenceUnit]:
        """Persist one canonical Evidence Unit for relation and Support.

        Agent patch intent is transient authority metadata. The projected
        Evidence Unit is the durable identity because it owns the exact,
        revision-pinned Evidence References used by Support Assertions.
        """

        support_reference_ids = {
            reference_id
            for mutation in plan.mutations
            if mutation.mutation_type is LifecycleMutationType.ATTACH_SUPPORT
            and mutation.memory_id == target_memory_id
            for reference_id in mutation.evidence_reference_ids
        }
        evidence_unit_ids = {
            reference.evidence_unit_id
            for reference in plan.evidence_references
            if reference.id in support_reference_ids
        }
        if len(evidence_unit_ids) != 1:
            raise ValueError("agent relation requires one revision-pinned projected Evidence Unit")
        evidence_unit_id = next(iter(evidence_unit_ids))
        projected_unit = next(
            (unit for unit in plan.evidence_units if unit.id == evidence_unit_id),
            None,
        )
        if projected_unit is None:
            raise ValueError("agent relation Evidence Unit is missing from the lifecycle plan")
        relation_unit = replace(
            projected_unit,
            client=intent.client,
            source_metadata={
                **dict(projected_unit.source_metadata),
                **dict(intent.source_metadata),
            },
        )
        rebound_plan = replace(
            plan,
            evidence_units=tuple(
                relation_unit if unit.id == evidence_unit_id else unit
                for unit in plan.evidence_units
            ),
        )
        rebound_plan.validate()
        return rebound_plan, relation_unit

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
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        source_type: str,
        replacement_reason: str,
        replacement_kind: ReplacementKind,
        memory_extraction_context: str | None = None,
        submitted_at: datetime,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_markdown_body: str | None = None,
    ) -> str:
        intent = _agent_relation_intent(
            proposal=proposal,
            client=client,
            session_id=session_id,
            workspace=workspace,
            concept_id=concept_id,
            claim_id=claim_id,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
        )
        projection, plan, new_memory_id = await self._build_agent_claim_lifecycle(
            concept_id=concept_id,
            source_id=source_id,
            client=client,
            session_id=session_id,
            workspace=workspace,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
            confidence=confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            submitted_at=submitted_at,
            source_updated_at=source_updated_at,
            concept_projection=None,
            concept_markdown_body=concept_markdown_body,
            incumbent_memory_id=old_memory_id,
            reconcile_action=(
                ReconcileAction.UPDATE
                if replacement_kind == "revision"
                else ReconcileAction.SUPERSEDE
            ),
            reconciliation_reason=replacement_reason,
            memory_extraction_context=memory_extraction_context,
        )
        plan, unit = self._bind_relation_evidence_to_plan(
            plan=plan,
            target_memory_id=new_memory_id,
            intent=intent,
        )
        relation_run_id = _relation_run_id(unit.id, session_id, proposal.action)
        existing_run = await self.db.get_relation_run(relation_run_id)
        if existing_run is not None and existing_run.result_memory_id == old_memory_id:
            committed_candidates = tuple(await self.db.get_relation_candidates(relation_run_id))
            relation_outcome = self._relation_outcome_bundle(
                unit=unit,
                relation_run_id=relation_run_id,
                lifecycle_action=LifecycleAction.SUPERSEDE_MEMORY,
                review_case=None,
                memory_id=old_memory_id,
                candidates=committed_candidates,
                incomplete_mandatory_buckets=existing_run.incomplete_mandatory_buckets,
                candidate_count=existing_run.candidate_count,
                confidence=confidence,
                reason=replacement_reason,
                submitted_at=submitted_at,
            )
            await self.db.record_relation_outcome_bundle(relation_outcome)
            if not await self.db.get_active_memory_support_reference_ids(old_memory_id):
                raise RuntimeError("agent claim replacement retry lacks active Source Projection support")
            current_memory = await self.db.get_memory(old_memory_id)
            if current_memory is None or current_memory.status != "active":
                raise RuntimeError("agent claim replacement retry target is not active")
            await self.memory_store.ensure_agent_claim_memory_projection(
                current_memory,
                doc_id=concept_id,
                source_type=source_type,
                excerpt=claim_text.strip(),
                source_updated_at=source_updated_at,
            )
            return old_memory_id
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
        memory = self._build_claim_memory(
            memory_id=new_memory_id,
            claim_text=claim_text,
            memory_content=memory_content,
            memory_type=memory_type,
            confidence=confidence,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            project_key=project_key,
            extraction_context=memory_extraction_context,
        )
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
            projection,
            plan,
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
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        extraction_context: str | None = None,
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
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=(extraction_context or claim_text).strip(),
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
            ),
            require_configured_source=True,
        )

    async def _render_concept_markdown_with_patch(
        self,
        concept: dict,
        *,
        claim_id: str,
        claim_text: str,
        citations: list[str],
    ) -> str:
        return await render_agent_concept_markdown_with_patch(
            self.db,
            concept,
            claim_id=claim_id,
            claim_text=claim_text,
            citations=citations,
        )

    async def _validated_covered_ids(
        self,
        *,
        proposal: AgentKnowledgePatchProposal,
        owner_user_id: str,
        repo_identifier: str | None,
    ) -> tuple[str | None, str | None]:
        concept_id = proposal.covered_concept_id
        claim_id = proposal.covered_claim_id
        if not concept_id:
            return None, None
        concept = await self.db.get_agent_concept(concept_id)
        if (
            concept is None
            or concept.get("owner_user_id") != owner_user_id
            or concept.get("repo_identifier") != repo_identifier
        ):
            return None, None
        if not claim_id:
            return concept_id, None
        claim = await self.db.get_agent_claim(claim_id)
        if claim is None or claim.get("concept_id") != concept_id:
            return concept_id, None
        return concept_id, claim_id


def render_agent_session_authority_prompt(
    *,
    owner_user_id: str,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    repo_identifier: str | None,
    branch: str | None,
    events: list[dict],
) -> str:
    """Render the semantic classifier prompt for candidate user authority."""
    candidate_lines = []
    context_lines = []
    for event in events:
        event_payload = {
            "evidence_id": event["evidence_id"],
            "kind": event.get("kind", "event"),
            "name": event.get("name"),
            "actor": event.get("actor"),
            "text": event.get("text") or event.get("summary") or "",
        }
        if event.get("authority_candidate"):
            candidate_lines.append(event_payload)
        else:
            context_lines.append(event_payload)
    candidates = json.dumps(candidate_lines, ensure_ascii=False, indent=2)
    context = json.dumps(context_lines, ensure_ascii=False, indent=2)
    operational_context = json.dumps(
        {
            "owner_user_id": owner_user_id,
            "client": client,
            "session_id": session_id,
            "trigger": trigger,
            "workspace": workspace,
            "repo_identifier": repo_identifier,
            "branch": branch,
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""You classify whether explicit user-authored agent-session evidence can authorize durable memory.

Decide semantically, not by keyword matching. A candidate is authoritative only when the user is expressing durable future-facing intent, a stable preference, a design decision, a rule/convention, or explicit approval of such a durable direction.

Not authoritative:
- generic task control such as continue, retry, do it, go ahead, ok, yes, good, or next
- transient requests to test, deploy, debug, explain, inspect, or continue current work
- user messages that only acknowledge assistant progress
- assistant reasoning, tool output, logs, summaries, or implementation narration

Authoritative examples:
- the user asks to remember or keep a rule for future agents
- the user sets a default, convention, source-of-truth boundary, or design policy
- the user explicitly approves a durable design direction, not just the next action

Return exactly one decision for every candidate evidence id and no decisions for non-candidates.

<operational_context_json>
{operational_context}
</operational_context_json>

<candidate_user_evidence_json>
{candidates}
</candidate_user_evidence_json>

<supporting_context_json>
{context}
</supporting_context_json>
"""


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
    """Build the semantic patch prompt for one managed agent-session window."""

    concepts = await db.list_agent_concepts(
        viewer_user_id=owner_user_id,
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
    primary_lines = []
    supporting_lines = []
    for event in events:
        name = event.get("name")
        label = event.get("kind", "event")
        if name:
            label = f"{label}:{name}"
        evidence_id = event["evidence_id"]
        text = event.get("text") or event.get("summary") or json.dumps(event, ensure_ascii=False)
        line = f"[{evidence_id}:{label}] {text}"
        if event.get("evidence_role") == "primary":
            primary_lines.append(line)
        else:
            supporting_lines.append(line)
    primary_evidence = "\n\n".join(primary_lines) or "- none"
    supporting_evidence = "\n\n".join(supporting_lines) or "- none"
    if transcript_markdown.strip():
        supporting_evidence = (
            f"{supporting_evidence}\n\nFull transcript evidence (supporting only):\n"
            f"{transcript_markdown.strip()}"
        ).strip()

    return f"""You are updating a private MemForge agent-memory bundle for one user.

Decision boundary:
- Write only durable preferences, conventions, procedures, decisions, pitfalls, or debugging takeaways.
- Do not summarize ordinary progress, transient status, or facts a future agent can rediscover from the current repo.
- Agent-session memories are private-only in this version.
- If the evidence updates existing durable knowledge, use update_existing_claim. Copy concept_id and claim_id only when the listed match is unambiguous; otherwise leave them null and MemForge will reconcile against memory rows.
- If the evidence replaces or invalidates existing durable knowledge, use supersede_existing_claim. Copy concept_id and claim_id only when the listed match is unambiguous; otherwise leave them null and MemForge will reconcile against memory rows.
- If it belongs in an existing concept but is a distinct durable claim, use add_new_claim and copy the exact concept_id when the listed concept is unambiguous.
- If it is a new durable concept, use create_new_concept with a concise title and concept_type.
- If nothing durable should be kept, use no_output.
- Agent-session memory is user-anchored: non-no_output actions require at least one primary_evidence_ids entry from <primary_evidence>.
- Primary evidence is explicit durable user intent: a user-authored preference, approval, design decision, rule, convention, or instruction to remember something for future work.
- Generic chat control such as "continue", "do it", "retry", or "ok" is supporting context, not durable memory authority.
- Primary evidence authorizes the durable claim. Supporting evidence can explain, qualify, or provide provenance, but Supporting evidence cannot by itself authorize create_new_concept or add_new_claim.
- Intermediate assistant reasoning, self-verification, tool logs, command output, handoff summaries, and deployment narration are supporting evidence only unless a primary user turn authorizes the durable outcome.
- If an existing listed claim, applied to the same situation, already predicts or covers the proposed statement, choose no_output and set covered_concept_id and covered_claim_id when known.
- Use add_new_claim only when the statement is independently checkable and not implied by any listed claim in the same concept.
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
IDs are optional for update_existing_claim and supersede_existing_claim when the durable evidence is clear but the listed context is incomplete.
{existing}
</comparison_context>

<primary_evidence>
Explicit durable user intent that may authorize durable memory. Non-no_output actions must cite one or more IDs from this section in primary_evidence_ids.
{primary_evidence}
</primary_evidence>

<supporting_evidence>
Context, provenance, intermediate reasoning, logs, tests, tool output, or transcript detail. Use this to understand scope and evidence, but do not create memory from this section alone.
{supporting_evidence}
</supporting_evidence>
"""


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must include timezone information")
    return value.astimezone(timezone.utc)


def _agent_relation_intent(
    *,
    proposal: AgentKnowledgePatchProposal,
    client: str,
    session_id: str,
    workspace: str,
    concept_id: str,
    claim_id: str,
    owner_user_id: str,
    repo_identifier: str | None,
) -> _AgentRelationIntent:
    claim_anchor = _claim_anchor(owner_user_id, repo_identifier, concept_id, claim_id)
    return _AgentRelationIntent(
        client=client,
        source_metadata={
            "concept_id": concept_id,
            "claim_id": claim_id,
            "claim_anchor": claim_anchor,
            "source_patch_intent": proposal.action,
            "session_id": session_id,
            "workspace": workspace,
            "reason": proposal.reason,
            "citations": [citation for citation in proposal.citations if citation.strip()],
        },
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
