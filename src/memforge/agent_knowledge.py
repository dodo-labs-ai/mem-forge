"""Private Agent Knowledge Bundle patching.

Agent-session clients upload evidence windows. This module owns the service-side
patch boundary that turns a structured patch proposal into private, stable
concept claims and then into searchable memories.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memforge.models import (
    DocumentRecord,
    Memory,
    Visibility,
    content_hash,
    generate_memory_id,
    slugify,
)


PatchAction = Literal[
    "create_new_concept",
    "update_existing_claim",
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


class AgentKnowledgePatchProposal(BaseModel):
    """Validated LLM proposal. The service validates scope before applying it."""

    model_config = ConfigDict(extra="ignore")

    action: PatchAction
    concept_id: str | None = None
    claim_id: str | None = None
    concept_type: Literal[
        "preference",
        "convention",
        "procedure",
        "debugging_takeaway",
        "decision",
        "pitfall",
    ] | None = None
    title: str | None = None
    claim_text: str = ""
    memory_type: Literal["fact", "decision", "convention", "procedure"] = "fact"
    tags: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentKnowledgePatchResult:
    outcome: PatchOutcome
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
    ) -> AgentKnowledgePatchResult:
        """Apply one structured patch proposal.

        V1 is private-only. Existing concept/claim writes must belong to
        ``owner_user_id`` and the same ``repo_identifier``.
        """

        submitted_at = _utc(submitted_at)
        if proposal.action == "no_output":
            return AgentKnowledgePatchResult(
                outcome="skipped_not_memory",
                reason=proposal.reason or "proposal returned no_output",
            )
        if not proposal.claim_text.strip():
            return AgentKnowledgePatchResult(
                outcome="skipped_not_memory",
                reason="claim_text is required",
            )

        if proposal.action == "create_new_concept":
            if not proposal.title or not proposal.concept_type:
                return AgentKnowledgePatchResult(
                    outcome="parse_failed",
                    reason="create_new_concept requires title and concept_type",
                )
            concept_id = proposal.concept_id or _new_id("akb_concept")
            claim_id = proposal.claim_id or _new_id("akb_claim")
            memory_id = generate_memory_id()
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
                markdown_body=_render_markdown(
                    title=proposal.title,
                    concept_type=proposal.concept_type,
                    repo_identifier=repo_identifier,
                    claim_id=claim_id,
                    claim_text=proposal.claim_text,
                    citations=proposal.citations,
                ),
            )
            await self.db.upsert_agent_concept(
                concept_id=concept_id,
                source_id=source_id,
                owner_user_id=owner_user_id,
                workspace=workspace,
                repo_identifier=repo_identifier,
                concept_type=proposal.concept_type,
                concept_path=_concept_path(owner_user_id, repo_identifier, proposal.concept_type, proposal.title),
                title=proposal.title,
                markdown_body=_render_markdown(
                    title=proposal.title,
                    concept_type=proposal.concept_type,
                    repo_identifier=repo_identifier,
                    claim_id=claim_id,
                    claim_text=proposal.claim_text,
                    citations=proposal.citations,
                ),
                frontmatter={
                    "visibility": Visibility.PRIVATE.value,
                    "owner_user_id": owner_user_id,
                    "repo_identifier": repo_identifier,
                    "source_id": source_id,
                    "source_type": "agent_session",
                },
                observed_at=submitted_at,
            )
            await self._insert_claim_memory(
                memory_id=memory_id,
                concept_id=concept_id,
                claim_text=proposal.claim_text,
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
                source_type="agent_session",
            )
            await self.db.upsert_agent_claim(
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=slugify(proposal.title),
                claim_text=proposal.claim_text.strip(),
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                memory_id=memory_id,
                observed_at=submitted_at,
            )
            await self._append_citations(claim_id, proposal.citations, submitted_at)
            return AgentKnowledgePatchResult(
                outcome="applied",
                concept_id=concept_id,
                claim_id=claim_id,
                memory_id=memory_id,
            )

        if proposal.action not in {"update_existing_claim", "add_new_claim"}:
            return AgentKnowledgePatchResult(outcome="parse_failed", reason="unsupported action")

        concept = await self.db.get_agent_concept(proposal.concept_id or "")
        if not self._can_patch_concept(concept, owner_user_id, repo_identifier):
            return AgentKnowledgePatchResult(
                outcome="rejected_scope",
                concept_id=proposal.concept_id,
                claim_id=proposal.claim_id,
                reason="private concept is outside the caller scope",
            )

        if proposal.action == "add_new_claim":
            claim_id = proposal.claim_id or _new_id("akb_claim")
            memory_id = generate_memory_id()
            await self._insert_claim_memory(
                memory_id=memory_id,
                concept_id=concept["id"],
                claim_text=proposal.claim_text,
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
                project_key=project_key,
                source_type="agent_session",
            )
            await self.db.upsert_agent_claim(
                claim_id=claim_id,
                concept_id=concept["id"],
                display_anchor=slugify(proposal.claim_text)[:80],
                claim_text=proposal.claim_text.strip(),
                memory_type=proposal.memory_type,
                tags=proposal.tags,
                confidence=proposal.confidence,
                memory_id=memory_id,
                observed_at=submitted_at,
            )
            await self._append_citations(claim_id, proposal.citations, submitted_at)
            await self._refresh_concept_markdown(concept["id"], observed_at=submitted_at)
            return AgentKnowledgePatchResult(
                outcome="applied",
                concept_id=concept["id"],
                claim_id=claim_id,
                memory_id=memory_id,
            )

        claim = await self.db.get_agent_claim(proposal.claim_id or "")
        if not claim or claim["concept_id"] != concept["id"]:
            return AgentKnowledgePatchResult(
                outcome="rejected_scope",
                concept_id=concept["id"],
                claim_id=proposal.claim_id,
                reason="claim is outside the target concept",
            )

        await self.db.upsert_agent_claim(
            claim_id=claim["id"],
            concept_id=concept["id"],
            display_anchor=claim["display_anchor"],
            claim_text=proposal.claim_text.strip(),
            memory_type=proposal.memory_type,
            tags=proposal.tags,
            confidence=proposal.confidence,
            memory_id=claim["memory_id"],
            observed_at=submitted_at,
        )
        await self.memory_store.update_memory(
            claim["memory_id"],
            proposal.claim_text.strip(),
            proposal.confidence,
            proposal.tags,
        )
        await self._append_citations(claim["id"], proposal.citations, submitted_at)
        await self._refresh_concept_markdown(concept["id"], observed_at=submitted_at)
        return AgentKnowledgePatchResult(
            outcome="applied",
            concept_id=concept["id"],
            claim_id=claim["id"],
            memory_id=claim["memory_id"],
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
        memory_id: str,
        concept_id: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        owner_user_id: str,
        repo_identifier: str | None,
        project_key: str | None,
        source_type: str,
    ) -> None:
        memory = Memory(
            id=memory_id,
            memory_type=memory_type,
            content=claim_text.strip(),
            content_hash=content_hash(claim_text.strip()),
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
        await self.memory_store.insert_memory(
            memory=memory,
            doc_id=concept_id,
            source_type=source_type,
            excerpt=claim_text.strip(),
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

    async def _append_citations(
        self,
        claim_id: str,
        citations: list[str],
        observed_at: datetime,
    ) -> None:
        for citation in citations:
            if citation.strip():
                await self.db.add_agent_claim_citation(
                    claim_id=claim_id,
                    citation_url=citation.strip(),
                    observed_at=observed_at,
                )

    async def _refresh_concept_markdown(self, concept_id: str, *, observed_at: datetime) -> None:
        concept = await self.db.get_agent_concept(concept_id)
        if not concept:
            return
        claims = await self.db.list_agent_claims(concept_id)
        citations_by_claim = {
            claim["id"]: await self.db.list_agent_claim_citations(claim["id"])
            for claim in claims
        }
        markdown = _render_markdown(
            title=concept["title"],
            concept_type=concept["concept_type"],
            repo_identifier=concept.get("repo_identifier"),
            claim_id=claims[0]["id"] if claims else "",
            claim_text="\n\n".join(claim["claim_text"] for claim in claims),
            citations=[
                citation["citation_url"]
                for claim in claims
                for citation in citations_by_claim[claim["id"]]
            ],
        )
        await self.db.update_agent_concept_markdown(
            concept_id=concept_id,
            markdown_body=markdown,
            observed_at=observed_at,
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
        claim_lines = [
            f"    - claim_id={claim['id']} :: {claim['claim_text']}"
            for claim in claims[:5]
        ]
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
        evidence = f"{evidence}\n\nTranscript fallback:\n{transcript_markdown.strip()}".strip()

    return f"""You are updating a private MemForge agent-memory bundle for one user.

Decision boundary:
- Write only durable preferences, conventions, procedures, decisions, pitfalls, or debugging takeaways.
- Do not summarize ordinary progress, transient status, or facts a future agent can rediscover from the current repo.
- Agent-session memories are private-only in this version.
- If the evidence updates an existing claim, use update_existing_claim and copy the exact concept_id and claim_id.
- If it belongs in an existing concept but is a distinct durable claim, use add_new_claim and copy the exact concept_id.
- If it is a new durable concept, use create_new_concept with a concise title and concept_type.
- If nothing durable should be kept, use no_output.

Caller:
- owner_user_id: {owner_user_id}
- client: {client}
- session_id: {session_id}
- trigger: {trigger}
- workspace: {workspace}
- repo_identifier: {repo_identifier or "none"}
- branch: {branch or "none"}

Existing private concepts for this user and repo:
{existing}

Canonical evidence:
{evidence or "- no evidence"}
"""


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


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
