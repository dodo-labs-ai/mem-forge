"""Reuse scenario judgments when exercising the current revision response contract.

These methods assemble fixture responses, not provider calls. Dedicated model
contract tests assert the actual single-call boundary and complete input shape.

Scenarios state their judgments through three optional hooks, each taking a
fixture prompt and returning one of the fixture models below:
``judge_support`` (does the current Source Unit still support an incumbent),
``select_support_evidence`` (which current Fragments carry that support) and
``prove_revisions`` (which REFINES pairs are eligible revisions).
``judge_change_impact`` labels one Change Impact work; it answers ``affected``
unless a scenario overrides it, so an exact Support is read like any other.
"""

import json
from typing import Literal

from pydantic import BaseModel, Field

from memforge.llm.structured import (
    ClaimRevisionDecision,
    MemoryRelationAssessment,
    RevisionAssessment,
)


class FixtureSupport(BaseModel):
    """One scenario Support judgment before it is written as an ordered-reading wire row."""

    status: Literal["supported", "unsupported"]
    primary_ref: str | None = None
    required_refs: list[str] = Field(default_factory=list)


class SupportJudgment(BaseModel):
    supported: bool
    reason: str = ""


class SupportJudgments(BaseModel):
    decisions: list[SupportJudgment]


class RevisionProof(BaseModel):
    pair_index: int
    same_memory_identity: bool
    preserves_incumbent_truth: bool
    candidate_is_canonical_composite: bool
    current_evidence_entails_candidate: bool
    reason: str = ""


class RevisionProofs(BaseModel):
    decisions: list[RevisionProof]


class RequiredSelection(BaseModel):
    selector: str
    evidence_ref: str


class EvidenceSelection(BaseModel):
    supported: bool
    reason: str = ""
    primary_ref: str | None = None
    required_evidence: list[RequiredSelection] = Field(default_factory=list)


def change_impact_payload(prompt):
    return json.loads(prompt.split("<change_impact>", 1)[1].split("</change_impact>", 1)[0])


def change_impact_response(prompt, judge=lambda work, payload: "affected"):
    """One Change Impact row per work, labelled by ``judge(work, payload)``."""
    from memforge.llm.structured import ChangeImpactWireResponse

    payload = change_impact_payload(prompt)
    return ChangeImpactWireResponse.model_validate({"results": [
        {"work_id": work["work_id"], "impact": judge(work, payload)} for work in payload["works"]
    ]})


def catalog_payload(prompt):
    return json.loads(prompt.split("<claim_catalog>\n", 1)[1].split("\n</claim_catalog>", 1)[0])


def legacy_groups(prompt):
    data = catalog_payload(prompt)
    return [dict(challenger={"content": c["text"], "type": c["type"], "valid_from": c["valid_from"], "valid_until": c["valid_until"]},
        current_evidence=[data["evidence_catalog"][ref] for ref in c["evidence_refs"]],
        candidates=[dict(pair_index=i * len(data["existing_claims"]) + j, content=m["text"], type=m["type"],
            valid_from=m["valid_from"], valid_until=m["valid_until"], incumbent_support=m["current_support"])
            for j, m in enumerate(data["existing_claims"])]) for i, c in enumerate(data["new_claims"])]


def sparse_response(prompt, decisions):
    from memforge.llm.structured import ClaimRevisionWireResponse
    data = catalog_payload(prompt)
    rows = [dict(candidate_id=c["id"], evidence_status="entailed", relations=[], uncertain_existing_ids=[]) for c in data["new_claims"]]
    count = len(data["existing_claims"])
    for d in decisions:
        i, j = divmod(d.pair_index, count)
        ref = data["existing_claims"][j]["id"]
        if d.status == "insufficient":
            rows[i]["uncertain_existing_ids"].append(ref)
        elif d.relation.classification != "unrelated":
            relation = d.relation
            rows[i]["relations"].append(dict(existing_id=ref,
                relation="refines_" + relation.direction if relation.classification == "refines" else relation.classification,
                reason=d.reason, revision_assessment=d.revision_assessment,
                contradiction=dict(same_subject_and_scope=relation.same_subject_and_scope,
                    incompatible_assertions=relation.incompatible_assertions) if relation.classification == "contradicts" else None))
    return ClaimRevisionWireResponse.model_validate(dict(results=rows))


class RevisionClientFixture:
    input_policy_identity = "test-input-policy"

    def request_budget(self, model=None):
        from memforge.llm.request_budget import RequestBudget
        return RequestBudget(model or "fixture", 200000, 200000, 64000, 0.8, "fixture")

    def request_fits(self, prompt, **kwargs):
        return True

    def input_policy_identity_for(self, model=None):
        return self.input_policy_identity + str(model)

    def request_tokens(self, prompt, **kwargs):
        return len(prompt)

    async def discover_memory_relations(self, prompt, **kwargs):
        """Adapt existing scenario judgments to the sparse wire fixture."""
        from memforge.llm.structured import MemoryRelationCatalogResponse

        payload = json.loads(prompt.split("<memory_relation_catalog>\n", 1)[1].split(
            "\n</memory_relation_catalog>", 1)[0])
        old = {row["id"]: row for row in payload["existing_claims"]}
        pairs = []
        groups = []
        for challenger in payload["new_claims"]:
            group = dict(challenger=challenger, candidates=[])
            for ref in payload["allowed_existing_ids"][challenger["id"]]:
                group["candidates"].append(dict(pair_index=len(pairs), candidate=old[ref]))
                pairs.append((challenger["id"], ref))
            groups.append(group)
        exact = await self.classify_memory_relations(
            "<memory_pair_groups>\n" + json.dumps(groups) + "\n</memory_pair_groups>", **kwargs)
        rows = {row["id"]: dict(candidate_id=row["id"], relations=[]) for row in payload["new_claims"]}
        indices = [decision.pair_index for decision in exact.decisions]
        if len(indices) != len(set(indices)) or set(indices) != set(range(len(pairs))):
            return MemoryRelationCatalogResponse(results=[])
        for decision in exact.decisions:
            if decision.classification == "unrelated":
                continue
            new_ref, old_ref = pairs[decision.pair_index]
            rows[new_ref]["relations"].append(dict(existing_id=old_ref,
                **decision.model_dump(exclude={"pair_index"})))
        return MemoryRelationCatalogResponse.model_validate(dict(results=list(rows.values())))

    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        """Answer an ordered Support reading from the scenario's ``assess_support`` judgment.

        A work that may not conclude yet continues; a concluding one is supported
        or, once the last reading group is read, unsupported. ``assess_support``
        judges one claim from a scenario payload: the request payload with that
        claim's work fields, its prior Evidence as text, and the carried witnesses
        merged back into the current candidates.
        """
        from memforge.llm.structured import ChangeImpactWireResponse, SupportAssessmentWireResponse

        if response_format is ChangeImpactWireResponse:
            return change_impact_response(prompt, self.judge_change_impact)
        assert response_format is SupportAssessmentWireResponse
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        groups = [{**group, **payload["current"].get("observations", {}).get(group.get("source"), {})} for group in payload["current"]["structural_groups"]]
        sources = {ref: group for group in groups for ref in group["refs"]}
        rows = [
            *payload["current"]["primary_candidates"],
            *payload["current"]["required_only_candidates"],
            *payload["carried_witness_catalog"],
        ]
        texts = {row[0]: row[1] for row in rows}
        results = []
        for work in payload["works"]:
            if not work["may_conclude"]:
                results.append(continued(work))
                continue
            prior = work["prior_evidence"]
            if all("current_ref" in part for part in prior):
                previous = [
                    {"role": part["role"], "excerpt": texts[part["current_ref"]],
                     **_source(sources.get(part["current_ref"]))}
                    for part in prior
                ]
            else:
                previous = self._configured_previous(work, rows, sources, groups)
            scenario_payload = {**payload, **work, "claims": [work], "previous_evidence": previous}
            scenario_payload["current"] = {**payload["current"], "primary_candidates": [
                *payload["current"]["primary_candidates"],
                *(row for row in payload["carried_witness_catalog"] if row[0].startswith("PRM-")),
            ], "required_only_candidates": [
                *payload["current"]["required_only_candidates"],
                *(row for row in payload["carried_witness_catalog"] if row[0].startswith("REQ-")),
            ]}
            assessment_prompt = "<assessment>" + json.dumps(scenario_payload) + "</assessment>"
            if "<correction>" in prompt:
                assessment_prompt += "previous selection used invalid refs"
            result = await self.assess_support(assessment_prompt, **kwargs)
            if result.status == "supported":
                results.append(supported(work, result.primary_ref, result.required_refs))
            elif payload["last"]:
                results.append(unsupported(work))
            else:
                results.append(continued(work))
        return SupportAssessmentWireResponse.model_validate({"results": results})

    def judge_change_impact(self, work, payload):
        return "affected"

    def _configured_previous(self, work, rows, sources, groups):
        """Prior Evidence that is no longer exactly current: follow the scenario's configured quotes."""
        configured_primary = getattr(self, "evidence_quote", "")
        if getattr(self, "prefer_artifact_primary", False):
            primary_row = next(
                (row for row in rows if len(row) > 2 and "image_source_observation_id" in row[2]),
                None,
            )
        else:
            primary_row = next(
                (row for row in rows if configured_primary and configured_primary in row[1]),
                None,
            )
            if primary_row is None:
                primary_row = next((row for row in rows if row[1] == work["claim"]), None)
        if primary_row is None:
            previous = [{"role": "primary", "excerpt": work["claim"], **_source(groups[0] if groups else None)}]
        else:
            previous = [{"role": "primary", "excerpt": primary_row[1], **_source(sources.get(primary_row[0]))}]
        configured_required = tuple(getattr(self, "required_evidence_quotes", ()))
        single_required = getattr(self, "required_evidence_quote", "")
        if single_required:
            configured_required = (*configured_required, single_required)
        for quote in configured_required:
            row = next((row for row in rows if quote in row[1]), None)
            if row is not None:
                previous.append({"role": "required", "excerpt": row[1], **_source(sources.get(row[0]))})
        return previous

    async def assess_claim_revisions(self, prompt, **kwargs):
        start, end = "<memory_pair_groups>\n", "\n</memory_pair_groups>"
        groups = legacy_groups(prompt)
        pairs = [
            {
                "pair_index": old["pair_index"],
                "challenger": group["challenger"],
                "candidate": old,
                "current_evidence": group["current_evidence"],
                "incumbent_support": old["incumbent_support"],
            }
            for group in groups
            for old in group["candidates"]
        ]
        legacy_prompt = prompt.split(start, 1)[0] + start + json.dumps(groups) + end
        relations = await self.classify_memory_relations(legacy_prompt, **kwargs)
        proofs = {}
        requests = [
            pair
            for pair, relation in zip(pairs, relations.decisions)
            if relation.classification == "refines" and relation.direction == "challenger_to_candidate"
        ]
        if requests and hasattr(self, "prove_revisions"):
            response = await self.prove_revisions(
                "<refinement_pairs>" + json.dumps(requests) + "</refinement_pairs>", **kwargs
            )
            proofs = {proof.pair_index: proof for proof in response.decisions}
        return sparse_response(prompt, [
                ClaimRevisionDecision(
                    pair_index=relation.pair_index,
                    status="resolved",
                    consistent_with_support=True,
                    relation=MemoryRelationAssessment.model_validate(relation.model_dump(exclude={"pair_index"})),
                    revision_assessment=RevisionAssessment(
                        same_knowledge_item=proofs[relation.pair_index].same_memory_identity,
                        preserves_incumbent_truth=proofs[relation.pair_index].preserves_incumbent_truth,
                        challenger_is_complete_current_claim=proofs[relation.pair_index].candidate_is_canonical_composite,
                        current_evidence_entails_challenger=proofs[relation.pair_index].current_evidence_entails_candidate,
                    )
                    if relation.pair_index in proofs
                    else (
                        RevisionAssessment(
                            same_knowledge_item=False,
                            preserves_incumbent_truth=False,
                            challenger_is_complete_current_claim=False,
                            current_evidence_entails_challenger=False,
                        )
                        if not hasattr(self, "prove_revisions")
                        and relation.classification == "refines"
                        and relation.direction == "challenger_to_candidate"
                        else None
                    ),
                )
                for relation in relations.decisions
            ])

    async def assess_support(self, prompt, **kwargs):
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        old_prompt = (
            "<incumbents>"
            + json.dumps([{"request_position": 0, "content": payload["claim"], "memory_type": payload["memory_type"]}])
            + "</incumbents>"
        )
        audit = await self.judge_support(old_prompt, **kwargs)
        if len(audit.decisions) != 1:
            from memforge.pipeline.reconciler import ReconciliationContractError

            raise ReconciliationContractError(
                "support_response_incomplete", "fixture support response omitted requested claim"
            )
        supported = audit.decisions[0].supported
        current = [*payload["current"]["primary_candidates"], *payload["current"]["required_only_candidates"]]
        groups = {ref: {**group, **payload["current"].get("observations", {}).get(group.get("source"), {})} for group in payload["current"]["structural_groups"] for ref in group["refs"]}
        current = [
            {"ref": row[0], "text": row[1],
             "kind": "artifact" if len(row) > 2 and "image_source_observation_id" in row[2] else "text",
             "type": row[2].get("format", "artifact") if len(row) > 2 else ("markdown-heading" if row[1].startswith("#") else "text"),
             **_source(groups.get(row[0]))}
            for row in current
        ]
        previous = payload["previous_evidence"]
        primary_old = next(item for item in previous if item["role"] == "primary")
        primary = next(
            (item for item in current if item["text"] == primary_old["excerpt"]),
            next(
                (item for item in current if item["kind"] != "artifact" and item["type"] != "markdown-heading"),
                current[0],
            ),
        )
        if hasattr(self, "select_support_evidence"):
            refs = {item["ref"]: f"f{index:06d}" for index, item in enumerate(current, 1)}
            reverse = {value: key for key, value in refs.items()}
            legacy = {
                "memory_claim": payload["claim"],
                "previous_primary_quote": primary_old["excerpt"],
                "primary_candidates": [
                    {**item, "ref": refs[item["ref"]]}
                    for item in current
                    if primary_old["observation_id"] in {None, item["observation_id"]}
                ],
                "required": [
                    {
                        "selector": f"r{index:06d}",
                        "previous_quote": old["excerpt"],
                        "candidates": [
                            {**item, "ref": refs[item["ref"]]}
                            for item in current
                            if old["observation_id"] in {None, item["observation_id"]}
                        ],
                    }
                    for index, old in enumerate((item for item in previous if item["role"] == "required"), 1)
                ],
            }
            legacy_prompt = "<case_json>\n" + json.dumps(legacy) + "\n</case_json>"
            if "previous selection used invalid refs" in prompt:
                legacy_prompt += (
                    "\n<selection_correction>\n"
                    + json.dumps(
                        {
                            "previous_error": "unknown_ref",
                            "allowed_primary_refs": [item["ref"] for item in legacy["primary_candidates"]],
                            "required": [],
                        }
                    )
                    + "\n</selection_correction>"
                )
            validation = await self.select_support_evidence(legacy_prompt, **kwargs)
            return FixtureSupport(
                status="supported" if validation.supported else "unsupported",
                primary_ref=reverse.get(validation.primary_ref, validation.primary_ref),
                required_refs=[
                    reverse.get(item.evidence_ref, item.evidence_ref) for item in validation.required_evidence
                ],
            )
        required = [
            item["ref"]
            for old in previous
            if old["role"] == "required"
            for item in current
            if item["text"] == old["excerpt"]
        ]
        return FixtureSupport(
            status="supported" if supported else "unsupported",
            primary_ref=primary["ref"],
            required_refs=required,
        )


def continued(work, support=(), opposing=()):
    """A wire row that keeps reading, adding the given witness refs."""
    return {"work_id": work["work_id"], "status": "continue",
            "witness_delta": {"support_witness_refs": list(support), "opposing_witness_refs": list(opposing)}}


def supported(work, primary, required=()):
    """A wire row that concludes supported, omitting every matched prior ref it does not select."""
    selected = {primary, *required}
    return {
        "work_id": work["work_id"], "status": "supported", "primary_ref": primary, "required_refs": list(required),
        "omitted_matched_refs": [
            p["current_ref"] for p in work["prior_evidence"] if "current_ref" in p and p["current_ref"] not in selected
        ],
    }


def unsupported(work):
    return {"work_id": work["work_id"], "status": "unsupported"}


def _source(group):
    """Observation identity of a supplied ref; carried and historical text have none."""
    return {"observation_id": group.get("observation_id") if group else None,
            "revision_id": group.get("revision_id") if group else None}
