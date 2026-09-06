"""Reuse scenario judgments when exercising the current revision response contract.

These methods assemble fixture responses, not provider calls. Dedicated model
contract tests assert the actual single-call boundary and complete input shape.
"""

import json

from memforge.llm.structured import (
    ClaimRevisionResponse,
    ClaimRevisionDecision,
    MemoryRelationAssessment,
    RevisionAssessment,
    RevisionSupportResponse,
)


class RevisionClientFixture:
    input_policy_identity = "test-input-policy"

    def request_fits(self, prompt, **kwargs):
        return True

    async def assess_claim_revisions(self, prompt, **kwargs):
        start, end = "<memory_pair_groups>\n", "\n</memory_pair_groups>"
        groups = json.loads(prompt.split(start, 1)[1].split(end, 1)[0])
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
        if requests and hasattr(self, "prove_revision_compositions"):
            response = await self.prove_revision_compositions(
                "<refinement_pairs>" + json.dumps(requests) + "</refinement_pairs>", **kwargs
            )
            proofs = {proof.pair_index: proof for proof in response.decisions}
        return ClaimRevisionResponse(
            decisions=[
                ClaimRevisionDecision(
                    pair_index=relation.pair_index,
                    status="resolved",
                    consistent_with_support=True,
                    relation=MemoryRelationAssessment.model_validate(relation.model_dump(exclude={"pair_index"})),
                    revision_assessment=RevisionAssessment.model_validate(
                        proofs[relation.pair_index].model_dump(exclude={"pair_index", "reason"})
                    )
                    if relation.pair_index in proofs
                    else (
                        RevisionAssessment(
                            same_memory_identity=False,
                            preserves_incumbent_truth=False,
                            candidate_is_canonical_composite=False,
                            current_evidence_entails_candidate=False,
                        )
                        if not hasattr(self, "prove_revision_compositions")
                        and relation.classification == "refines"
                        and relation.direction == "challenger_to_candidate"
                        else None
                    ),
                )
                for relation in relations.decisions
            ]
        )

    async def assess_revision_support(self, prompt, **kwargs):
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        old_prompt = (
            "<incumbents>"
            + json.dumps([{"request_position": 0, "content": payload["claim"], "memory_type": payload["memory_type"]}])
            + "</incumbents>"
        )
        audit = await self.audit_incumbent_support(old_prompt, **kwargs)
        if len(audit.decisions) != 1:
            from memforge.pipeline.reconciler import ReconciliationContractError

            raise ReconciliationContractError(
                "support_response_incomplete", "fixture support response omitted requested claim"
            )
        supported = audit.decisions[0].supported
        current = [*payload["current"]["primary_candidates"], *payload["current"]["required_only_candidates"]]
        previous = payload["previous_evidence"]
        primary_old = next(item for item in previous if item["role"] == "primary")
        primary = next(
            (item for item in current if item["text"] == primary_old["excerpt"]),
            next(
                (item for item in current if item["kind"] != "artifact" and item["type"] != "markdown-heading"),
                current[0],
            ),
        )
        if hasattr(self, "validate_memory_support"):
            refs = {item["ref"]: f"f{index:06d}" for index, item in enumerate(current, 1)}
            reverse = {value: key for key, value in refs.items()}
            legacy = {
                "memory_claim": payload["claim"],
                "previous_primary_quote": primary_old["excerpt"],
                "primary_candidates": [
                    {**item, "ref": refs[item["ref"]]}
                    for item in current
                    if item["observation_id"] == primary_old["observation_id"]
                ],
                "required": [
                    {
                        "selector": f"r{index:06d}",
                        "previous_quote": old["excerpt"],
                        "candidates": [
                            {**item, "ref": refs[item["ref"]]}
                            for item in current
                            if item["observation_id"] == old["observation_id"]
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
            validation = await self.validate_memory_support(legacy_prompt, **kwargs)
            return RevisionSupportResponse(
                status="supported" if validation.supported else "unsupported",
                primary_ref=reverse.get(validation.primary_ref, validation.primary_ref),
                required_refs=[
                    reverse.get(item.evidence_ref, item.evidence_ref) for item in validation.required_evidence
                ],
                reason=validation.reason,
            )
        required = [
            item["ref"]
            for old in previous
            if old["role"] == "required"
            for item in current
            if item["text"] == old["excerpt"]
        ]
        return RevisionSupportResponse(
            status="supported" if supported else "unsupported",
            primary_ref=primary["ref"],
            required_refs=required,
            reason=audit.decisions[0].reason,
        )
