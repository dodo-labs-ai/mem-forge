from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memforge.llm.relation_catalog import RelationCoverage, RequestCatalog, request_ref
from memforge.llm.structured import MemoryRelationCatalogResponse, StructuredLlmError
from memforge.memory.relation_classifier import MemoryPair, MemoryPairClassificationError
from memforge.memory.relation_classifier import MEMORY_RELATION_PROMPT, _grouped_pair_payload
from memforge.memory.sparse_relation_classifier import SparseMemoryRelationClassifier
from memforge.models import Memory, content_hash
from tests.revision_client_fixture import RevisionClientFixture


def memory(id, text):
    return Memory(id=id, content=text, content_hash=content_hash(text), memory_type="fact")


def payload(prompt):
    return json.loads(prompt.split("<memory_relation_catalog>\n", 1)[1].split(
        "\n</memory_relation_catalog>", 1)[0])


class Client(RevisionClientFixture):
    def __init__(self, response=None):
        self.prompts = []
        self.response = response

    async def discover_memory_relations(self, prompt, **kwargs):
        self.prompts.append(prompt)
        data = payload(prompt)
        if self.response is not None:
            return self.response(data)
        return dict(results=[dict(candidate_id=c["id"], relations=[]) for c in data["new_claims"]])


def edge(ref, **kwargs):
    return dict(existing_id=ref, classification="equivalent", direction="symmetric",
                same_subject_and_scope=False, incompatible_assertions="", reason="same proposition", **kwargs)


@pytest.mark.asyncio
async def test_five_by_ten_becomes_fifteen_records_and_five_completions():
    new = [memory(f"new-{i}", f"new text {i}") for i in range(5)]
    old = [memory(f"old-{i}", f"old text {i}") for i in range(10)]
    pairs = tuple(MemoryPair(n, o) for n in new for o in old)
    client = Client()
    classifier = SparseMemoryRelationClassifier(client=client, model="fixture")
    result = await classifier.classify(pairs)
    data = payload(client.prompts[0])
    assert result.decisions == ()
    assert result.llm_calls == 1
    assert len(data["new_claims"]) == 5 and len(data["existing_claims"]) == 10
    assert sum(len(claim["allowed_existing_ids"]) for claim in data["new_claims"]) == 50
    assert data["new_claims"][0]["id"] == "NEW-0001"
    assert data["existing_claims"][-1]["id"] == "MEM-0010"
    assert "pair_index" not in client.prompts[0]
    assert all(client.prompts[0].count(o.content) == 1 for o in old)


@pytest.mark.asyncio
async def test_only_explicit_edges_return_and_pair_order_is_preserved():
    pairs = (MemoryPair(memory("n1", "cutoff Friday"), memory("m1", "cutoff Friday")),
             MemoryPair(memory("n2", "pay slips PDF"), memory("m2", "dark mode")))
    client = Client(lambda data: dict(results=[
        dict(candidate_id="NEW-0002", relations=[]),
        dict(candidate_id="NEW-0001", relations=[edge("MEM-0001")]),
    ]))
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    assert [d.pair.key for d in result.decisions] == [("n1", "m1")]


@pytest.mark.asyncio
async def test_four_discovered_edges_replace_fifty_dense_decisions():
    pairs = tuple(MemoryPair(memory(f"n{i}", f"new claim {i}"), memory(f"m{j}", f"old claim {j}"))
                  for i in range(5) for j in range(10))
    wire = {}
    def response(data):
        wire.update(results=[dict(candidate_id=c["id"], relations=[edge(f"MEM-{i+1:04d}")]
                     if i < 4 else []) for i, c in enumerate(data["new_claims"])])
        return wire
    client = Client(response)
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    assert len(result.decisions) == 4 and len(wire["results"]) == 5
    dense = dict(decisions=[dict(pair_index=i, classification="unrelated", direction="symmetric",
        same_subject_and_scope=False, incompatible_assertions="", reason="no discovered relationship") for i in range(50)])
    assert len(json.dumps(wire)) < len(json.dumps(dense)) * 0.4
    baseline = MEMORY_RELATION_PROMPT.format(groups_json=_grouped_pair_payload(tuple(enumerate(pairs))))
    assert len(client.prompts[0]) < len(baseline)


@pytest.mark.asyncio
async def test_capacity_subdivision_keeps_the_complete_allowed_workset():
    pairs = tuple(MemoryPair(memory("n", "new"), memory(f"m{i}", f"old {i}")) for i in range(5))
    class CapacityClient(Client):
        def request_fits(self, prompt, **kwargs):
            return len(payload(prompt)["existing_claims"]) <= 2
    client = CapacityClient()
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    assert result.decisions == () and result.llm_calls == 3
    assert sorted(m["content"] for prompt in client.prompts for m in payload(prompt)["existing_claims"]) == [f"old {i}" for i in range(5)]


@pytest.mark.asyncio
async def test_conflicting_catalog_snapshot_is_rejected():
    pairs = (MemoryPair(memory("n1", "A"), memory("same-id", "shared prefix first")),
             MemoryPair(memory("n2", "B"), memory("same-id", "shared prefix second")))
    client = Client()
    with pytest.raises(MemoryPairClassificationError, match="conflicting catalog snapshots"):
        await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    assert client.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "duplicate_row", "unknown_new", "unknown_old", "outside_allowed", "duplicate_edge"])
async def test_invalid_completion_never_becomes_empty_success(kind):
    bad, good = (MemoryPair(memory("n1", "A"), memory("m1", "B")),
                 MemoryPair(memory("n2", "C"), memory("m2", "D")))

    def response(data):
        """Every request that carries claim A answers it invalidly."""
        rows = [dict(candidate_id=c["id"], relations=[]) for c in data["new_claims"]]
        [row] = [row for row, claim in zip(rows, data["new_claims"]) if claim["content"] == "A"] or [None]
        if row is None:
            return dict(results=rows)
        if kind == "missing":
            rows.remove(row)
        elif kind == "duplicate_row":
            rows.append(row)
        elif kind == "unknown_new":
            row["candidate_id"] = "NEW-9999"
        else:
            refs = ["MEM-9999"] if kind == "unknown_old" else ["MEM-0002"] if kind == "outside_allowed" else ["MEM-0001"] * 2
            row["relations"] = [edge(ref) for ref in refs]
        return dict(results=rows)

    client = Client(response)
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify((bad, good))

    # The pair request and its correction, then A alone with its correction, then C alone.
    assert result.decisions == () and result.llm_calls == 5
    [unjudged] = result.unjudged
    assert unjudged.pair == bad
    assert (unjudged.failure.category, unjudged.failure.error_code) == ("invalid_response", "output_invalid")
    assert "<correction>" in client.prompts[1]


@pytest.mark.asyncio
async def test_malformed_output_for_one_pair_leaves_only_that_pair_unjudged():
    bad, good = (MemoryPair(memory("n1", "A"), memory("m1", "B")),
                 MemoryPair(memory("n2", "C"), memory("m2", "D")))
    malformed = StructuredLlmError(
        "ambiguous structured JSON objects", terminal_category="invalid_response", error_code="ValueError",
    )

    class Malformed(Client):
        async def discover_memory_relations(self, prompt, **kwargs):
            if any(claim["content"] == "A" for claim in payload(prompt)["new_claims"]):
                self.prompts.append(prompt)
                raise malformed
            return await super().discover_memory_relations(prompt, **kwargs)

    client = Malformed(lambda data: dict(results=[
        dict(candidate_id=c["id"], relations=[edge(c["allowed_existing_ids"][0])]) for c in data["new_claims"]]))
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify((bad, good))

    assert [decision.pair.key for decision in result.decisions] == [good.key]
    [unjudged] = result.unjudged
    assert (unjudged.pair, unjudged.failure.error) == (bad, malformed)
    assert result.llm_calls == 3


@pytest.mark.asyncio
async def test_each_new_claim_carries_its_own_allowed_existing_ids():
    shared = memory("m1", "shared old claim")
    pairs = (MemoryPair(memory("n1", "A"), shared), MemoryPair(memory("n1", "A"), memory("m2", "B")),
             MemoryPair(memory("n2", "C"), shared))
    client = Client()
    await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    data = payload(client.prompts[0])
    assert [(claim["id"], claim["allowed_existing_ids"]) for claim in data["new_claims"]] == [
        ("NEW-0001", ["MEM-0001", "MEM-0002"]), ("NEW-0002", ["MEM-0001"]),
    ]
    assert "allowed_existing_ids" not in data


@pytest.mark.asyncio
async def test_one_coverage_correction_uses_original_allowed_catalog():
    pair = MemoryPair(memory("n1", "A"), memory("m1", "B"))
    client = Client()
    def response(data):
        if len(client.prompts) == 1:
            return dict(results=[])
        return dict(results=[dict(candidate_id="NEW-0001", relations=[])])
    client.response = response
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify((pair,))
    assert result.decisions == () and result.llm_calls == 2
    assert payload(client.prompts[0]) == payload(client.prompts[1])


@pytest.mark.asyncio
async def test_provider_failure_is_not_replaced_by_completion():
    class Failing(Client):
        async def discover_memory_relations(self, prompt, **kwargs):
            raise StructuredLlmError("deadline", error_code="deadline_exceeded", terminal_category="deadline_exceeded")
    with pytest.raises(MemoryPairClassificationError) as error:
        await SparseMemoryRelationClassifier(client=Failing(), model="fixture").classify((
            MemoryPair(memory("n", "A"), memory("m", "B")),))
    assert error.value.error_code == "deadline_exceeded"
    assert error.value.terminal_category == "deadline_exceeded"


@pytest.mark.asyncio
async def test_timeout_halves_the_catalog_until_every_pair_completes():
    pairs = tuple(MemoryPair(memory(f"n{i}", f"new {i}"), memory(f"m{i}", f"old {i}")) for i in range(4))

    class SlowForLargeCatalogs(Client):
        async def discover_memory_relations(self, prompt, **kwargs):
            if len(payload(prompt)["new_claims"]) > 1:
                self.prompts.append(prompt)
                raise StructuredLlmError("deadline", error_code="logical_deadline_exceeded",
                                         terminal_category="deadline_exceeded")
            return await super().discover_memory_relations(prompt, **kwargs)

    client = SlowForLargeCatalogs(lambda data: dict(results=[
        dict(candidate_id=c["id"], relations=[edge("MEM-0001")]) for c in data["new_claims"]]))
    result = await SparseMemoryRelationClassifier(client=client, model="fixture").classify(pairs)
    assert [d.pair.key for d in result.decisions] == [pair.key for pair in pairs]
    assert result.llm_calls == 7


def test_prefixes_overflow_and_conflicting_snapshots():
    for prefix in ("NEW", "MEM", "PRM", "REQ", "WRK"):
        assert request_ref(prefix, 1) == prefix + "-0001"
        assert request_ref(prefix, 9999) == prefix + "-9999"
    for prefix, ordinal in (("N", 1), ("new", 1), ("NEW", 0), ("NEW", 10000)):
        with pytest.raises(ValueError):
            request_ref(prefix, ordinal)
    catalog = RequestCatalog("MEM")
    assert catalog.add("id", {"text": "A"}) == catalog.add("id", {"text": "A"})
    with pytest.raises(ValueError, match="conflicting"):
        catalog.add("id", {"text": "B"})


def test_uncertain_refs_share_the_same_duplicate_and_allowed_validation():
    coverage = RelationCoverage({"NEW-0001": frozenset({"MEM-0001"})})
    coverage.validate([SimpleNamespace(candidate_id="NEW-0001", relations=[], uncertain_existing_ids=["MEM-0001"])])
    with pytest.raises(ValueError, match="names MEM-0001 more than once"):
        coverage.validate([SimpleNamespace(candidate_id="NEW-0001", relations=[SimpleNamespace(existing_id="MEM-0001")], uncertain_existing_ids=["MEM-0001"])])


@pytest.mark.parametrize("overrides", [dict(classification="unrelated"), dict(classification="refines"),
    dict(classification="contradicts", same_subject_and_scope=False, incompatible_assertions="Friday versus Thursday")])
def test_sparse_schema_retains_direction_and_conflict_proofs(overrides):
    data = edge("MEM-0001")
    data.update(overrides)
    if overrides["classification"] == "unrelated":
        # The shape itself has no unrelated edge.
        with pytest.raises(ValueError):
            MemoryRelationCatalogResponse.model_validate(dict(results=[dict(candidate_id="NEW-0001", relations=[data])]))
        return
    response = MemoryRelationCatalogResponse.model_validate(dict(results=[dict(candidate_id="NEW-0001", relations=[data])]))
    assert response.results[0].relations[0].row_error() is not None
