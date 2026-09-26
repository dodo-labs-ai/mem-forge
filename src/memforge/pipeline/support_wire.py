"""Typed presentation aliases; canonical Evidence identities stay inside the pipeline."""

from copy import deepcopy

from memforge.llm.structured import (
    ChangeImpactWireResponse,
    ContinueReadingWireResult,
    SupportAssessmentWireResponse,
    SupportedWireResult,
    SupportWitnessDelta,
)
from memforge.pipeline.projection_fragments import FragmentSelectionError, FragmentSelectionErrorCode


class SupportWireAliases:
    """Keep task and Evidence names stable across one revision's Change Impact and Support reading."""

    def __init__(self, catalog, removed, works):
        self.works = works
        # Preserve the catalog's numeric identity across subsets and requests.
        self.refs = {
            part.reference: f"{'PRM' if part.primary_eligible else 'REQ'}-{int(part.reference[1:]):04d}"
            for part in catalog.fragments
        }
        self.history = {part['ref']: f"HIS-{index:04d}" for index, part in enumerate(removed)}
        self._primary_ids = {self.refs[part.reference] for part in catalog.fragments if part.primary_eligible}
        self._work_ids = {alias: original for original, alias in works.items()}
        self._ref_ids = {alias: original for original, alias in self.refs.items()}

    def _encode_source(self, payload, row_lists):
        """Translate the identifier fields of the supplied source text, never the text itself."""
        result = deepcopy(payload)
        current = result['current']
        for key in row_lists:
            current[key] = [(self.refs[row[0]], *row[1:]) for row in current[key]]
        for group in current['structural_groups']:
            group['refs'] = [self.refs[ref] for ref in group['refs']]
        result['removed_historical'] = [
            [self.history[row[0]], *row[1:]] for row in result['removed_historical']
        ]
        for group in result['removed_groups']:
            group['refs'] = [self.history[ref] for ref in group['refs']]
        return result

    def encode(self, payload):
        """An ordered Support reading request: source, carried witnesses and per-work state."""
        result = self._encode_source(payload, ('primary_candidates', 'required_only_candidates'))
        result['carried_witness_catalog'] = [
            (self.refs[row[0]], *row[1:]) for row in result['carried_witness_catalog']
        ]
        for work in result['works']:
            work['work_id'] = self.works[work['work_id']]
            for part in work['prior_evidence']:
                if 'current_ref' in part:
                    part['current_ref'] = self.refs[part['current_ref']]
            state = work['previous_state']
            for key in ('support_witness_refs', 'opposing_witness_refs'):
                state[key] = [self.refs[ref] for ref in state[key]]
        return result

    def encode_changes(self, payload):
        """A Change Impact request: the changes, which of their refs changed, and the works."""
        result = self._encode_source(payload, ('fragments',))
        result['changed_refs'] = [self.refs[ref] for ref in result['changed_refs']]
        for work in result['works']:
            work['work_id'] = self.works[work['work_id']]
        return result

    def _ref(self, alias, location, *, allowed=None):
        allowed = self._ref_ids.keys() if allowed is None else allowed
        if alias not in allowed:
            if alias in self._ref_ids:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.INELIGIBLE_ROLE, f"Evidence ID is not eligible as Primary: {alias}",
                    location=location, received=alias, allowed_refs=sorted(allowed),
                )
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.UNKNOWN_REF, f"unknown supplied Evidence ID: {alias}",
                location=location, received=alias, allowed_refs=sorted(allowed),
            )
        return self._ref_ids[alias]

    def _refs(self, aliases, location):
        return [self._ref(alias, f"{location}[{index}]") for index, alias in enumerate(aliases)]

    def _work(self, alias, location):
        if alias not in self._work_ids:
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.UNKNOWN_REF, f"unknown supplied task ID: {alias}",
                location=location, received=alias, allowed_refs=sorted(self._work_ids),
            )
        return self._work_ids[alias]

    def decode_impacts(self, response: ChangeImpactWireResponse) -> list[tuple[str, str]]:
        """Each row's canonical work id with its impact label; a row for an unknown work is not an answer."""
        return [(self._work_ids[row.work_id], row.impact) for row in response.results if row.work_id in self._work_ids]

    def decode(self, response):
        """Return the same response with canonical work ids and catalog refs; any invalid row raises."""
        return SupportAssessmentWireResponse(
            results=[self._decode_row(row, f"results[{index}]") for index, row in enumerate(response.results)],
        )

    def decode_rows(self, response):
        """Each row of a known work, decoded alone: its canonical work id and the decoded row or its error."""
        for index, row in enumerate(response.results):
            if row.work_id not in self._work_ids:
                continue
            try:
                yield self._work_ids[row.work_id], self._decode_row(row, f"results[{index}]")
            except FragmentSelectionError as error:
                yield self._work_ids[row.work_id], error

    def _decode_row(self, row, at: str):
        work_id = self._work(row.work_id, f"{at}.work_id")
        if isinstance(row, ContinueReadingWireResult):
            delta = row.witness_delta
            return row.model_copy(update={"work_id": work_id, "witness_delta": SupportWitnessDelta(
                support_witness_refs=self._refs(delta.support_witness_refs, f"{at}.support_witness_refs"),
                opposing_witness_refs=self._refs(delta.opposing_witness_refs, f"{at}.opposing_witness_refs"),
            )})
        if isinstance(row, SupportedWireResult):
            return row.model_copy(update={
                "work_id": work_id,
                "primary_ref": self._ref(row.primary_ref, f"{at}.primary_ref", allowed=self._primary_ids),
                "required_refs": self._refs(row.required_refs, f"{at}.required_refs"),
            })
        return row.model_copy(update={"work_id": work_id})
