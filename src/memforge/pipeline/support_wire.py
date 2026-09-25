"""Typed presentation aliases; canonical Evidence identities stay inside the pipeline."""

from copy import deepcopy

from memforge.llm.structured import (
    ContinueReadingWireResult,
    SupportAssessmentWireResponse,
    SupportedWireResult,
    SupportWitnessDelta,
)
from memforge.pipeline.projection_fragments import FragmentSelectionError, FragmentSelectionErrorCode


class SupportWireAliases:
    """Keep task and Evidence names stable across one ordered Support reading."""

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

    def encode(self, payload):
        """Translate identifier fields only, never source text."""
        result = deepcopy(payload)
        current = result['current']
        for key in ('primary_candidates', 'required_only_candidates'):
            current[key] = [(self.refs[row[0]], *row[1:]) for row in current[key]]
        for group in current['structural_groups']:
            group['refs'] = [self.refs[ref] for ref in group['refs']]
        result['carried_witness_catalog'] = [
            (self.refs[row[0]], *row[1:]) for row in result['carried_witness_catalog']
        ]
        result['removed_historical'] = [
            [self.history[row[0]], *row[1:]] for row in result['removed_historical']
        ]
        for group in result['removed_groups']:
            group['refs'] = [self.history[ref] for ref in group['refs']]
        for work in result['works']:
            work['work_id'] = self.works[work['work_id']]
            for part in work['prior_evidence']:
                if 'current_ref' in part:
                    part['current_ref'] = self.refs[part['current_ref']]
            state = work['previous_state']
            for key in ('support_witness_refs', 'opposing_witness_refs'):
                state[key] = [self.refs[ref] for ref in state[key]]
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

    def decode(self, response):
        """Return the same response with canonical work ids and catalog refs."""
        rows = []
        for index, row in enumerate(response.results):
            at = f"results[{index}]"
            if row.work_id not in self._work_ids:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.UNKNOWN_REF, f"unknown supplied task ID: {row.work_id}",
                    location=f"{at}.work_id", received=row.work_id, allowed_refs=sorted(self._work_ids),
                )
            work_id = self._work_ids[row.work_id]
            if isinstance(row, ContinueReadingWireResult):
                delta = row.witness_delta
                rows.append(row.model_copy(update={"work_id": work_id, "witness_delta": SupportWitnessDelta(
                    support_witness_refs=self._refs(delta.support_witness_refs, f"{at}.support_witness_refs"),
                    opposing_witness_refs=self._refs(delta.opposing_witness_refs, f"{at}.opposing_witness_refs"),
                )}))
            elif isinstance(row, SupportedWireResult):
                rows.append(row.model_copy(update={
                    "work_id": work_id,
                    "primary_ref": self._ref(row.primary_ref, f"{at}.primary_ref", allowed=self._primary_ids),
                    "required_refs": self._refs(row.required_refs, f"{at}.required_refs"),
                    "omitted_matched_refs": self._refs(row.omitted_matched_refs, f"{at}.omitted_matched_refs"),
                }))
            else:
                rows.append(row.model_copy(update={"work_id": work_id}))
        return SupportAssessmentWireResponse(results=rows)
