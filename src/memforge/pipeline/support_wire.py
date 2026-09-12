"""Typed presentation aliases; canonical Evidence identities stay inside the pipeline."""

from copy import deepcopy

from memforge.llm.structured import SupportAssessmentResponse, SupportAssessmentResult

from memforge.pipeline.projection_fragments import FragmentSelectionError, FragmentSelectionErrorCode


class SupportWireAliases:
    """Keep task and Evidence names stable across one cumulative assessment."""

    def __init__(self, catalog, removed, works):
        self.works = works
        # Preserve the catalog's numeric identity across subsets and batches.
        self.refs = {
            part.reference: f"{'PRM' if part.primary_eligible else 'REQ'}-{int(part.reference[1:]):04d}"
            for part in catalog.fragments
        }
        self.history = {part['ref']: f"HIS-{index:04d}" for index, part in enumerate(removed)}
        self._primary_ids = {self.refs[part.reference] for part in catalog.fragments if part.primary_eligible}
        self._work_ids = {alias: original for original, alias in works.items()}
        self._ref_ids = {alias: original for original, alias in self.refs.items()}

    def _state(self, row):
        row['work_id'] = self.works[row['work_id']]
        if row.get('primary_ref') is not None:
            row['primary_ref'] = self.refs[row['primary_ref']]
        if 'required_refs' in row:
            row['required_refs'] = [self.refs[ref] for ref in row['required_refs']]

    def encode(self, payload):
        """Translate identifier fields only, never source text or model reasons."""
        result = deepcopy(payload)
        for key in ('claims', 'previous_state'):
            for row in result[key]:
                self._state(row)
        current = result['current']
        for key in ('primary_candidates', 'required_only_candidates'):
            current[key] = [(self.refs[row[0]], *row[1:]) for row in current[key]]
        for group in current['structural_groups']:
            group['refs'] = [self.refs[ref] for ref in group['refs']]
        result['removed_historical'] = [
            [self.history[row[0]], *row[1:]] for row in result['removed_historical']
        ]
        for group in result['removed_groups']:
            group['refs'] = [self.history[ref] for ref in group['refs']]
        for row in result.get('previous_evidence', []):
            row['work_id'] = self.works[row['work_id']]
            for part in row['parts']:
                if 'current_ref' in part:
                    part['current_ref'] = self.refs[part['current_ref']]
        return result

    @staticmethod
    def _resolve(mapping, alias):
        if alias not in mapping:
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.UNKNOWN_REF, f"unknown supplied task/Evidence ID: {alias}"
            )
        return mapping[alias]

    def decode(self, response):
        rows = []
        for row in response.results:
            if row.status == 'supported' and row.primary_ref in self._ref_ids and row.primary_ref not in self._primary_ids:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.INELIGIBLE_ROLE,
                    f"Evidence ID is not eligible as Primary: {row.primary_ref}",
                )
            rows.append(SupportAssessmentResult.model_validate({**row.model_dump(),
                'reason': getattr(row, 'reason', 'Supported by selected current Evidence.'),
                'work_id': self._resolve(self._work_ids, row.work_id),
                'primary_ref': self._resolve(self._ref_ids, row.primary_ref) if row.primary_ref is not None else None,
                'required_refs': [self._resolve(self._ref_ids, ref) for ref in row.required_refs],
            }))
        return SupportAssessmentResponse(results=rows)
