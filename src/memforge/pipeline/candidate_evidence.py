"""Present a Candidate's selected Evidence the same way to every judgment that reads it.

Candidate admission and Sparse Relation both read a Candidate with the exact
Primary and Required parts its extraction selected. The Evidence catalog interns
each part once per request under ``PRM``/``REQ`` references, and Artifact parts
carry their image bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from memforge.derivation_work import payload_hash
from memforge.llm.batch_runner import RequestTooLarge
from memforge.llm.relation_catalog import RequestCatalog
from memforge.llm.structured_images import StructuredLlmImage
from memforge.memory.evidence import EvidencePartKind, EvidenceRole
from memforge.models import RawMemory

_REF_PREFIX_BY_ROLE = {EvidenceRole.PRIMARY: "PRM", EvidenceRole.REQUIRED: "REQ"}


class EvidenceArtifactUnavailable(ValueError):
    """The bytes of a selected Artifact part cannot be supplied to the model."""


@dataclass(frozen=True)
class CandidateEvidenceCatalog:
    """Request-local Evidence references for a set of Candidates."""

    refs: Mapping[str, tuple[str, ...]]
    entries: Mapping[str, Mapping[str, object]]

    @property
    def artifact_observation_ids(self) -> frozenset[str]:
        return frozenset(
            str(entry["observation_id"]) for entry in self.entries.values()
            if entry["kind"] == EvidencePartKind.ARTIFACT.value
        )


def candidate_evidence_catalog(candidates: Mapping[str, RawMemory]) -> CandidateEvidenceCatalog:
    """Intern every selected part of the named Candidates; identical parts share one reference."""

    catalogs = {role: RequestCatalog(prefix) for role, prefix in _REF_PREFIX_BY_ROLE.items()}
    entries: dict[str, Mapping[str, object]] = {}
    refs: dict[str, tuple[str, ...]] = {}
    for candidate_id, raw in candidates.items():
        selection = raw.resolved_evidence_selection
        candidate_refs = []
        for part in selection.parts if selection is not None else ():
            entry = {
                "role": part.role.value,
                "excerpt": part.excerpt,
                "observation_id": part.anchor.observation_id,
                "revision_id": part.anchor.observation_revision_id,
                "kind": part.kind.value,
            }
            ref = catalogs[part.role].add(payload_hash(entry), entry)
            entries[ref] = entry
            candidate_refs.append(ref)
        refs[candidate_id] = tuple(candidate_refs)
    return CandidateEvidenceCatalog(refs, entries)


def load_evidence_images(
    observation_ids: Collection[str], *, images: tuple[StructuredLlmImage, ...] = (),
    image_loader: Callable[[Collection[str]], tuple[StructuredLlmImage, ...]] | None = None,
) -> tuple[StructuredLlmImage, ...]:
    """Return the image of every selected Artifact part, or fail when any is missing."""

    from memforge.pipeline.projection_images import ProjectionImageLoadError

    wanted = set(observation_ids)
    if not wanted:
        return ()
    try:
        loaded = image_loader(wanted) if image_loader is not None else tuple(
            image for image in images if image.source_observation_id in wanted
        )
    except ProjectionImageLoadError as error:
        if error.error_code != "image_batch_too_large":
            raise
        raise RequestTooLarge(error.error_code) from error
    if wanted != {image.source_observation_id for image in loaded}:
        raise EvidenceArtifactUnavailable("current Artifact Evidence bytes are required")
    return tuple(loaded)
