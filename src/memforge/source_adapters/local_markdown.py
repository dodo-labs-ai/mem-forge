"""local_markdown source projection ownership."""

from __future__ import annotations


def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage.

    A file is identified by its vault and relative path, and its Observation
    is the file's own text, the stored input ``native`` decodes to, read
    under the profile its extension chooses.
    """
    from memforge.source_projection import ProjectionCoverage
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import (
        _ObservationInput, _NativeProjection, _unit_title, file_evidence_profile,
    )

    if not isinstance(native, str):
        raise ValueError(f"local Markdown input {item.item_id} is not a file's text")
    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    vault = str(item.extra["vault_id"])
    path = str(item.extra["relative_path"])
    return _NativeProjection(
        unit_type="local_file",
        provider_key=f"{vault}:{path}",
        observations=(
            _ObservationInput(
                "file_content", "content", native, native, {"path": path}, body_time,
                evidence_profile=file_evidence_profile(path),
            ),
        ),
        relations=(),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={"vault_id": vault, "path": path, "url": item.source_url},
        title=_unit_title("Markdown file", ("Vault", vault), ("Path", path)),
    )
