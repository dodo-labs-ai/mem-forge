"""local_markdown source projection ownership."""

from __future__ import annotations

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title

    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    data = native if isinstance(native, dict) else {}
    vault = str(data.get("vault_id") or item.space_or_project or "default")
    path = str(data.get("relative_path") or item.extra.get("relative_path") or item.item_id)
    lineage = str(data.get("file_lineage_id") or item.extra.get("file_lineage_id") or path)
    body = str(data.get("markdown") or normalized.markdown_body)
    return _NativeProjection(
        unit_type="local_file",
        provider_key=f"{vault}:{lineage}",
        observations=(_ObservationInput("file_content", "content", body, body, {"path": path}, body_time),),
        relations=(),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={"vault_id": vault, "path": path, "url": item.source_url},
        title=_unit_title(
            "Markdown file",
            ("Vault", data.get("vault_id") or item.space_or_project),
            ("Path", path),
        ),
    )
