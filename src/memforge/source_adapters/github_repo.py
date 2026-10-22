"""github_repo source projection ownership."""

from __future__ import annotations

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title

    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    semantics = normalized.source_semantics
    repo = "/".join(
        value
        for value in (
            str(item.extra.get("repo_owner") or semantics.get("repo_owner") or ""),
            str(item.extra.get("repo_name") or semantics.get("repo_name") or ""),
        )
        if value
    ) or str(item.extra.get("repo_url") or semantics.get("repo_url") or item.space_or_project)
    path = str(item.extra.get("relative_path") or semantics.get("relative_path") or item.item_id)
    explicit_lineage = item.extra.get("file_lineage_id") or semantics.get("file_lineage_id")
    # Git/GitHub does not expose an immutable file id. Built-in cloud-pull
    # and local-push connectors therefore define path as file identity. A
    # renamed file keeps its Unit only when discovery attests the rename
    # (``item.renamed_from``): the sync run then binds it to its
    # predecessor's Unit. A matching blob SHA is never read as a move,
    # because copy+delete is ambiguous.
    lineage = str(explicit_lineage or path)
    return _NativeProjection(
        unit_type="github_file",
        provider_key=f"{repo}:{lineage}",
        observations=(
            _ObservationInput(
                "file_content",
                "content",
                normalized.markdown_body,
                normalized.markdown_body,
                {"path": path},
                body_time,
            ),
        ),
        relations=(),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={"repository": repo, "path": path, "ref": item.extra.get("repo_ref"), "url": item.source_url},
        title=_unit_title(
            "GitHub file",
            ("Repository", repo),
            ("Path", path),
            ("Ref", item.extra.get("repo_ref") or semantics.get("repo_ref")),
        ),
    )
