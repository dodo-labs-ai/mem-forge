"""github_repo source projection ownership."""

from __future__ import annotations

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _UnitEndpoint, _unit_title
    from memforge.github_repo_utils import build_github_repo_doc_id

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
    rename_attested = (
        item.extra.get("rename_evidence_authoritative") is True
        or semantics.get("rename_evidence_authoritative") is True
    )
    previous = (
        item.extra.get("previous_filename") or semantics.get("previous_filename")
        if rename_attested
        else None
    )
    explicit_lineage = item.extra.get("file_lineage_id") or semantics.get("file_lineage_id")
    # Git/GitHub does not expose an immutable file id. Built-in cloud-pull
    # and local-push connectors therefore define path as file identity.
    # Rename continuity is optional and accepted only when a provider
    # adapter explicitly attests authoritative rename evidence (for
    # example, a validated Compare API `renamed` record). Never infer a
    # move from a matching blob SHA because copy+delete is ambiguous.
    lineage = str(explicit_lineage or previous or path)
    relations = ()
    if previous:
        predecessor_document_id = item.extra.get("previous_document_id") or semantics.get("previous_document_id")
        repo_url = str(item.extra.get("repo_url") or semantics.get("repo_url") or "")
        repo_ref = str(item.extra.get("repo_ref") or semantics.get("repo_ref") or "")
        if not predecessor_document_id and repo_url and repo_ref:
            predecessor_document_id = build_github_repo_doc_id(
                source_id=source_id,
                repo_url=repo_url,
                repo_ref=repo_ref,
                relative_path=str(previous),
            )
        relations = (
            (
                SourceRelationType.RENAMED_FROM,
                "$unit",
                _UnitEndpoint("github_file", f"{repo}:{previous}"),
                None,
                ({"predecessor_document_id": str(predecessor_document_id)} if predecessor_document_id else {}),
            ),
        )
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
        relations=relations,
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={"repository": repo, "path": path, "ref": item.extra.get("repo_ref"), "url": item.source_url},
        title=_unit_title(
            "GitHub file",
            ("Repository", repo),
            ("Path", path),
            ("Ref", item.extra.get("repo_ref") or semantics.get("repo_ref")),
        ),
    )
