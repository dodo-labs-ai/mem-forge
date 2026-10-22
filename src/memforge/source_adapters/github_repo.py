"""github_repo source projection ownership."""

from __future__ import annotations

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _UnitEndpoint, _unit_title
    from memforge.github_repo_utils import build_github_repo_doc_id, github_file_lineage, github_file_provider_key

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
    # and local-push connectors therefore define path as file identity.
    # A file keeps its Unit across a rename only when discovery attests the
    # rename through ``item.renamed_from``; a matching blob SHA is never read
    # as a move, because copy+delete is ambiguous.
    provider_key = (
        github_file_provider_key(repository=repo, lineage=str(explicit_lineage))
        if explicit_lineage
        else item.renamed_from or github_file_provider_key(repository=repo, lineage=path)
    )
    relations = ()
    if item.renamed_from:
        previous_path = github_file_lineage(item.renamed_from, repository=repo)
        repo_url = str(item.extra.get("repo_url") or semantics.get("repo_url") or "")
        repo_ref = str(item.extra.get("repo_ref") or semantics.get("repo_ref") or "")
        predecessor_document_id = (
            build_github_repo_doc_id(
                source_id=source_id,
                repo_url=repo_url,
                repo_ref=repo_ref,
                relative_path=previous_path,
            )
            if previous_path and repo_url and repo_ref
            else None
        )
        relations = (
            (
                SourceRelationType.RENAMED_FROM,
                "$unit",
                _UnitEndpoint("github_file", item.renamed_from),
                None,
                ({"predecessor_document_id": predecessor_document_id} if predecessor_document_id else {}),
            ),
        )
    return _NativeProjection(
        unit_type="github_file",
        provider_key=provider_key,
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
