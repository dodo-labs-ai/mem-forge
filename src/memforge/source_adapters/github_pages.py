"""github_pages source projection ownership."""

from __future__ import annotations

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title

    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    canonical_url = str(
        item.extra.get("canonical_url") or normalized.source_semantics.get("canonical_url") or item.source_url
    )
    semantic_value = native if isinstance(native, str) else normalized.markdown_body
    semantic_content = normalized.markdown_body
    return _NativeProjection(
        unit_type="rendered_page",
        provider_key=canonical_url,
        observations=(
            _ObservationInput("page_content", "content", semantic_content, semantic_value, {}, body_time),
        ),
        relations=(),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={"canonical_url": canonical_url, "title": item.title},
        title=_unit_title("GitHub Pages page", ("Title", item.title), ("URL", canonical_url)),
    )
