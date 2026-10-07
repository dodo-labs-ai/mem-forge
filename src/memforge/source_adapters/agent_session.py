"""Agent-session window projection ownership."""

from __future__ import annotations

from memforge.source_adapters.contracts import _NativeProjection, _ObservationInput, _unit_title
from memforge.source_projection import ProjectionCoverage
from memforge.source_time import SOURCE_UPDATED_AT_KEY


def project_native(*, source_id, item, native, normalized):
    """Project a submitted summary window and its receipt, with partial coverage."""
    data = native if isinstance(native, dict) else {}
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    window_id = str(data.get("doc_id") or item.item_id)
    body = str(data.get("markdown") or normalized.markdown_body)
    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    return _NativeProjection(
        unit_type="agent_session_window",
        provider_key=window_id,
        observations=(_ObservationInput("session_summary", window_id, body, body, {}, body_time),),
        relations=(),
        coverage=ProjectionCoverage.PARTIAL_PROJECTION,
        locator={
            "client": receipt.get("client"),
            "session_id": receipt.get("session_id"),
            "history_window_kind": receipt.get("history_window_kind"),
            "url": item.source_url,
        },
        title=_unit_title(
            "Agent session",
            ("Client", receipt.get("client")),
            ("Window", receipt.get("history_window_kind")),
            ("Title", item.title),
        ),
    )
