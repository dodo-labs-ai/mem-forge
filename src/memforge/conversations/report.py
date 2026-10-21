"""What a collection attempt would change, reported without changing anything.

A dry run plans one attempt as the sync run would and reports each Unit it
would project or remove, so the outcome can be checked before it is applied:
which Units are new, which change and how, and which go.
"""

from __future__ import annotations

from typing import Any

from memforge.conversations.layer import SourcePlan
from memforge.conversations.projection import conversation_document_id


def plan_report(plan: SourcePlan, *, source_id: str) -> dict[str, Any]:
    """The plan of one attempt as JSON-compatible data."""

    units = []
    for unit in plan.units:
        stored = {member.identity: member for member in (unit.stored.members if unit.stored else ())}
        planned = {member.identity: member for member in unit.members}
        changed_records = sorted(
            identity.message_id
            for identity, member in planned.items()
            if identity in stored and stored[identity].content_identity != member.content_identity
        )
        units.append(
            {
                "provider_key": unit.provider_key,
                "kind": unit.kind.value,
                "document_id": unit.stored.document_id
                if unit.stored is not None
                else conversation_document_id(source_id, unit.provider_key),
                "change": "changed" if unit.stored is not None else "new",
                "conversation_id": unit.conversation.conversation_id,
                "members": len(planned),
                "added": sorted(identity.message_id for identity in planned.keys() - stored.keys()),
                "removed": sorted(identity.message_id for identity in stored.keys() - planned.keys()),
                "changed": changed_records,
                # New or changed member content is read by the extraction model again.
                "extracts": bool(planned.keys() - stored.keys() or changed_records),
            }
        )
    return {
        "units": units,
        "removals": [
            {
                "document_id": removal.unit.document_id,
                "provider_key": removal.unit.provider_key,
                "conversation_id": removal.unit.conversation_id,
                "reason": removal.reason.value,
            }
            for removal in plan.removals
        ],
        "unreadable": [
            {"conversation_id": evidence.conversation_id, "provider_answer": evidence.status.provider_answer}
            for evidence in plan.unreadable
        ],
        "summary": {
            "new_units": sum(1 for unit in units if unit["change"] == "new"),
            "changed_units": sum(1 for unit in units if unit["change"] == "changed"),
            "units_extracted_again": sum(1 for unit in units if unit["extracts"]),
            "removed_units": len(plan.removals),
        },
    }
