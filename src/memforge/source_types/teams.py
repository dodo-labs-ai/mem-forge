"""Microsoft Teams: selected conversations, collected by the local agent through the conversation layer."""

from __future__ import annotations

from typing import Any

from memforge.source_adapters import teams as adapter
from memforge.source_adapters.contracts import canonical_record_profile
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor


def _job_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("initial_history_days", config.get("max_age_days", 14))
    config.setdefault("rolling_retention_days", None)
    return config


DESCRIPTOR = SourceTypeDescriptor(
    source_type="teams",
    gene="memforge.genes.teams_gene:TeamsGene",
    unit_noun="message",
    # Initial history changes collection breadth but never removes established
    # lineage. Only an explicit rolling-retention policy changes membership.
    scope_fields=("conversation_ids", "channels", "group_chats", "individual_chats", "rolling_retention_days"),
    evidence_profiles={"message": canonical_record_profile("teams-message", 2)},
    legacy_evidence_profiles={"message": canonical_record_profile("teams-message", 1)},
    record_schemas=adapter.CANONICAL_RECORD_SCHEMAS,
    local_agent=LocalAgentCollection(
        operation="teams_sync",
        job_config_fields=frozenset(
            {
                "region",
                "conversation_ids",
                "channels",
                "group_chats",
                "individual_chats",
                "initial_history_days",
                "rolling_retention_days",
            }
        ),
        job_config_defaults=_job_config_defaults,
    ),
)
