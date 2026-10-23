"""Microsoft Teams: a conversation Source collected on the user's device.

Teams is read through the user's browser session (ADR 0002), so a local
agent collects it: the device executor runs the Teams provider
(:mod:`memforge.genes.teams_provider`) for the server's collection request
and uploads the Collection Manifest. On the server, this Gene only describes
the Source and plans from that manifest through the conversation layer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from memforge.conversations.layer import ConversationSelection
from memforge.conversations.membership import PlannedUnit
from memforge.conversations.projection import unit_payload
from memforge.conversations.records import MessageIdentity, record_created_at, record_revision
from memforge.conversations.units import Member
from memforge.genes.conversation_gene import ConversationGene
from memforge.genes.teams_provider import teams_conversation_label
from memforge.genes.toolkit.config import FieldIssue, SourceConfigInvalid
from memforge.local_agent.source_contract import TEAMS_MAX_INITIAL_HISTORY_DAYS, TEAMS_ROLLING_RETENTION_PRESETS
from memforge.local_agent.teams_contract import validate_teams_canonical_messages
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ConfigOption,
    ConfigPanel,
    GeneConfigSchema,
    GeneMetadata,
)
from memforge.source_adapters.teams import MESSAGE_SCHEMA, message_record, teams_unit_shape

__all__ = ["TeamsGene"]


def _refused(key: str, message: str) -> SourceConfigInvalid:
    return SourceConfigInvalid([FieldIssue(key, message)])

# A Teams Unit's stored input in window-package form: one window's messages
# with its conversation's framing, as a local agent uploaded it.
TEAMS_WINDOW_PACKAGE_KIND = "teams_window_document"


class TeamsGene(ConversationGene):
    """Microsoft Teams channels, group chats and direct messages."""

    record_schema = MESSAGE_SCHEMA.qualified_name

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="teams",
            display_name="Microsoft Teams",
            description="Channel messages, group chats, and direct messages",
            default_sync_interval_minutes=60,
            auth_method="browser_cookie",
            data_shape="message",
            execution_kinds=("local_agent",),
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(
                    key="connection",
                    label="Teams session",
                    order=0,
                    purpose="connection",
                    panels=[ConfigPanel("teams_session")],
                ),
                ConfigGroup(key="scope", label="Conversations to sync", order=1, purpose="content"),
                ConfigGroup(key="sync", label="Sync settings", order=2, purpose="options"),
            ],
            fields=[
                ConfigField(
                    key="region",
                    label="Teams Region",
                    field_type=ConfigFieldType.SELECT,
                    required=False,
                    default="emea",
                    choices=[
                        ConfigOption("emea", "EMEA"),
                        ConfigOption("amer", "Americas"),
                        ConfigOption("apac", "Asia Pacific"),
                    ],
                    help_text="Teams API regional endpoint",
                    group="connection",
                    order=0,
                ),
                ConfigField(
                    key="conversation_ids",
                    label="Teams Conversations",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=True,
                    placeholder="Conversation IDs selected through Browse Teams",
                    help_text="Direct Teams conversation IDs selected through Browse Teams",
                    group="scope",
                    order=0,
                    widget="teams_conversations",
                ),
                ConfigField(
                    key="initial_history_days",
                    label="Initial History (days)",
                    field_type=ConfigFieldType.INTEGER,
                    required=False,
                    default="14",
                    help_text="How far back to collect, and to follow edits, deletions and replies",
                    group="sync",
                    order=0,
                ),
                ConfigField(
                    key="rolling_retention_days",
                    label="Rolling Retention",
                    field_type=ConfigFieldType.SELECT,
                    required=False,
                    default="forever",
                    choices=[
                        ConfigOption("forever", "Forever"),
                        *(
                            ConfigOption(str(days), f"{days // 365} year{'s' if days > 365 else ''}")
                            for days in TEAMS_ROLLING_RETENTION_PRESETS
                        ),
                    ],
                    help_text="Explicitly remove MemForge support older than the selected period",
                    group="sync",
                    order=1,
                ),
            ],
        )

    @classmethod
    def normalize_config(cls, config: dict) -> None:
        """Persist only canonical provider ids for the Teams projection scope."""

        from memforge.local_agent.source_contract import (
            TEAMS_CONVERSATION_SELECTOR_FIELDS,
            canonical_teams_conversation_ids,
        )

        try:
            conversation_ids = canonical_teams_conversation_ids(config, require_nonempty=True)
        except ValueError as exc:
            raise _refused("conversation_ids", str(exc)) from None
        for field in TEAMS_CONVERSATION_SELECTOR_FIELDS:
            config.pop(field, None)
        config["conversation_ids"] = list(conversation_ids)
        if "initial_history_days" not in config:
            config["initial_history_days"] = config.get("max_age_days", 14)
        history_rule = f"Teams initial history must be between 1 and {TEAMS_MAX_INITIAL_HISTORY_DAYS} days"
        try:
            initial_history_days = int(config["initial_history_days"])
        except (TypeError, ValueError):
            raise _refused("initial_history_days", history_rule) from None
        if not 1 <= initial_history_days <= TEAMS_MAX_INITIAL_HISTORY_DAYS:
            raise _refused("initial_history_days", history_rule)
        config["initial_history_days"] = initial_history_days
        retention = config.get("rolling_retention_days")
        if retention in (None, "", "forever", "none"):
            config["rolling_retention_days"] = None
        else:
            retention_rule = "Teams rolling retention must be Forever, 1 year, 2 years, or 3 years"
            try:
                retention_days = int(retention)
            except (TypeError, ValueError):
                raise _refused("rolling_retention_days", retention_rule) from None
            if retention_days not in TEAMS_ROLLING_RETENTION_PRESETS:
                raise _refused("rolling_retention_days", retention_rule)
            config["rolling_retention_days"] = retention_days
        config.pop("max_age_days", None)
        config.pop("conversation_gap_minutes", None)
        config.pop("max_block_messages", None)

    def __init__(self, config: dict, source_id: str) -> None:
        config = dict(config)
        self.normalize_config(config)
        super().__init__(config, source_id)

    async def authenticate(self) -> None:
        """Nothing to authenticate on the server: the device holds the Teams session."""

    def conversation_selection(self) -> ConversationSelection:
        return ConversationSelection(
            conversation_ids=self.selected_conversation_ids(self.config),
            initial_history_days=int(self.config["initial_history_days"]),
            rolling_retention_days=self.config.get("rolling_retention_days"),
        )

    def selected_conversation_ids(self, projection_scope: Mapping[str, Any]) -> tuple[str, ...]:
        from memforge.local_agent.source_contract import canonical_teams_conversation_ids

        return canonical_teams_conversation_ids(projection_scope)

    def stored_unit_payload(self, body: bytes) -> Mapping[str, Any] | None:
        """The conversation Unit payload of a stored input in window-package form."""

        package = json.loads(body.decode("utf-8"))
        if not isinstance(package, dict) or package.get("package_kind") != TEAMS_WINDOW_PACKAGE_KIND:
            return None
        window = package["raw_payload"]
        conversation_id = str(package["conversation_id"])
        kind, thread_root_id = teams_unit_shape(str(package["window_id"]))
        conversation = teams_conversation_label(
            {
                "id": conversation_id,
                "topic": window.get("conversation_name") or window.get("channel_name"),
                "type": window.get("conversation_type"),
            }
        )
        members = []
        for message in validate_teams_canonical_messages(window.get("messages")):
            record = message_record(
                message,
                conversation_id=conversation_id,
                conversation_name=window.get("conversation_name") or window.get("channel_name"),
            )
            members.append(
                Member(
                    identity=MessageIdentity(conversation_id, str(message["id"])),
                    created_at=record_created_at(record),
                    record=record,
                    revision=record_revision(record),
                )
            )
        unit = PlannedUnit(
            kind=kind,
            provider_key=str(package["window_id"]),
            conversation=conversation,
            members=tuple(sorted(members, key=lambda member: (member.created_at, member.identity.message_id))),
            thread_root_id=thread_root_id,
        )
        return unit_payload(unit, record_schema=self.record_schema)
