"""Agent Session Gene: the registered type of per-client agent-session Sources."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from memforge.genes.base import Gene, SourceConfigurationError
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)

__all__ = ["AgentSessionGene"]


_NO_PROVIDER_CONTENT = (
    "Agent Session Sources receive knowledge through agent-session windows; "
    "they have no provider content to fetch"
)


class AgentSessionGene(Gene):
    """Private per-client, per-user Sources written by agent-session windows.

    Codex and Claude Code upload transcript windows to the agent-session API,
    which patches the owner's Agent Knowledge directly under the Source activity
    lease. The Source declares no execution kinds, so it is never synced and
    discovery finds nothing.
    """

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="agent_session",
            display_name="Agent Session",
            description="Private coding-agent knowledge from Codex and Claude Code session windows",
            default_sync_interval_minutes=0,
            auth_method="local_file",
            data_shape="message",
            execution_kinds=(),
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[ConfigGroup(key="client", label="Client", order=0)],
            fields=[
                ConfigField(
                    key="client",
                    label="Client",
                    field_type=ConfigFieldType.STRING,
                    required=False,
                    placeholder="codex",
                    help_text="Coding client whose session windows this Source receives (e.g. 'codex', 'claude-code').",
                    group="client",
                    order=0,
                ),
            ],
            project_field="repo",
        )

    async def authenticate(self) -> None:
        """Agent Session Sources hold no provider credentials."""

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Agent Session Sources have no provider to enumerate."""
        return
        yield

    async def fetch(self, item: ContentItem) -> RawContent:
        raise SourceConfigurationError(_NO_PROVIDER_CONTENT)

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        raise SourceConfigurationError(_NO_PROVIDER_CONTENT)
