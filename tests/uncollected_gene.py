"""A Gene with no collection: it declares no execution kinds and belongs to neither family."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from memforge.genes.base import Gene, SourceConfigurationError
from memforge.models import ContentItem, GeneConfigSchema, GeneMetadata, NormalizedContent, RawContent


class UncollectedGene(Gene):
    """Holds content it is given; it has nothing to fetch, so it is never synced."""

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="uncollected",
            display_name="Uncollected",
            description="Holds content it is given",
            default_sync_interval_minutes=0,
            auth_method="none",
            data_shape="document",
            execution_kinds=(),
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema()

    async def authenticate(self) -> None:
        """Nothing to authenticate."""

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        return
        yield

    async def fetch(self, item: ContentItem) -> RawContent:
        raise SourceConfigurationError("nothing to fetch")

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        raise SourceConfigurationError("nothing to normalize")
