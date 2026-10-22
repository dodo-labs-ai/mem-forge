"""A base for test Genes that implement only the steps a test exercises."""

from __future__ import annotations

from memforge.genes.base import Gene
from memforge.models import GeneConfigSchema, GeneMetadata


class FakeGene(Gene):
    """A Gene whose unexercised steps raise; it inherits every optional default."""

    config: dict = {}
    source_id = "src-fake"

    def __init__(self, config: dict | None = None, source_id: str = "src-fake") -> None:
        self.config = dict(config or {})
        self.source_id = source_id

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="fake",
            display_name="Fake",
            description="A test Gene",
            default_sync_interval_minutes=60,
            auth_method="none",
            data_shape="document",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(fields=[], groups=[])

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        raise NotImplementedError
        yield

    async def fetch(self, item):
        raise NotImplementedError

    async def normalize(self, raw):
        raise NotImplementedError
