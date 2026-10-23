"""A base for test Genes that implement only the steps a test exercises."""

from __future__ import annotations

from memforge.genes.document_gene import DocumentGene
from memforge.models import GeneConfigSchema, GeneMetadata


class FakeGene(DocumentGene):
    """A document Gene whose unexercised steps raise; it inherits every optional default.

    It cannot list its scope. A fake that lists answers ``lists_scope`` with
    ``True`` and implements ``list_scope``.
    """

    listing_kinds = frozenset()

    def __init__(self, config: dict | None = None, source_id: str = "src-fake") -> None:
        super().__init__(dict(config or {}), source_id)

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

    @classmethod
    def lists_scope(cls, config) -> bool:
        return False

    async def list_scope(self):
        raise NotImplementedError

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        raise NotImplementedError
        yield

    async def fetch(self, item):
        raise NotImplementedError

    async def normalize(self, raw):
        raise NotImplementedError
