"""Gene base class -- the abstract interface every data-source plugin implements.

A gene encapsulates the complete lifecycle for syncing data from a single
external source (Confluence, Jira, Teams, Outlook, ...) into MemInception's
memory layer.  Concrete genes override the five abstract methods and register
themselves via the ``@register_gene`` decorator in ``genes/__init__.py``.

All gene-related dataclasses (GeneMetadata, GeneConfigSchema, ConfigField,
ConfigGroup, ConfigFieldType, ContentItem, RawContent, NormalizedContent) live
in ``meminception.models`` and are re-exported here for convenience.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from meminception.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)

__all__ = [
    "Gene",
    # Re-exports from models
    "GeneMetadata",
    "GeneConfigSchema",
    "ConfigField",
    "ConfigFieldType",
    "ConfigGroup",
    "ContentItem",
    "RawContent",
    "NormalizedContent",
]

logger = logging.getLogger(__name__)


class Gene(ABC):
    """Abstract base class for all MemInception data-source plugins.

    Lifecycle (called by the sync orchestrator)::

        gene = create_gene("confluence", config, source_id)
        await gene.authenticate()
        async for item in gene.discover(since=last_sync):
            raw  = await gene.fetch(item)
            norm = await gene.normalize(raw)
            # ... hand off to enricher / memory extraction pipeline

    Subclasses **must** implement:
    - ``metadata()``      -- static gene description
    - ``config_schema()`` -- dynamic UI config fields
    - ``authenticate()``  -- validate credentials / obtain tokens
    - ``discover()``      -- yield content items modified since a timestamp
    - ``fetch()``         -- retrieve raw bytes for a single content item
    - ``normalize()``     -- convert raw content to clean markdown

    Subclasses **may** override:
    - ``health_check()``  -- connectivity / credential validity probe
    """

    # ------------------------------------------------------------------
    # Class-level metadata
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def metadata(cls) -> GeneMetadata:
        """Return static metadata describing this gene type.

        The returned ``GeneMetadata`` is used by the registry, UI, and
        scheduling layer.  It must be deterministic -- no I/O allowed.
        """

    @classmethod
    @abstractmethod
    def config_schema(cls) -> GeneConfigSchema:
        """Return the configuration schema the UI renders for this gene.

        Each ``ConfigField`` describes one user-editable setting (base URL,
        API token, space keys, etc.).  Fields are grouped via ``ConfigGroup``.
        """

    # ------------------------------------------------------------------
    # Instance initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: dict, source_id: str) -> None:
        """Initialise a gene instance for a specific configured source.

        Parameters
        ----------
        config:
            User-provided configuration values matching the schema returned
            by ``config_schema()``.
        source_id:
            Unique identifier for this configured source instance (e.g.
            ``"confluence-engineering"``).  Used as a foreign key in the
            document and sync-state tables.
        """
        self.config = config
        self.source_id = source_id
        self._log = logging.getLogger(f"{__name__}.{type(self).__name__}[{source_id}]")

    # ------------------------------------------------------------------
    # Abstract instance methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def authenticate(self) -> None:
        """Validate credentials and establish an authenticated session.

        Raises
        ------
        AuthenticationError
            (or a gene-specific subclass) if credentials are invalid or
            the source is unreachable.
        """

    @abstractmethod
    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Yield content items that have been created or modified since *since*.

        If *since* is ``None`` the gene should perform a full discovery of
        all available content (initial sync).

        Parameters
        ----------
        since:
            Only yield items modified after this timestamp.  ``None`` means
            "discover everything".

        Yields
        ------
        ContentItem
            One item per discoverable content unit (page, ticket, thread, ...).
        """

    @abstractmethod
    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch the raw content for a single discovered item.

        Parameters
        ----------
        item:
            A ``ContentItem`` previously yielded by ``discover()``.

        Returns
        -------
        RawContent
            The raw bytes and content-type for downstream normalisation.
        """

    @abstractmethod
    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert raw source content into clean, comprehensive markdown.

        The normaliser is the critical quality gate.  It **must** surface
        all meaningful structured data (labels, assignee, status, dates,
        comments, ...) as readable markdown so the enricher can extract
        memories from it.

        Parameters
        ----------
        raw:
            A ``RawContent`` previously returned by ``fetch()``.

        Returns
        -------
        NormalizedContent
            Contains ``markdown_body`` (for enrichment and embedding) and
            ``source_semantics`` (structured dict for faceted search).
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Probe whether the source is reachable and credentials are valid.

        Returns a dict with at least a ``"healthy"`` boolean key.  Concrete
        genes should override this to perform an actual API ping and return
        richer diagnostics (e.g. remaining rate-limit quota, auth expiry).
        """
        return {"healthy": True}
