"""The Gene of a document connector: pages, issues, files and other items a provider lists.

A connector whose Gene is a :class:`DocumentGene` belongs to the document
family. Its provider already defines the items: a Confluence page, a Jira
issue or a file in a repository has an identifier and can be listed. Each item
becomes one Document, and the sync run keeps the Documents of a Source in step
with what the provider holds.

Scope listing
-------------

The *scope* is what the user configured the Source to collect: a Confluence
space or page tree, a JQL query, a repository branch and path filter, a site's
sitemap. A *scope listing* (:class:`~memforge.models.ScopeListing`) is the
identifier of every item in that scope right now, listed completely and
without fetching any content.

The listing is how a run shows that something is gone. A Document the Source
holds but the listing does not name may have been deleted at the provider,
and the run decides what to do with it from the listing's kind. Discovery
still decides what to fetch: ``discover(since)`` yields the items that changed,
and the listing only names identifiers.

Choosing the listing kind
-------------------------

Ask what defines the scope:

- **The listing itself defines it: existence.** An item the listing does not
  name is outside the scope by definition, so it is gone from this Source.
  Examples: a repository tree, where a file's path is part of its Document id
  and a moved file is a new Document; an authoritative sitemap; a list of
  pages the user configured.
- **A condition over items that exist on their own defines it: query.** An
  item can stop matching and keep its identifier, so an unlisted item is gone
  only if the provider says so when asked by identifier
  (:meth:`DocumentGene.confirm_absent`). Leaving the query is not deletion.
  Examples: a Jira issue that no longer matches the JQL; a Confluence page
  moved to another page tree.

A Gene declares the kinds it can produce in :attr:`DocumentGene.listing_kinds`.
When in doubt, choose query: it never removes an item the provider still has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from memforge.genes.base import Gene
from memforge.models import ContentItem, ScopeListing, ScopeListingKind

__all__ = ["DocumentGene"]


class DocumentGene(Gene):
    """Base class of every document connector.

    Besides the steps every Gene implements (``authenticate``, ``discover``,
    ``fetch`` and ``normalize``), a document connector says how its scope is
    listed and, for a query listing, how one item is confirmed gone. See the
    module docstring for the concepts.
    """

    listing_kinds: ClassVar[frozenset[ScopeListingKind]]
    """The kinds of :class:`~memforge.models.ScopeListing` this connector produces.

    For example ``frozenset({ScopeListingKind.QUERY})`` for Jira, whose JQL is
    a query, and ``frozenset({ScopeListingKind.EXISTENCE})`` for a repository
    tree. An empty set means no configuration of this connector can list its
    scope, so its Sources never detect deletions through a listing.
    """

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._discovered_scope: ScopeListing | None = None

    def begin_discovery(self) -> None:
        super().begin_discovery()
        self._discovered_scope = None

    @classmethod
    def rediscovers_documents(cls, config: Mapping[str, Any]) -> bool:
        """Whether a Source with this configuration can ask its provider for one Document by id.

        Answer ``True`` only when :meth:`rediscover` is implemented for this
        configuration. Content a local agent collected or a user uploaded has
        no provider to ask, because its only copy is the stored input, so the
        answer is ``False`` for it.
        """
        return False

    def record_scope_listing(self, kind: ScopeListingKind, doc_ids: set[str] | frozenset[str]) -> None:
        """Record the listing a discovery walk produced, for :meth:`list_scope` to return.

        A connector whose discovery visits every item in scope on every run,
        and uses ``since`` only to choose which items to yield, calls this once
        the walk has completed, with the Document id of every item it visited.
        Confluence walks its page tree this way. Never call it after a walk
        that stopped early: an incomplete listing would make the items it
        missed look deleted.
        """

        self._discovered_scope = ScopeListing(kind=kind, doc_ids=frozenset(doc_ids))

    async def list_scope(self) -> ScopeListing | None:
        """List every item in the configured scope by Document id, whatever ``since`` discovery used.

        Called after this run's :meth:`discover`. Return identifiers only:
        fetch no content and call no model. The listing is complete or it is
        not returned. Raise when paging is incomplete, a result is truncated
        or the provider fails, and the run then removes nothing.

        By default this returns what discovery recorded with
        :meth:`record_scope_listing`. A connector whose discovery asks the
        provider only for items changed since ``since``, such as Jira's JQL
        search, lists its scope with a separate identifier-only request.

        ``None`` means this configuration cannot list its scope, so no
        Document becomes absent through a listing.
        """

        return self._discovered_scope

    async def confirm_absent(self, items: Sequence[ContentItem]) -> frozenset[str]:
        """Return the Document ids of the unlisted ``items`` that the provider no longer has.

        Only a connector whose :attr:`listing_kinds` include
        :attr:`~memforge.models.ScopeListingKind.QUERY` implements it. Each item
        is rebuilt from its stored Document, as for :meth:`rediscover`.

        Read each item by its identifier. An item is absent only when the
        provider answers that it is not found or deleted (for example HTTP 404
        or 410, or a page in the trash). Keep an item the provider still
        returns, whether or not it still matches the query, and keep one the
        provider refuses to show for permission reasons (HTTP 403). Raise on
        authentication, rate-limit, transport and server errors: the run then
        removes nothing.
        """

        raise NotImplementedError(f"{type(self).__name__} cannot confirm absence by identifier")

    async def rediscover(self, item: ContentItem) -> ContentItem | None:
        """Return the provider's current item for one stored Document.

        ``item`` is rebuilt from the stored Document. Return the item
        discovery would yield for that Document now, so :meth:`fetch` reads its
        current state, or ``None`` when the provider no longer returns it.
        Implemented exactly when :meth:`rediscovers_documents` can answer
        ``True``.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot rediscover one Document")
