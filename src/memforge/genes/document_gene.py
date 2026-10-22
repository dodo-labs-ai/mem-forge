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
Document id of every item in that scope right now, listed completely and
without fetching any content.

The listing is how a run shows that something is gone. A Document
the Source holds but the listing does not name may have been deleted at the
provider, and the run decides what to do with it from the listing's kind.
Discovery still decides what to fetch: ``discover(since)`` yields the items
that changed, and the listing only names identifiers.

Choosing the listing kind
-------------------------

Ask what defines the scope:

- **The listing itself defines it: existence.** An item the listing does not
  name is outside the scope by definition, so it is gone from this Source.
  Examples: a GitHub repository tree, where a file's path is part of its
  Document id and a moved file is a new Document; a sitemap the site declares
  complete; a list of pages the user configured.
- **A condition over items that exist on their own defines it: query.** An
  item can stop matching and keep its identifier, so an unlisted item is gone
  only if the provider says so when asked by identifier
  (:meth:`DocumentGene.confirm_absent`). Leaving the query is not deletion.
  Examples: a Jira issue that no longer matches the JQL; a Confluence page
  moved to another page tree.

A Gene declares the kinds it can produce in :attr:`DocumentGene.listing_kinds`.
When in doubt, choose query: it never removes an item the provider still has.

What the listing decides
------------------------

A run lists its scope once, after discovery and before it retries failed
Documents, and keeps that listing for the rest of the run.

Absence. A held Document an existence listing leaves out is removed. One a
query listing leaves out is removed only when ``confirm_absent`` answers
``absent``; one that still exists keeps its Document and is not refreshed
while it stays outside the query. A configuration that cannot list, a listing
that fails and a confirmation that fails each remove nothing. A Document that
failed in this run does not hold back the removal of another one.

Failed Documents. A failed Document the listing names is retried. One an
existence listing leaves out is gone, so it leaves the failure ledger. One a
query listing leaves out is looked up through :meth:`DocumentGene.rediscover`,
which settles it when the provider no longer yields it.

Renames. A connector whose provider attests a rename sets
:attr:`~memforge.models.ContentItem.renamed_from` on the discovered item to
the Document id it was renamed from. The item commits to that Document's
Unit, which is kept, even unlisted, until the renamed item commits.

Implementing a document connector
---------------------------------

Besides ``authenticate``, ``discover``, ``fetch`` and ``normalize``:

1. Declare :attr:`~DocumentGene.listing_kinds`.
2. Answer :meth:`~DocumentGene.lists_scope` for each configuration. Answer
   ``False`` for a configuration whose scope cannot be listed completely, such
   as a bounded link crawl: its Sources then never remove a Document through a
   listing, which is safe but lets deleted items linger.
3. Implement :meth:`~DocumentGene.list_scope`. When ``discover`` already
   visits the whole scope whatever ``since`` is, call
   :meth:`~DocumentGene.record_scope_listing` at the end of that walk and
   return the recorded listing instead of walking again.
4. With a query listing kind, implement :meth:`~DocumentGene.confirm_absent`.
5. Optionally, implement :meth:`~DocumentGene.rediscover` together with
   :meth:`~DocumentGene.rediscovers_documents`.

Startup refuses a Gene that leaves out a required member, and the contract
suite in ``tests/contracts/test_document_connector_contract.py`` checks the
behaviour against fakes of the provider; its ``document_fakes`` package says
how to add a connector's fakes.

Collected on a device
---------------------

A Source collected on a user's device, such as local markdown, a GitHub
repository pushed from the internal network, or Jira read through the local
agent, submits a complete snapshot manifest of its scope. The run reads that
manifest as the listing, with the Gene's single listing kind, and never calls
:meth:`DocumentGene.list_scope`. Such a configuration answers ``True`` from
:meth:`DocumentGene.lists_scope`, and its ``list_scope`` raises.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from memforge.genes.base import Gene
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, ScopeListingKind

__all__ = ["AbsenceAnswer", "DocumentGene"]


class DocumentGene(Gene):
    """Base class of every document connector.

    Besides the steps every Gene implements (``authenticate``, ``discover``,
    ``fetch`` and ``normalize``), a document connector says whether and how
    its scope is listed and, for a query listing, how one item is confirmed
    gone. See the module docstring for the concepts.
    """

    listing_kinds: ClassVar[frozenset[ScopeListingKind]]
    """The kinds of :class:`~memforge.models.ScopeListing` this connector produces.

    For example ``frozenset({ScopeListingKind.QUERY})`` for Jira, whose JQL is
    a query, and ``frozenset({ScopeListingKind.EXISTENCE})`` for a repository
    tree. An empty set means no configuration of this connector can list its
    scope, so :meth:`lists_scope` answers ``False`` for every configuration.

    Checked by ``test_a_listing_is_one_of_the_declared_kinds`` and
    ``test_lists_scope_matches_each_fake_configuration``.
    """

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._discovered_scope: ScopeListing | None = None

    def reset_scope_listing(self) -> None:
        """Forget the listing an earlier discovery walk recorded, before a run discovers again."""

        self._discovered_scope = None

    @classmethod
    @abstractmethod
    def lists_scope(cls, config: Mapping[str, Any]) -> bool:
        """Whether a Source with this configuration can list its whole scope.

        A class method, because a scope change is checked before any Gene
        exists. For example, GitHub Pages answers ``True`` for a subtree read
        from the repository's branch tree, and ``False`` for a link crawl,
        which only reaches the pages currently linked.

        Answer ``True`` only when :meth:`list_scope` either returns a complete
        listing for this configuration or raises; it must never quietly list
        less. A configuration collected on a device answers ``True``, because
        its snapshot manifest is the listing. When unsure, answer ``False``:
        the Source then removes nothing through a listing.

        Checked by ``test_lists_scope_matches_each_fake_configuration``.
        """

    @abstractmethod
    async def list_scope(self) -> ScopeListing:
        """List every item in the configured scope by Document id, whatever ``since`` discovery used.

        Called only for a configuration :meth:`lists_scope` answers ``True``
        for, after or without this run's :meth:`discover`, so it must not
        depend on one. Return identifiers only: fetch no content and call no
        model. For example, Jira runs one identifier-only JQL search without
        the ``since`` filter, and Confluence returns the page walk its
        discovery recorded with :meth:`record_scope_listing`, or walks the
        tree itself when discovery did not.

        The listing is complete or it is not returned. Raise when paging is
        incomplete, a result is truncated or the provider fails, and the run
        then removes nothing. A configuration collected on a device raises,
        saying that its listing is the snapshot manifest.

        Checked by ``test_a_truncated_listing_raises``,
        ``test_a_failed_listing_raises``,
        ``test_a_standalone_listing_equals_the_listing_after_discovery``,
        ``test_a_listing_reuses_the_discovery_walk`` and
        ``test_a_device_collected_configuration_lists_through_its_manifest``.
        """

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

    async def confirm_absent(self, items: Sequence[ContentItem]) -> Mapping[str, AbsenceAnswer]:
        """Ask the provider, by identifier, whether each item a query listing left out still exists.

        Required when :attr:`listing_kinds` include
        :attr:`~memforge.models.ScopeListingKind.QUERY`. Each item is rebuilt
        from its stored Document, as for :meth:`rediscover`. Return one
        :class:`~memforge.models.AbsenceAnswer` per item, keyed by Document id:

        - ``absent`` when the provider answers not found or deleted, such as
          HTTP 404 or 410, or a Confluence page in the trash;
        - ``exists`` when the provider returns the item, whether or not it
          still matches the query: a Jira issue moved to another project;
        - ``hidden`` when the provider refuses to show this one item, such as
          HTTP 403. It may still exist, so it is kept.

        Raise on authentication, rate-limit (after retries), transport and
        server errors, and whenever the answer is unclear: the run then
        removes nothing.

        Checked by ``test_confirm_absent_maps_each_provider_status``.
        """

        raise NotImplementedError(f"{type(self).__name__} cannot confirm absence by identifier")

    @classmethod
    def rediscovers_documents(cls, config: Mapping[str, Any]) -> bool:
        """Whether a Source with this configuration can ask its provider for one Document by id.

        Answer ``True`` only when :meth:`rediscover` is implemented for this
        configuration. Content a local agent collected or a user uploaded has
        no provider to ask, because its only copy is the stored input, so the
        answer is ``False`` for it.
        """
        return False

    async def rediscover(self, item: ContentItem) -> ContentItem | None:
        """Return the provider's current item for one stored Document, as discovery would list it.

        ``item`` is rebuilt from the stored Document. Apply the same scope
        predicates as :meth:`discover`: return the item a full discovery would
        yield for that Document now, so :meth:`fetch` reads its current state,
        or ``None`` when a full discovery would not yield it, because the
        provider no longer has it or it has left the scope. For example,
        Confluence returns ``None`` for a page moved out of the configured page
        tree. Implemented exactly when :meth:`rediscovers_documents` can answer
        ``True``.

        Answer ``None`` only when the provider's answer shows it. Raise on
        authentication, rate-limit, transport and server errors, and on an
        answer that is incomplete or names another item: the Document then
        keeps its state and is tried again on a later run.

        Checked by ``test_rediscover_keeps_the_predicates_of_discover``.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot rediscover one Document")
