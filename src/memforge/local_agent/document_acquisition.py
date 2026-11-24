"""What a device run of a document Source hands the server, and what the server asks of it.

A document Source collected on its owner's device runs through a
``document_acquisition`` local-agent job. The daemon creates the type's own
Gene with the job's configuration and its own credentials and makes the
provider calls a server run makes; the server then runs the sync with the
type's recorded Gene (:mod:`memforge.genes.recorded`), which answers those
calls from what the device handed over:

1. The **job** carries the configuration without secrets
   (:func:`acquisition_job_config`), the ``since`` discovery reads from, the
   excluded scopes of an open transition when the device must list them, and
   the Documents of a reprocess.
2. The **manifest** (:class:`AcquisitionManifest`) holds the scope listing
   (:class:`AcquiredListing`), the items ``discover`` yielded in order, and the
   listing of each excluded scope.
3. The **manifest response** (:class:`AcquisitionRequests`) names what the
   server run will ask beyond discovery, by the server run's own rules: the
   unlisted Units to ``confirm_absent``, the failed Documents to
   ``rediscover``, and the failed items to ``fetch``.
4. Each fetched item becomes one **acquired input**
   (:class:`AcquiredInputPackage`): the item as ``fetch`` left it, its bytes
   and content type, its PDF export and the artifacts it describes, or the
   error ``fetch`` raised. The **answers** (:class:`AcquisitionAnswers`) carry
   what ``confirm_absent`` and ``rediscover`` returned.

An error the device's acquisition raised travels as an
:class:`AcquisitionError`, and the recorded Gene raises it again where the
server run asks, so the run treats it as it treats that error from the
provider itself.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from memforge.genes.recorded import AcquisitionFailed
from memforge.llm.structured import failure_retryable
from memforge.local_agent.snapshot_manifest import SNAPSHOT_MANIFEST_MAX_ITEMS, ExcludedScopeListing
from memforge.local_agent.source_contract import DOCUMENT_ACQUISITION_OPERATION
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, ScopeListingKind
from memforge.source_artifacts import RawSourceArtifact

__all__ = [
    "ACQUISITION_ATTEMPT_PAYLOAD_FIELD",
    "CONFIG_PAYLOAD_FIELD",
    "DOCUMENT_ACQUISITION_OPERATION",
    "REPROCESS_PAYLOAD_FIELD",
    "SINCE_PAYLOAD_FIELD",
    "AcquiredArtifact",
    "AcquiredInputPackage",
    "AcquiredListing",
    "AcquisitionAnswers",
    "AcquisitionError",
    "AcquisitionManifest",
    "AcquisitionManifestTooLarge",
    "AcquisitionRequests",
    "StoredAcquisition",
    "acquisition_job_config",
    "expected_input_ids",
]

CONFIG_PAYLOAD_FIELD = "config"
"""The job payload field holding the configuration the device runs the Gene with, without secrets."""
SINCE_PAYLOAD_FIELD = "since"
"""The job payload field holding the instant discovery reads changes from, or ``None`` to discover everything."""
REPROCESS_PAYLOAD_FIELD = "reprocess"
"""The job payload field naming the Documents of a reprocess; such a job neither discovers nor lists."""
ACQUISITION_ATTEMPT_PAYLOAD_FIELD = "acquisition_attempt"
"""The job payload field counting the job's acquisition attempt: 1, and one more for each job that acquires
again after its predecessor's run failed on what its acquisition answered."""

_MAX_RETAINED_ERROR_CHARS = 512
"""The longest error text a failure keeps, the bound the sync's failure ledger keeps."""


def acquisition_job_config(source_type: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """The configuration a device runs a document Source's Gene with: every field but its secrets.

    The secrets are the type's secret fields and the stored forms the server
    keeps of each (encrypted value, configured flag, hint); the device holds
    its own credentials.
    """

    from memforge.genes import GENE_REGISTRY
    from memforge.source_secrets import source_secret_fields
    from memforge.source_types import source_type_descriptor

    descriptor = source_type_descriptor(source_type)
    secrets = set(source_secret_fields(source_type, GENE_REGISTRY))
    if descriptor is not None:
        secrets.update(descriptor.config.extra_secret_fields)
    stored_forms = {
        f"{secret}{suffix}"
        for secret in secrets
        for suffix in ("", "_encrypted", "_configured", "_hint", "_decrypt_failed")
    }
    return {key: value for key, value in config.items() if key not in stored_forms}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True, slots=True)
class AcquisitionError:
    """An error the device's acquisition raised, as the server run raises it again.

    ``failure_class`` is the qualified class of the error and ``message`` the
    text the failure ledger keeps of it. ``transient`` says whether it may
    pass on a later run. ``declared_retryable`` is the error's own
    ``retryable`` declaration, ``None`` when it made none, and ``error_code``
    its code, when it has one.
    """

    failure_class: str
    message: str
    transient: bool
    declared_retryable: bool | None = None
    error_code: str | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> AcquisitionError:
        detail = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else ""
        retained = detail.strip() if 0 < len(detail) <= _MAX_RETAINED_ERROR_CHARS else ""
        declared = getattr(exc, "retryable", None)
        error_code = getattr(exc, "error_code", None)
        return cls(
            failure_class=f"{type(exc).__module__}.{type(exc).__qualname__}",
            message=retained or f"{type(exc).__name__}: error details omitted",
            transient=failure_retryable(exc),
            declared_retryable=bool(declared) if declared is not None else None,
            error_code=error_code if isinstance(error_code, str) else None,
        )

    def raised(self) -> AcquisitionFailed:
        """The error to raise where the server run asks what acquisition answered."""

        return AcquisitionFailed(
            self.message,
            failure_class=self.failure_class,
            transient=self.transient,
            declared_retryable=self.declared_retryable,
            error_code=self.error_code,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "failure_class": self.failure_class,
            "message": self.message,
            "transient": self.transient,
            "declared_retryable": self.declared_retryable,
            "error_code": self.error_code,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquisitionError:
        declared = payload.get("declared_retryable")
        error_code = payload.get("error_code")
        return cls(
            failure_class=str(payload["failure_class"]),
            message=str(payload["message"]),
            transient=bool(payload["transient"]),
            declared_retryable=bool(declared) if declared is not None else None,
            error_code=str(error_code) if error_code is not None else None,
        )


def _error(payload: object) -> AcquisitionError | None:
    return AcquisitionError.from_payload(payload) if isinstance(payload, Mapping) else None


class AcquisitionManifestTooLarge(ValueError):
    """A listing or discovery holds more items than one device manifest may name.

    The Source stays that large, so its job fails without a retry.
    """

    retryable = False

    def __init__(self, what: str, count: int) -> None:
        super().__init__(
            f"the {what} holds {count} items, more than the {SNAPSHOT_MANIFEST_MAX_ITEMS} one device manifest may name"
        )


@dataclass(frozen=True, slots=True)
class AcquiredListing:
    """The listing acquisition took of the configured scope.

    ``kind`` and ``doc_ids`` are the listing ``list_scope`` returned;
    ``failure`` is the error it raised instead. Neither means the
    configuration does not list, so acquisition asked for none.
    """

    kind: ScopeListingKind | None = None
    doc_ids: frozenset[str] = frozenset()
    failure: AcquisitionError | None = None

    def __post_init__(self) -> None:
        if self.failure is not None and (self.kind is not None or self.doc_ids):
            raise ValueError("a failed listing lists nothing")
        if self.kind is None and self.doc_ids:
            raise ValueError("a listing names its kind")

    @classmethod
    def of(cls, listing: ScopeListing) -> AcquiredListing:
        return cls(kind=listing.kind, doc_ids=frozenset(listing.doc_ids))

    def scope_listing(self) -> ScopeListing | None:
        """The listing the run reads, ``None`` when acquisition took none or it failed."""

        return ScopeListing(kind=self.kind, doc_ids=self.doc_ids) if self.kind is not None else None

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value if self.kind is not None else None,
            "doc_ids": sorted(self.doc_ids),
            "failure": self.failure.to_payload() if self.failure is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquiredListing:
        kind = payload.get("kind")
        return cls(
            kind=ScopeListingKind(str(kind)) if kind is not None else None,
            doc_ids=frozenset(str(doc_id) for doc_id in payload.get("doc_ids") or ()),
            failure=_error(payload.get("failure")),
        )


def _items_payload(items: Sequence[ContentItem]) -> list[dict[str, object]]:
    return [item.to_payload() for item in items]


def _items(payload: object) -> tuple[ContentItem, ...]:
    return tuple(ContentItem.from_payload(dict(item)) for item in payload or ())  # type: ignore[union-attr]


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """What one device run's acquisition found before it fetched anything.

    ``listing`` is the run's scope listing, ``discovered`` the items
    ``discover(since)`` yielded in order, and ``excluded_listings`` what the
    device listed in each excluded scope its job carried, each listed by its
    own Gene. A reprocess discovers and lists nothing.
    """

    listing: AcquiredListing = AcquiredListing()
    discovered: tuple[ContentItem, ...] = ()
    excluded_listings: tuple[ExcludedScopeListing, ...] = ()

    def __post_init__(self) -> None:
        ids = [item.item_id for item in self.discovered]
        if len(set(ids)) != len(ids):
            raise ValueError("a manifest discovers each item once")
        keys = [listing.scope_key for listing in self.excluded_listings]
        if len(set(keys)) != len(keys):
            raise ValueError("a manifest lists each excluded scope once")

    def check_bound(self) -> None:
        """Refuse a manifest that names more items than one device manifest may."""

        if len(self.listing.doc_ids) > SNAPSHOT_MANIFEST_MAX_ITEMS:
            raise AcquisitionManifestTooLarge("scope listing", len(self.listing.doc_ids))
        if len(self.discovered) > SNAPSHOT_MANIFEST_MAX_ITEMS:
            raise AcquisitionManifestTooLarge("discovery", len(self.discovered))
        for listing in self.excluded_listings:
            if len(listing.doc_ids) > SNAPSHOT_MANIFEST_MAX_ITEMS:
                raise AcquisitionManifestTooLarge(f"listing of excluded scope {listing.scope_key}", len(listing.doc_ids))

    def to_payload(self) -> dict[str, object]:
        return {
            "listing": self.listing.to_payload(),
            "discovered": _items_payload(self.discovered),
            "excluded_listings": [
                {"scope_key": listing.scope_key, "doc_ids": sorted(listing.doc_ids), "rejection": listing.rejection}
                for listing in sorted(self.excluded_listings, key=lambda listing: listing.scope_key)
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquisitionManifest:
        return cls(
            listing=AcquiredListing.from_payload(payload.get("listing") or {}),
            discovered=_items(payload.get("discovered")),
            excluded_listings=tuple(
                ExcludedScopeListing(
                    scope_key=str(listing["scope_key"]),
                    doc_ids=frozenset(str(doc_id) for doc_id in listing.get("doc_ids") or ()),
                    rejection=str(listing["rejection"]) if listing.get("rejection") is not None else None,
                )
                for listing in payload.get("excluded_listings") or ()
            ),
        )

    def digest(self) -> str:
        """The digest that binds the manifest: a replay must carry the same content."""

        return hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class AcquisitionRequests:
    """What the server run will ask beyond discovery, which the device answers after its manifest.

    - ``confirm``: the items ``confirm_absent`` is asked about;
    - ``rediscover``: the items ``rediscover`` is asked with, each fetched
      when the provider still returns it;
    - ``fetch``: the items ``fetch`` is asked for.
    """

    confirm: tuple[ContentItem, ...] = ()
    rediscover: tuple[ContentItem, ...] = ()
    fetch: tuple[ContentItem, ...] = ()

    @property
    def awaits_answers(self) -> bool:
        """Whether the device answers ``confirm`` or ``rediscover`` before the run can start."""

        return bool(self.confirm or self.rediscover)

    def to_payload(self) -> dict[str, object]:
        return {
            "confirm": _items_payload(self.confirm),
            "rediscover": _items_payload(self.rediscover),
            "fetch": _items_payload(self.fetch),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquisitionRequests:
        return cls(
            confirm=_items(payload.get("confirm")),
            rediscover=_items(payload.get("rediscover")),
            fetch=_items(payload.get("fetch")),
        )


@dataclass(frozen=True, slots=True)
class AcquisitionAnswers:
    """What ``confirm_absent`` and ``rediscover`` returned for the items a manifest response named.

    ``absence`` holds one answer per ``confirm`` item, unless
    ``confirm_absent`` raised, which ``absence_failure`` then holds.
    ``rediscovered`` maps each ``rediscover`` item's id to the item the
    provider returned, ``None`` when it returned none, unless ``rediscover``
    raised for it, which ``rediscover_failures`` then holds.
    """

    absence: Mapping[str, AbsenceAnswer] = field(default_factory=dict)
    absence_failure: AcquisitionError | None = None
    rediscovered: Mapping[str, ContentItem | None] = field(default_factory=dict)
    rediscover_failures: Mapping[str, AcquisitionError] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.absence_failure is not None and self.absence:
            raise ValueError("a failed confirmation answers no item")
        if set(self.rediscovered) & set(self.rediscover_failures):
            raise ValueError("a rediscovery is answered or failed, not both")
        object.__setattr__(self, "absence", MappingProxyType(dict(self.absence)))
        object.__setattr__(self, "rediscovered", MappingProxyType(dict(self.rediscovered)))
        object.__setattr__(self, "rediscover_failures", MappingProxyType(dict(self.rediscover_failures)))

    def check_answers(self, requests: AcquisitionRequests) -> None:
        """Refuse answers that do not answer exactly what ``requests`` asked."""

        confirm = {item.item_id for item in requests.confirm}
        if self.absence_failure is None and set(self.absence) != confirm:
            raise ValueError("the answers must confirm exactly the items the manifest response named")
        if self.absence_failure is not None and not confirm:
            raise ValueError("a confirmation failed that was never asked for")
        rediscover = {item.item_id for item in requests.rediscover}
        if set(self.rediscovered) | set(self.rediscover_failures) != rediscover:
            raise ValueError("the answers must rediscover exactly the items the manifest response named")

    def fetched_rediscoveries(self) -> frozenset[str]:
        """The rediscovered Documents the device fetched: each one the provider returned under its own id."""

        return frozenset(
            item_id for item_id, item in self.rediscovered.items() if item is not None and item.item_id == item_id
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "absence": {item_id: answer.value for item_id, answer in sorted(self.absence.items())},
            "absence_failure": self.absence_failure.to_payload() if self.absence_failure is not None else None,
            "rediscovered": {
                item_id: item.to_payload() if item is not None else None
                for item_id, item in sorted(self.rediscovered.items())
            },
            "rediscover_failures": {
                item_id: failure.to_payload() for item_id, failure in sorted(self.rediscover_failures.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquisitionAnswers:
        return cls(
            absence={
                str(item_id): AbsenceAnswer(str(answer)) for item_id, answer in (payload.get("absence") or {}).items()
            },
            absence_failure=_error(payload.get("absence_failure")),
            rediscovered={
                str(item_id): ContentItem.from_payload(dict(item)) if item is not None else None
                for item_id, item in (payload.get("rediscovered") or {}).items()
            },
            rediscover_failures={
                str(item_id): AcquisitionError.from_payload(failure)
                for item_id, failure in (payload.get("rediscover_failures") or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class AcquiredArtifact:
    """One artifact an acquired input describes: the uploaded input holding its body, or the error reading it raised.

    ``input_sha256`` names the uploaded input that holds the body. ``failure``
    is the error opening or reading the body raised instead, which the server
    run raises again where it opens the artifact, so the Document fails as it
    fails when the server reads the provider itself.
    """

    artifact: RawSourceArtifact
    input_sha256: str | None = None
    failure: AcquisitionError | None = None

    def __post_init__(self) -> None:
        if (self.input_sha256 is None) == (self.failure is None):
            raise ValueError("an acquired artifact holds its uploaded body or its failure")

    def to_payload(self) -> dict[str, object]:
        artifact = self.artifact
        return {
            "provider_key": artifact.provider_key,
            "parent_observation_type": artifact.parent_observation_type,
            "parent_provider_key": artifact.parent_provider_key,
            "provider_revision": artifact.provider_revision,
            "filename": artifact.filename,
            "media_type": artifact.media_type,
            "declared_size_bytes": artifact.declared_size_bytes,
            "locator": dict(artifact.locator),
            "input_sha256": self.input_sha256,
            "failure": self.failure.to_payload() if self.failure is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquiredArtifact:
        size = payload.get("declared_size_bytes")
        input_sha256 = payload.get("input_sha256")
        return cls(
            artifact=RawSourceArtifact(
                provider_key=str(payload["provider_key"]),
                parent_observation_type=str(payload["parent_observation_type"]),
                parent_provider_key=str(payload["parent_provider_key"]),
                provider_revision=str(payload["provider_revision"]),
                filename=str(payload["filename"]),
                media_type=str(payload["media_type"]),
                declared_size_bytes=int(size) if size is not None else None,
                locator=dict(payload.get("locator") or {}),
            ),
            input_sha256=str(input_sha256) if input_sha256 is not None else None,
            failure=_error(payload.get("failure")),
        )


@dataclass(frozen=True, slots=True)
class AcquiredInputPackage:
    """What ``fetch`` returned for one item on the device: the input its Unit stores when it commits.

    ``item`` is the item as ``fetch`` left it, or as ``fetch`` was asked for
    it when ``failure`` holds the error ``fetch`` raised. ``body`` and
    ``content_type`` are the bytes ``fetch`` returned. ``pdf`` is the item's
    PDF export when its type exports PDFs, or ``pdf_failure`` the error that
    export raised. ``artifacts`` are the bodies ``fetch`` described, in its
    order, each uploaded before the package that names it or failed.
    """

    item: ContentItem
    body: bytes = b""
    content_type: str = ""
    pdf: bytes | None = None
    pdf_failure: AcquisitionError | None = None
    artifacts: tuple[AcquiredArtifact, ...] = ()
    failure: AcquisitionError | None = None

    def __post_init__(self) -> None:
        if self.failure is not None and (self.body or self.pdf is not None or self.pdf_failure or self.artifacts):
            raise ValueError("a failed fetch acquired nothing")

    def to_payload(self) -> dict[str, object]:
        return {
            "item": self.item.to_payload(),
            "body": base64.b64encode(self.body).decode("ascii"),
            "content_type": self.content_type,
            "pdf": base64.b64encode(self.pdf).decode("ascii") if self.pdf is not None else None,
            "pdf_failure": self.pdf_failure.to_payload() if self.pdf_failure is not None else None,
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "failure": self.failure.to_payload() if self.failure is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AcquiredInputPackage:
        pdf = payload.get("pdf")
        return cls(
            item=ContentItem.from_payload(dict(payload["item"])),
            body=base64.b64decode(str(payload.get("body") or "")),
            content_type=str(payload.get("content_type") or ""),
            pdf=base64.b64decode(str(pdf)) if pdf is not None else None,
            pdf_failure=_error(payload.get("pdf_failure")),
            artifacts=tuple(AcquiredArtifact.from_payload(artifact) for artifact in payload.get("artifacts") or ()),
            failure=_error(payload.get("failure")),
        )

    def encode(self) -> bytes:
        """The package as it is stored: canonical JSON, so the same input is the same bytes."""

        return _canonical_json(self.to_payload())

    @classmethod
    def decode(cls, body: bytes) -> AcquiredInputPackage:
        return cls.from_payload(json.loads(body.decode("utf-8")))


def expected_input_ids(
    manifest: AcquisitionManifest, requests: AcquisitionRequests, answers: AcquisitionAnswers | None
) -> frozenset[str]:
    """The items one device run must upload an acquired input for before the server run can start.

    Every discovered item and every ``fetch`` item, and each ``rediscover``
    item the provider returned under its own id.
    """

    ids: set[str] = {item.item_id for item in manifest.discovered}
    ids.update(item.item_id for item in requests.fetch)
    if answers is not None:
        ids.update(answers.fetched_rediscoveries())
    return frozenset(ids)


@dataclass(frozen=True, slots=True)
class StoredAcquisition:
    """One device run's acquisition as the server keeps it.

    ``snapshot_id`` names the job attempt of the Source in
    ``workspace_id``. ``requests`` are what the manifest response asked,
    computed when the attempt's manifest is first recorded and returned
    unchanged on a replay, and ``answers`` the device's answers, ``None``
    until it answers.
    ``started_at`` is when the attempt was leased: the device read the
    provider from then on, so it is the run's start for the sync cursor.
    """

    workspace_id: str
    snapshot_id: str
    manifest: AcquisitionManifest
    manifest_sha256: str
    requests: AcquisitionRequests
    answers: AcquisitionAnswers | None
    started_at: datetime
    local_agent_job_id: str
    local_agent_attempt_count: int
    source_config_revision: str
