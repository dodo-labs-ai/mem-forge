"""Schema and canonical hashing for retrieval golden case sets."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from memforge.storage.adapters.context import AccessScope


class CaseSetValidationError(ValueError):
    """Raised when a retrieval golden case set violates its contract."""


_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RetrievalScope:
    """AccessScope-shaped request context from a golden case."""

    raw: Mapping[str, Any]

    def to_access_scope(self) -> AccessScope:
        return AccessScope(
            user_id=str(self.raw["user_id"]),
            include_private=bool(self.raw["include_private"]),
            allowed_statuses=tuple(str(value) for value in self.raw["allowed_statuses"]),
            active_project=self.raw["active_project"],
            scope_mode=self.raw["scope_mode"],
            active_repo_identifier=self.raw.get("active_repo_identifier"),
        )

    def to_data(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class ExpectedSpec:
    """Expected behavior for one retrieval golden case."""

    relevant: Mapping[str, int]
    max_rank: Mapping[str, int]
    required_channels: Mapping[str, tuple[str, ...]]
    required_profile: str | None = None
    total_candidates: int | None = None

    def with_required_channels(
        self,
        memory_id: str,
        channels: tuple[str, ...],
    ) -> ExpectedSpec:
        required_channels = dict(self.required_channels)
        required_channels[memory_id] = channels
        return replace(self, required_channels=required_channels)

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> ExpectedSpec:
        payload = data or {}
        _reject_unknown_fields(
            "expected",
            payload,
            {
                "relevant",
                "max_rank",
                "required_channels",
                "required_profile",
                "total_candidates",
            },
        )
        relevant_payload = payload.get("relevant", {})
        max_rank_payload = payload.get("max_rank", {})
        required_channels_payload = payload.get("required_channels", {})
        _require_mapping("expected.relevant", relevant_payload)
        _require_mapping("expected.max_rank", max_rank_payload)
        _require_mapping("expected.required_channels", required_channels_payload)
        if "required_profile" in payload and payload["required_profile"] is not None:
            _require_str("expected.required_profile", payload["required_profile"])
        if "total_candidates" in payload and payload["total_candidates"] is not None:
            _require_int("expected.total_candidates", payload["total_candidates"])
        for memory_id, channels in dict(required_channels_payload).items():
            _require_str("expected.required_channels.memory_id", memory_id)
            _require_sequence_of_str(f"expected.required_channels.{memory_id}", channels)
        required_channels = {
            str(memory_id): tuple(str(channel) for channel in channels)
            for memory_id, channels in dict(required_channels_payload).items()
        }
        return cls(
            relevant={
                str(key): _require_int(f"expected.relevant.{key}", value)
                for key, value in dict(relevant_payload).items()
            },
            max_rank={
                str(key): _require_int(f"expected.max_rank.{key}", value)
                for key, value in dict(max_rank_payload).items()
            },
            required_channels=required_channels,
            required_profile=payload.get("required_profile"),
            total_candidates=payload.get("total_candidates"),
        )

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "relevant": dict(self.relevant),
        }
        if self.max_rank:
            data["max_rank"] = dict(self.max_rank)
        if self.required_channels:
            data["required_channels"] = {
                memory_id: list(channels)
                for memory_id, channels in self.required_channels.items()
            }
        if self.required_profile is not None:
            data["required_profile"] = self.required_profile
        if self.total_candidates is not None:
            data["total_candidates"] = self.total_candidates
        return data


@dataclass(frozen=True)
class RetrievalCase:
    """One deterministic retrieval golden case."""

    id: str
    family: str
    description: str
    query: str
    top_k: int
    fixture_variant: str
    scope: RetrievalScope
    expected: ExpectedSpec
    offset: int = 0
    source_filter: Mapping[str, Any] | None = None
    time_range: Mapping[str, Any] | None = None
    entities: tuple[str, ...] = ()
    control_case_id: str | None = None
    parity_gate: str | None = None

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> RetrievalCase:
        _reject_unknown_fields(
            "case",
            data,
            {
                "id",
                "family",
                "description",
                "query",
                "top_k",
                "offset",
                "source_filter",
                "time_range",
                "entities",
                "fixture_variant",
                "scope",
                "expected",
                "control_case_id",
                "parity_gate",
            },
        )
        _require_str("case.id", data.get("id"))
        _require_str("case.family", data.get("family"))
        _require_str("case.description", data.get("description"))
        query = _require_present("case.query", data, "query")
        top_k = _require_present("case.top_k", data, "top_k")
        offset = data["offset"] if "offset" in data else 0
        entities = _require_present("case.entities", data, "entities")
        fixture_variant = _require_present("case.fixture_variant", data, "fixture_variant")
        _require_str("case.query", query)
        _require_positive_int("case.top_k", top_k)
        _require_non_negative_int("case.offset", offset)
        if data.get("source_filter") is not None:
            _require_mapping("case.source_filter", data["source_filter"])
            _reject_unknown_fields(
                "case.source_filter",
                data["source_filter"],
                {"source_ids", "clients", "repo_identifiers"},
            )
        if data.get("time_range") is not None:
            _require_mapping("case.time_range", data["time_range"])
            _reject_unknown_fields(
                "case.time_range",
                data["time_range"],
                {"after", "before", "date_type"},
            )
        _require_sequence("case.entities", entities)
        _require_str("case.fixture_variant", fixture_variant)
        _require_mapping("case.scope", data.get("scope"))
        _require_mapping("case.expected", data.get("expected"))
        if data.get("control_case_id") is not None:
            _require_str("case.control_case_id", data["control_case_id"])
        if data.get("parity_gate") is not None:
            _require_str("case.parity_gate", data["parity_gate"])
        return cls(
            id=str(data["id"]),
            family=str(data["family"]),
            description=str(data["description"]),
            query=query,
            top_k=top_k,
            offset=offset,
            source_filter=data.get("source_filter"),
            time_range=data.get("time_range"),
            entities=tuple(str(entity) for entity in entities),
            fixture_variant=fixture_variant,
            scope=RetrievalScope(dict(data["scope"])),
            expected=ExpectedSpec.from_data(data.get("expected")),
            control_case_id=data.get("control_case_id"),
            parity_gate=data.get("parity_gate"),
        )

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "description": self.description,
            "query": self.query,
            "top_k": self.top_k,
            "fixture_variant": self.fixture_variant,
            "scope": self.scope.to_data(),
            "expected": self.expected.to_data(),
        }
        if self.offset:
            data["offset"] = self.offset
        if self.source_filter is not None:
            data["source_filter"] = self.source_filter
        if self.time_range is not None:
            data["time_range"] = self.time_range
        if self.entities:
            data["entities"] = list(self.entities)
        if self.control_case_id is not None:
            data["control_case_id"] = self.control_case_id
        if self.parity_gate is not None:
            data["parity_gate"] = self.parity_gate
        return data


@dataclass(frozen=True)
class CaseSetManifest:
    """Version and fixture manifest for a retrieval golden case set."""

    case_schema_version: int
    case_set_id: str
    case_set_sha: str
    case_files: tuple[str, ...]
    fixtures: Mapping[str, Any]

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> CaseSetManifest:
        _reject_unknown_fields(
            "manifest",
            data,
            {
                "case_schema_version",
                "case_set_id",
                "case_set_sha",
                "case_files",
                "fixtures",
            },
        )
        _require_positive_int("manifest.case_schema_version", data.get("case_schema_version"))
        _require_str("manifest.case_set_id", data.get("case_set_id"))
        _require_str("manifest.case_set_sha", data.get("case_set_sha"))
        _require_sequence("manifest.case_files", data.get("case_files"))
        _require_mapping("manifest.fixtures", data.get("fixtures"))
        return cls(
            case_schema_version=int(data["case_schema_version"]),
            case_set_id=str(data["case_set_id"]),
            case_set_sha=str(data["case_set_sha"]),
            case_files=tuple(str(value) for value in data["case_files"]),
            fixtures=copy.deepcopy(dict(data["fixtures"])),
        )

    def to_data(self, *, include_sha: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "case_schema_version": self.case_schema_version,
            "case_set_id": self.case_set_id,
            "case_files": list(self.case_files),
            "fixtures": copy.deepcopy(dict(self.fixtures)),
        }
        if include_sha:
            data["case_set_sha"] = self.case_set_sha
        return data


@dataclass(frozen=True)
class RetrievalCaseSet:
    """Loaded and validated retrieval golden cases plus their manifest."""

    manifest: CaseSetManifest
    cases_by_file: Mapping[str, tuple[RetrievalCase, ...]]

    @property
    def cases(self) -> tuple[RetrievalCase, ...]:
        return tuple(
            case
            for file_name in self.manifest.case_files
            for case in self.cases_by_file.get(file_name, ())
        )

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.id for case in self.cases)

    def get_case(self, case_id: str) -> RetrievalCase:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(case_id)

    def with_manifest_sha(self, case_set_sha: str) -> RetrievalCaseSet:
        return replace(
            self,
            manifest=replace(self.manifest, case_set_sha=case_set_sha),
        )

    def replace_case(self, case_id: str, **changes: Any) -> RetrievalCaseSet:
        cases_by_file: dict[str, tuple[RetrievalCase, ...]] = {}
        found = False
        for file_name, cases in self.cases_by_file.items():
            updated_cases: list[RetrievalCase] = []
            for case in cases:
                if case.id == case_id:
                    updated_cases.append(replace(case, **changes))
                    found = True
                else:
                    updated_cases.append(case)
            cases_by_file[file_name] = tuple(updated_cases)
        if not found:
            raise KeyError(case_id)
        return replace(self, cases_by_file=cases_by_file)

    def to_canonical_data(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_data(include_sha=False),
            "cases": {
                file_name: [
                    case.to_data()
                    for case in self.cases_by_file.get(file_name, ())
                ]
                for file_name in self.manifest.case_files
            },
        }


def load_case_set(
    case_set_id: str = "retrieval-core-v1",
    *,
    case_root: Path | None = None,
    verify_sha: bool = True,
) -> RetrievalCaseSet:
    case_dir = case_root or _case_resource_root()
    manifest = _load_yaml_mapping(case_dir.joinpath("manifest.yaml").read_text(encoding="utf-8"))
    if manifest.get("case_set_id") != case_set_id:
        raise CaseSetValidationError(
            f"Case set {case_set_id!r} is not packaged; found {manifest.get('case_set_id')!r}"
        )
    cases_by_file = {
        file_name: _load_yaml_sequence(case_dir.joinpath(file_name).read_text(encoding="utf-8"))
        for file_name in manifest["case_files"]
    }
    return load_case_set_from_data(manifest, cases_by_file, verify_sha=verify_sha)


def load_case_set_from_data(
    manifest_data: Mapping[str, Any],
    cases_by_file_data: Mapping[str, list[Mapping[str, Any]]],
    *,
    verify_sha: bool = False,
) -> RetrievalCaseSet:
    _require_mapping("manifest", manifest_data)
    manifest = CaseSetManifest.from_data(manifest_data)
    cases_by_file = {
        file_name: tuple(RetrievalCase.from_data(case) for case in cases_by_file_data[file_name])
        for file_name in manifest.case_files
    }
    case_set = RetrievalCaseSet(manifest=manifest, cases_by_file=cases_by_file)
    _validate_case_set(case_set)
    if verify_sha and not validate_case_set_sha(case_set):
        raise CaseSetValidationError(
            f"case_set_sha mismatch for {manifest.case_set_id}: "
            f"expected {manifest.case_set_sha}, computed {compute_case_set_sha(case_set)}"
        )
    return case_set


def compute_case_set_sha(case_set: RetrievalCaseSet) -> str:
    payload = case_set.to_canonical_data()
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8") + b"\n"
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def validate_case_set_sha(case_set: RetrievalCaseSet) -> bool:
    return compute_case_set_sha(case_set) == case_set.manifest.case_set_sha


def _validate_case_set(case_set: RetrievalCaseSet) -> None:
    seen: set[str] = set()
    cases_by_id: dict[str, RetrievalCase] = {}
    for case in case_set.cases:
        if case.id in seen:
            raise CaseSetValidationError(f"Duplicate case id: {case.id}")
        seen.add(case.id)
        cases_by_id[case.id] = case
        _validate_case(case_set, case)

    for case in case_set.cases:
        if case.control_case_id is None:
            continue
        control_case = cases_by_id.get(case.control_case_id)
        if control_case is None:
            raise CaseSetValidationError(
                f"Case {case.id} references missing control_case_id {case.control_case_id}"
            )
        if control_case.fixture_variant != case.fixture_variant:
            raise CaseSetValidationError(
                f"Case {case.id} and control {case.control_case_id} use different fixture variants"
            )

    for fixture_name, fixture in case_set.manifest.fixtures.items():
        _validate_fixture_source_subscriptions(fixture_name, fixture)
        _validate_fixture_documents(fixture_name, fixture)


def _validate_case(case_set: RetrievalCaseSet, case: RetrievalCase) -> None:
    if not _CASE_ID_RE.fullmatch(case.id):
        raise CaseSetValidationError(
            f"Case id {case.id!r} must contain only letters, digits, dots, underscores, or hyphens"
        )
    if case.fixture_variant not in case_set.manifest.fixtures:
        raise CaseSetValidationError(
            f"Case {case.id} references missing fixture variant {case.fixture_variant}"
        )

    scope_keys = set(case.scope.raw)
    access_scope_keys = {field.name for field in fields(AccessScope)}
    if scope_keys != access_scope_keys:
        raise CaseSetValidationError(
            f"Case {case.id} scope keys must match AccessScope fields: "
            f"expected {sorted(access_scope_keys)}, got {sorted(scope_keys)}"
        )
    _validate_scope_shape(case.id, case.scope.raw)
    case.scope.to_access_scope()

    fixture = case_set.manifest.fixtures[case.fixture_variant]
    memory_ids = {
        str(memory["id"]) if isinstance(memory, Mapping) else str(memory)
        for memory in fixture.get("memories") or ()
    }
    expected_memory_ids = set(case.expected.relevant)
    expected_memory_ids.update(case.expected.max_rank)
    expected_memory_ids.update(case.expected.required_channels)
    missing_memory_ids = sorted(expected_memory_ids - memory_ids)
    if missing_memory_ids:
        raise CaseSetValidationError(
            f"Case {case.id} references memories absent from fixture "
            f"{case.fixture_variant}: {missing_memory_ids}"
        )


def _validate_fixture_source_subscriptions(fixture_name: str, fixture: Mapping[str, Any]) -> None:
    users = [str(user) for user in fixture.get("users") or ()]
    source_ids = [
        str(source["id"]) if isinstance(source, Mapping) else str(source)
        for source in fixture.get("sources") or ()
    ]
    subscription_rows = {
        (
            str(row.get("user_id")),
            str(row.get("source_id")),
        )
        for row in fixture.get("source_subscriptions") or ()
    }
    missing = [
        (user_id, source_id)
        for user_id in users
        for source_id in source_ids
        if (user_id, source_id) not in subscription_rows
    ]
    if missing:
        missing_text = ", ".join(f"{user_id}/{source_id}" for user_id, source_id in missing)
        raise CaseSetValidationError(
            f"Fixture {fixture_name} is missing explicit source_subscriptions rows: {missing_text}"
        )


def _validate_fixture_documents(fixture_name: str, fixture: Mapping[str, Any]) -> None:
    source_ids = {
        str(source["id"]) if isinstance(source, Mapping) else str(source)
        for source in fixture.get("sources") or ()
    }
    missing_document_source_ids = [
        str(document.get("doc_id") or "<unknown>")
        for document in fixture.get("documents") or ()
        if isinstance(document, Mapping) and "source_id" not in document
    ]
    if missing_document_source_ids:
        raise CaseSetValidationError(
            f"Fixture {fixture_name} documents are missing source_id: {missing_document_source_ids}"
        )

    missing_source_ids = sorted(
        str(document.get("source_id"))
        for document in fixture.get("documents") or ()
        if isinstance(document, Mapping) and str(document.get("source_id")) not in source_ids
    )
    if missing_source_ids:
        raise CaseSetValidationError(
            f"Fixture {fixture_name} documents reference undeclared sources: {missing_source_ids}"
        )


def _validate_scope_shape(case_id: str, scope: Mapping[str, Any]) -> None:
    _require_str(f"case {case_id} scope.user_id", scope["user_id"])
    _require_bool(f"case {case_id} scope.include_private", scope["include_private"])
    _require_sequence_of_str(f"case {case_id} scope.allowed_statuses", scope["allowed_statuses"])
    if scope["active_project"] is not None:
        _require_str(f"case {case_id} scope.active_project", scope["active_project"])
    _require_str(f"case {case_id} scope.scope_mode", scope["scope_mode"])
    if scope["active_repo_identifier"] is not None:
        _require_str(
            f"case {case_id} scope.active_repo_identifier",
            scope["active_repo_identifier"],
        )


def _case_resource_root() -> resources.abc.Traversable:
    return resources.files("memforge.evals.retrieval.cases")


def _reject_unknown_fields(context: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CaseSetValidationError(f"Unknown fields in {context}: {unknown}")


def _require_present(field_name: str, data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise CaseSetValidationError(f"{field_name} is required")
    return data[key]


def _require_mapping(field_name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaseSetValidationError(f"{field_name} must be a mapping")
    return value


def _require_sequence(field_name: str, value: Any) -> list[Any] | tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise CaseSetValidationError(f"{field_name} must be a sequence")
    return value


def _require_sequence_of_str(field_name: str, value: Any) -> tuple[str, ...]:
    sequence = _require_sequence(field_name, value)
    for item in sequence:
        _require_str(field_name, item)
    return tuple(sequence)


def _require_str(field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise CaseSetValidationError(f"{field_name} must be a string")
    return value


def _require_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseSetValidationError(f"{field_name} must be an integer")
    return value


def _require_positive_int(field_name: str, value: Any) -> int:
    integer = _require_int(field_name, value)
    if integer <= 0:
        raise CaseSetValidationError(f"{field_name} must be positive")
    return integer


def _require_non_negative_int(field_name: str, value: Any) -> int:
    integer = _require_int(field_name, value)
    if integer < 0:
        raise CaseSetValidationError(f"{field_name} must be non-negative")
    return integer


def _require_bool(field_name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise CaseSetValidationError(f"{field_name} must be a boolean")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CaseSetValidationError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(data, dict):
        raise CaseSetValidationError("Expected YAML mapping")
    return data


def _load_yaml_sequence(text: str) -> list[dict[str, Any]]:
    data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(data, list):
        raise CaseSetValidationError("Expected YAML sequence")
    if not all(isinstance(item, dict) for item in data):
        raise CaseSetValidationError("Expected YAML sequence of mappings")
    return data
