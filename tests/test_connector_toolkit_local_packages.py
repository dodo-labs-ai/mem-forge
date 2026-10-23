"""The connector toolkit's reading of device packages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memforge.genes.toolkit.local_packages import DevicePackage, device_packages, package_time

KIND = "local_markdown_document"
SUBMITTED = "2026-07-07T09:30:00+00:00"
SUBMITTED_AT = datetime(2026, 7, 7, 9, 30, tzinfo=timezone.utc)


def _no_inbox() -> Path:
    raise AssertionError("a run with a manifest never reads the inbox")


def _write(folder: Path, name: str, package: object) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"last_modified": "2026-07-07T11:30:00+02:00", "submitted_at": "2026-07-08T00:00:00+00:00"}, SUBMITTED_AT),
        ({"last_modified": "2026-07-07T09:30:00Z"}, SUBMITTED_AT),
        ({"last_modified": "2026-07-07T09:30:00"}, SUBMITTED_AT),
        ({"last_modified": datetime(2026, 7, 7, 9, 30)}, SUBMITTED_AT),
        ({"submitted_at": "2026-07-07T11:30:00+02:00"}, SUBMITTED_AT),
        ({"last_modified": "", "submitted_at": SUBMITTED}, SUBMITTED_AT),
        ({"last_modified": None, "submitted_at": "2026-07-07T09:30:00"}, SUBMITTED_AT),
    ],
)
def test_a_package_time_is_its_last_modified_or_else_its_submission_time_in_utc(fields, expected):
    parsed = package_time(fields)

    assert parsed == expected
    assert parsed.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "fields",
    [{"doc_id": "doc-a"}, {"doc_id": "doc-a", "last_modified": "", "submitted_at": ""}],
)
def test_a_package_without_any_time_is_refused(fields):
    with pytest.raises(ValueError, match="doc-a has neither last_modified nor submitted_at"):
        package_time(fields)


def test_a_package_time_that_is_not_iso_8601_is_refused():
    with pytest.raises(ValueError):
        package_time({"last_modified": "yesterday"})


def test_manifest_entries_are_the_packages_with_their_locator():
    config = {
        "local_agent_package_manifest": [
            {
                "doc_id": "doc-a",
                "submitted_at": SUBMITTED,
                "package_uri": "artifact://a",
                "package_path": "/inbox/a.json",
                "package_sha256": "a" * 64,
                "input_sha256": "b" * 64,
                "relative_path": "a.md",
            },
        ],
    }

    [package] = device_packages(config, package_kind=KIND, inbox=_no_inbox)

    assert package.last_modified == SUBMITTED_AT
    assert package.fields["relative_path"] == "a.md"
    assert package.locator == {
        "package_uri": "artifact://a",
        "package_path": "/inbox/a.json",
        "package_sha256": "a" * 64,
        "input_sha256": "b" * 64,
    }


def test_a_manifest_entry_without_a_package_uri_is_not_a_package():
    config = {
        "local_agent_package_manifest": [
            {"doc_id": "no-uri", "submitted_at": SUBMITTED},
            {"doc_id": "blank-uri", "submitted_at": SUBMITTED, "package_uri": "  "},
            "not an entry",
            {"doc_id": "kept", "submitted_at": SUBMITTED, "package_uri": "artifact://kept"},
        ],
    }

    packages = device_packages(config, package_kind=KIND, inbox=_no_inbox)

    assert [package.text("doc_id") for package in packages] == ["kept"]


def test_an_empty_manifest_is_a_run_without_packages_and_never_reads_the_inbox():
    assert device_packages({"local_agent_package_manifest": []}, package_kind=KIND, inbox=_no_inbox) == []


def test_packages_are_ordered_by_their_utc_time_and_then_doc_id():
    config = {
        "local_agent_package_manifest": [
            # Its text sorts first, but its time is the latest.
            {"doc_id": "a-late", "last_modified": "2026-07-07T08:00:00-05:00", "package_uri": "u1"},
            {"doc_id": "c-same", "last_modified": "2026-07-07T10:00:00+00:00", "package_uri": "u2"},
            {"doc_id": "b-same", "submitted_at": "2026-07-07T12:00:00+02:00", "package_uri": "u3"},
            {"doc_id": "d-early", "last_modified": "2026-07-07T09:00:00", "package_uri": "u4"},
        ],
    }

    packages = device_packages(config, package_kind=KIND, inbox=_no_inbox)

    assert [package.text("doc_id") for package in packages] == ["d-early", "b-same", "c-same", "a-late"]


def test_inbox_files_of_the_kind_are_the_packages_when_there_is_no_manifest(tmp_path, caplog):
    nested = _write(
        tmp_path,
        "vault/notes/b.json",
        {"package_kind": KIND, "doc_id": "doc-b", "last_modified": "2026-07-07T10:00:00+00:00"},
    )
    stamped = _write(tmp_path, "a.json", {"package_kind": KIND, "doc_id": "doc-a", "submitted_at": SUBMITTED})
    _write(tmp_path, "other-kind.json", {"package_kind": "jira_document", "doc_id": "jira", "submitted_at": SUBMITTED})
    _write(tmp_path, "list.json", [{"package_kind": KIND}])
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="memforge.genes.toolkit.local_packages"):
        packages = device_packages({}, package_kind=KIND, inbox=lambda: tmp_path)

    assert packages == [
        DevicePackage(
            fields={"package_kind": KIND, "doc_id": "doc-a", "submitted_at": SUBMITTED},
            last_modified=SUBMITTED_AT,
            locator={"package_path": str(stamped)},
        ),
        DevicePackage(
            fields={"package_kind": KIND, "doc_id": "doc-b", "last_modified": "2026-07-07T10:00:00+00:00"},
            last_modified=datetime(2026, 7, 7, 10, tzinfo=timezone.utc),
            locator={"package_path": str(nested)},
        ),
    ]
    assert "broken.json" in caplog.text


def test_an_inbox_package_without_any_time_is_refused(tmp_path):
    _write(tmp_path, "a.json", {"package_kind": KIND, "doc_id": "doc-a"})

    with pytest.raises(ValueError, match="doc-a has neither last_modified nor submitted_at"):
        device_packages({}, package_kind=KIND, inbox=lambda: tmp_path)


def test_a_source_without_an_inbox_raises_what_its_inbox_raises():
    def missing_inbox() -> Path:
        raise ValueError("source is missing documents_dir")

    with pytest.raises(ValueError, match="missing documents_dir"):
        device_packages({}, package_kind=KIND, inbox=missing_inbox)


@pytest.mark.parametrize(
    ("since", "expected"),
    [
        (None, True),
        (datetime(2026, 7, 7, 9, 29, tzinfo=timezone.utc), True),
        (SUBMITTED_AT, False),
        (datetime(2026, 7, 7, 9, 31, tzinfo=timezone.utc), False),
    ],
)
def test_an_incremental_run_reads_only_packages_after_its_cursor(since, expected):
    package = DevicePackage(fields={}, last_modified=SUBMITTED_AT, locator={})

    assert package.changed_after(since) is expected


def test_text_reads_a_missing_or_empty_field_as_its_default():
    package = DevicePackage(fields={"title": "", "version": 7, "doc_id": "doc-a"}, last_modified=SUBMITTED_AT, locator={})

    assert package.text("title", "doc-a") == "doc-a"
    assert package.text("version") == "7"
    assert package.text("source_url") == ""
