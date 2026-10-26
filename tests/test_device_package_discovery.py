"""Device-collected connectors discover submitted packages through the shared toolkit.

Local Markdown, GitHub repository local push and Jira local agent read the
same manifest and inbox, and a package without its own time carries the time
it was submitted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memforge.genes.document_gene import DocumentGene
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.jira_gene import JiraGene
from memforge.genes.local_markdown_gene import LocalMarkdownGene
from memforge.models import ContentItem
from memforge.pipeline.sync import _same_provider_revision

SUBMITTED_AT = datetime(2026, 7, 7, 9, 30, tzinfo=timezone.utc)
REPO_URL = "https://github.example.test/payroll/architecture"


@dataclass(frozen=True)
class Connector:
    """How one device-collected connector is configured, and what its packages hold."""

    name: str
    package_kind: str
    package_fields: dict
    gene: Callable[[dict], DocumentGene]
    inbox_key: str


CONNECTORS = [
    Connector(
        name="local_markdown",
        package_kind="local_markdown_document",
        package_fields={"vault_id": "notes", "relative_path": "ideas.md", "content_type": "text/markdown"},
        gene=lambda config: LocalMarkdownGene(config, "src-local"),
        inbox_key="documents_dir",
    ),
    Connector(
        name="github_repo",
        package_kind="github_repo_document",
        package_fields={"repo_url": REPO_URL, "repo_ref": "main", "relative_path": "README.md"},
        gene=lambda config: GitHubRepoGene({"connection_mode": "local_push", "repo_url": REPO_URL, **config}, "src-repo"),
        inbox_key="documents_dir",
    ),
    Connector(
        name="jira",
        package_kind="jira_document",
        package_fields={"issue_key": "PAY-1", "content_type": "application/json"},
        gene=lambda config: JiraGene(
            {"base_url": "https://jira.example.test", "sync_mode": "local_agent", **config},
            "src-jira",
        ),
        inbox_key="local_agent_documents_dir",
    ),
]


def _package(connector: Connector, *, submitted_at: object = SUBMITTED_AT.isoformat(), version: str = "v1") -> dict:
    """A package that names only its submission time, as ``package_time`` allows."""
    return {
        "doc_id": f"{connector.name}-doc",
        "title": "Payroll notes",
        "source_url": f"local://{connector.name}/doc",
        "space_or_project": "PAY",
        "version": version,
        "submitted_at": submitted_at,
        "submitted_by": "riley",
        **connector.package_fields,
    }


def _manifest_gene(connector: Connector, tmp_path: Path, package: dict) -> DocumentGene:
    entry = {**package, "package_uri": f"artifact://{connector.name}/doc", "input_sha256": "c" * 64}
    return connector.gene({connector.inbox_key: str(tmp_path), "local_agent_package_manifest": [entry]})


def _inbox_gene(connector: Connector, tmp_path: Path, package: dict) -> DocumentGene:
    (tmp_path / "doc.json").write_text(json.dumps({"package_kind": connector.package_kind, **package}), encoding="utf-8")
    return connector.gene({connector.inbox_key: str(tmp_path)})


async def _discover(gene: DocumentGene, since: datetime | None = None) -> list[ContentItem]:
    return [item async for item in gene.discover(since)]


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS, ids=lambda connector: connector.name)
@pytest.mark.parametrize("source", ["manifest", "inbox"])
async def test_a_package_without_a_time_carries_its_submission_time_in_utc(connector, source, tmp_path):
    package = _package(connector, submitted_at="2026-07-07T11:30:00+02:00")
    gene = (_manifest_gene if source == "manifest" else _inbox_gene)(connector, tmp_path, package)

    [item] = await _discover(gene)

    assert item.item_id == f"{connector.name}-doc"
    assert item.last_modified == SUBMITTED_AT
    assert item.last_modified.tzinfo is timezone.utc
    assert item.author == "riley"
    if source == "manifest":
        assert item.extra["package_uri"] == f"artifact://{connector.name}/doc"
        assert item.extra["input_sha256"] == "c" * 64
    else:
        assert item.extra["package_path"] == str(tmp_path / "doc.json")


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS, ids=lambda connector: connector.name)
@pytest.mark.parametrize("source", ["manifest", "inbox"])
async def test_a_naive_package_time_is_read_as_utc(connector, source, tmp_path):
    package = {**_package(connector), "last_modified": "2026-07-07T09:30:00"}
    gene = (_manifest_gene if source == "manifest" else _inbox_gene)(connector, tmp_path, package)

    [item] = await _discover(gene, since=SUBMITTED_AT - timedelta(minutes=1))

    assert item.last_modified == SUBMITTED_AT


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS, ids=lambda connector: connector.name)
async def test_an_incremental_run_skips_a_package_submitted_at_or_before_its_cursor(connector, tmp_path):
    gene = _manifest_gene(connector, tmp_path, _package(connector))

    assert await _discover(gene, since=SUBMITTED_AT) == []
    assert len(await _discover(gene, since=SUBMITTED_AT - timedelta(seconds=1))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS, ids=lambda connector: connector.name)
async def test_an_unchanged_package_without_a_time_is_the_same_provider_revision_in_every_run(connector, tmp_path):
    """A failed package the device has not resubmitted is left alone; a resubmitted one is processed again.

    The failure ledger stores the failed item through ``to_payload`` and the
    next run compares what discovery returns with it.
    """
    package = _package(connector)
    [failed] = await _discover(_manifest_gene(connector, tmp_path, package))
    recorded = ContentItem.from_payload(failed.to_payload())

    [unchanged] = await _discover(_manifest_gene(connector, tmp_path, package))
    resubmitted_package = _package(connector, submitted_at=(SUBMITTED_AT + timedelta(hours=1)).isoformat())
    [resubmitted] = await _discover(_manifest_gene(connector, tmp_path, resubmitted_package))

    assert _same_provider_revision(unchanged, recorded) is True
    assert _same_provider_revision(resubmitted, recorded) is False
