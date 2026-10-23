"""The incremental watermark: the next run reads every item changed since this run began discovering."""

from __future__ import annotations

from datetime import datetime, timezone

from memforge.pipeline.collection_family import DEFAULT_INCREMENTAL_SYNC_OVERLAP
from tests.test_document_absence import JiraProject, workspace  # noqa: F401  (fixture)


class TimedJiraProject(JiraProject):
    """Records when each run's discovery began."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.discovery_began: list[datetime] = []

    async def discover(self, since=None):
        self.discovery_began.append(datetime.now(timezone.utc))
        async for item in super().discover(since=since):
            yield item


async def test_the_next_run_reads_from_before_the_previous_discovery_began(workspace):  # noqa: F811
    gene = TimedJiraProject(existing={1, 2})

    await workspace.sync(gene)
    state = await workspace.db.get_sync_state("src-document-absence")
    gene.move_on(existing={1, 2}, changed={2})
    await workspace.sync(gene)

    assert state is not None and state.last_sync_at is not None
    assert state.last_sync_at <= gene.discovery_began[0]
    assert gene.seen_since[1] == state.last_sync_at - DEFAULT_INCREMENTAL_SYNC_OVERLAP
    assert gene.fetched == ["jira-2"]
