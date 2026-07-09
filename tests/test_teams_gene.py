"""Tests for the Microsoft Teams gene."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from memforge.genes.teams_gene import (
    AuthenticationError,
    TeamsGene,
    _TeamsAPIClient,
    _group_into_blocks,
)
from memforge.models import ConfigFieldType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(id: str, author: str, content: str, time: datetime, parent_id: str | None = None) -> dict:
    """Build a mock Teams message dict."""
    return {
        "id": id,
        "from": author,
        "content": content,
        "time": time,
        "parentMessageId": parent_id,
        "mentions": [],
        "attachments": [],
    }


NOW = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)


class FakeDocumentStore:
    def __init__(self):
        self._objects: dict[str, bytes] = {}

    def store_raw(self, source_name, title, content, content_type, extension=None):
        uri = f"fake://{source_name}/{title}{extension or ''}"
        self._objects[uri] = content
        return uri

    def read_artifact(self, uri):
        return self._objects[uri]


# ---------------------------------------------------------------------------
# Gene metadata and config
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_fields(self):
        meta = TeamsGene.metadata()
        assert meta.name == "teams"
        assert meta.display_name == "Microsoft Teams"
        assert meta.data_shape == "message"
        assert meta.auth_method == "browser_cookie"
        assert meta.default_sync_interval_minutes == 60

    def test_config_schema_groups(self):
        schema = TeamsGene.config_schema()
        group_keys = [g.key for g in schema.groups]
        assert group_keys == ["connection", "scope", "sync"]

    def test_config_schema_fields(self):
        schema = TeamsGene.config_schema()
        field_keys = [f.key for f in schema.fields]
        assert "region" in field_keys
        assert "channels" in field_keys
        assert "group_chats" in field_keys
        assert "individual_chats" in field_keys
        assert "max_age_days" in field_keys
        assert "conversation_gap_minutes" in field_keys
        assert "max_block_messages" in field_keys

    def test_conversation_gap_defaults_to_one_hour(self):
        schema = TeamsGene.config_schema()
        gap = next(f for f in schema.fields if f.key == "conversation_gap_minutes")
        assert gap.default == "60"

        gene = TeamsGene(
            config={"channels": "Team/Channel"},
            source_id="test",
        )
        assert gene._gap_minutes == 60

    def test_initial_history_defaults_to_two_weeks(self):
        schema = TeamsGene.config_schema()
        max_age = next(f for f in schema.fields if f.key == "max_age_days")
        assert max_age.default == "14"

        gene = TeamsGene(
            config={"channels": "Team/Channel"},
            source_id="test",
        )
        assert gene._max_age_days == 14

    def test_local_agent_package_mode_does_not_require_remote_selectors(self, tmp_path):
        gene = TeamsGene(
            config={"local_agent_documents_dir": str(tmp_path)},
            source_id="test",
        )

        assert gene._local_agent_documents_dir() == tmp_path

    @pytest.mark.asyncio
    async def test_discovers_local_agent_package_without_remote_selectors(self, tmp_path):
        package = {
            "package_kind": "teams_window_document",
            "doc_id": "teams-doc-1",
            "title": "PCC Agent Dev",
            "source_url": "https://teams.microsoft.com/l/message/19:conversation/1",
            "last_modified": NOW.isoformat(),
            "space_or_project": "PCC Agent Dev",
            "version": "sha256:revision-1",
            "conversation_id": "19:conversation@thread.v2",
            "root_message_id": "1",
            "window_id": "teams-block:v1:opaque",
            "window_type": "time_block",
            "revision_hash": "sha256:revision-1",
            "raw_payload": {
                "conversation_type": "group_chat",
                "messages": [{"id": "1", "from": "Ada", "content": "Ship it", "time": NOW.isoformat()}],
            },
        }
        (tmp_path / "teams-doc-1.json").write_text(json.dumps(package), encoding="utf-8")
        gene = TeamsGene(
            config={"local_agent_documents_dir": str(tmp_path)},
            source_id="test",
        )

        items = [item async for item in gene.discover(since=None)]

        assert len(items) == 1
        assert items[0].item_id == "teams-doc-1"
        assert items[0].extra["window_id"] == "teams-block:v1:opaque"

    @pytest.mark.asyncio
    async def test_discovers_and_fetches_document_store_package_manifest(self):
        package = {
            "package_kind": "teams_window_document",
            "doc_id": "teams-doc-1",
            "title": "PCC Agent Dev",
            "source_url": "https://teams.microsoft.com/l/message/19:conversation/1",
            "last_modified": NOW.isoformat(),
            "space_or_project": "PCC Agent Dev",
            "version": "sha256:revision-1",
            "conversation_id": "19:conversation@thread.v2",
            "root_message_id": "1",
            "window_id": "teams-block:v1:opaque",
            "window_type": "time_block",
            "revision_hash": "sha256:revision-1",
            "raw_payload": {
                "conversation_type": "group_chat",
                "messages": [{"id": "1", "from": "Ada", "content": "Ship it", "time": NOW.isoformat()}],
            },
        }
        store = FakeDocumentStore()
        package_uri = store.store_raw(
            "PCC Agent Dev",
            "teams-doc-1-package",
            json.dumps(package).encode("utf-8"),
            "application/json",
            extension=".teams-package.json",
        )
        gene = TeamsGene(
            config={
                "local_agent_package_manifest": [
                    {
                        "doc_id": "teams-doc-1",
                        "title": "PCC Agent Dev",
                        "source_url": package["source_url"],
                        "last_modified": NOW.isoformat(),
                        "space_or_project": "PCC Agent Dev",
                        "version": "sha256:revision-1",
                        "conversation_id": "19:conversation@thread.v2",
                        "root_message_id": "1",
                        "window_id": "teams-block:v1:opaque",
                        "window_type": "time_block",
                        "revision_hash": "sha256:revision-1",
                        "package_uri": package_uri,
                    }
                ]
            },
            source_id="test",
        )
        gene.bind_document_store(store)

        items = [item async for item in gene.discover(since=None)]
        raw = await gene.fetch(items[0])
        normalized = await gene.normalize(raw)

        assert len(items) == 1
        assert items[0].extra["package_uri"] == package_uri
        assert b"teams_window_document" in raw.body
        assert "Ship it" in normalized.markdown_body

    def test_numeric_fields_use_integer_type(self):
        schema = TeamsGene.config_schema()
        numeric_fields = {"max_age_days", "conversation_gap_minutes", "max_block_messages"}
        for f in schema.fields:
            if f.key in numeric_fields:
                assert f.field_type == ConfigFieldType.INTEGER, f"Field {f.key} should be INTEGER"

    def test_config_validation_rejects_empty_scope(self):
        with pytest.raises(ValueError, match="At least one"):
            TeamsGene(config={"region": "emea"}, source_id="test")

    def test_config_accepts_channels_only(self):
        gene = TeamsGene(
            config={"channels": "Team/Channel"},
            source_id="test",
        )
        assert gene.source_id == "test"


# ---------------------------------------------------------------------------
# Conversation block grouping
# ---------------------------------------------------------------------------

class TestBlockGrouping:
    def test_empty_messages(self):
        assert _group_into_blocks([], gap_minutes=180, max_messages=100) == []

    def test_single_message(self):
        msgs = [_msg("1", "Alice", "Hello", NOW)]
        blocks = _group_into_blocks(msgs, gap_minutes=180, max_messages=100)
        assert len(blocks) == 1
        assert len(blocks[0]) == 1

    def test_messages_within_gap_form_one_block(self):
        msgs = [
            _msg("1", "Alice", "Hello", NOW),
            _msg("2", "Bob", "Hi there", NOW + timedelta(minutes=5)),
            _msg("3", "Alice", "How are you?", NOW + timedelta(minutes=10)),
        ]
        blocks = _group_into_blocks(msgs, gap_minutes=180, max_messages=100)
        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_gap_splits_into_two_blocks(self):
        msgs = [
            _msg("1", "Alice", "Morning discussion", NOW),
            _msg("2", "Bob", "Agreed", NOW + timedelta(minutes=30)),
            # 4-hour gap
            _msg("3", "Alice", "Afternoon update", NOW + timedelta(hours=5)),
            _msg("4", "Carol", "Got it", NOW + timedelta(hours=5, minutes=10)),
        ]
        blocks = _group_into_blocks(msgs, gap_minutes=180, max_messages=100)
        assert len(blocks) == 2
        assert len(blocks[0]) == 2
        assert len(blocks[1]) == 2

    def test_max_messages_splits_block(self):
        # Create 10 messages 1 minute apart, max_messages=4
        msgs = [
            _msg(str(i), "Alice", f"Message {i}", NOW + timedelta(minutes=i))
            for i in range(10)
        ]
        blocks = _group_into_blocks(msgs, gap_minutes=180, max_messages=4)
        assert len(blocks) >= 3
        for block in blocks:
            assert len(block) <= 4

    def test_messages_sorted_by_time(self):
        msgs = [
            _msg("3", "Carol", "Third", NOW + timedelta(minutes=20)),
            _msg("1", "Alice", "First", NOW),
            _msg("2", "Bob", "Second", NOW + timedelta(minutes=10)),
        ]
        blocks = _group_into_blocks(msgs, gap_minutes=180, max_messages=100)
        assert len(blocks) == 1
        assert blocks[0][0]["id"] == "1"
        assert blocks[0][1]["id"] == "2"
        assert blocks[0][2]["id"] == "3"

    def test_exact_gap_does_not_split_block(self):
        msgs = [
            _msg("1", "Alice", "First", NOW),
            _msg("2", "Bob", "Exactly one hour later", NOW + timedelta(minutes=60)),
        ]

        blocks = _group_into_blocks(msgs, gap_minutes=60, max_messages=100)

        assert [[msg["id"] for msg in block] for block in blocks] == [["1", "2"]]


class TestMessageParsing:
    @pytest.mark.asyncio
    async def test_request_raises_auth_error_without_retry_on_401(self):
        client = _TeamsAPIClient(region="emea")
        calls = 0

        class FakeClient:
            async def request(self, method: str, url: str, **kwargs):
                nonlocal calls
                calls += 1
                request = httpx.Request(method, f"https://teams.cloud.microsoft{url}")
                return httpx.Response(
                    401,
                    json={"errorCode": 911, "message": "Authentication failed."},
                    request=request,
                )

        with pytest.raises(AuthenticationError, match="Teams session expired"):
            await client._request(FakeClient(), "GET", "/conversations")

        assert calls == 1

    @pytest.mark.asyncio
    async def test_request_retries_remote_protocol_error(self, monkeypatch):
        client = _TeamsAPIClient(region="emea")
        calls = 0

        async def no_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr("memforge.pipeline.retry.asyncio.sleep", no_sleep)

        class FakeClient:
            async def request(self, method: str, url: str, **kwargs):
                nonlocal calls
                calls += 1
                request = httpx.Request(method, f"https://teams.cloud.microsoft{url}")
                if calls == 1:
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response.",
                        request=request,
                    )
                return httpx.Response(200, json={"conversations": []}, request=request)

        response = await client._request(FakeClient(), "GET", "/conversations")

        assert response.status_code == 200
        assert calls == 2

    @pytest.mark.asyncio
    async def test_validate_uses_retrying_request(self):
        client = _TeamsAPIClient(region="emea")
        client._ensure_clients = AsyncMock()
        client._chat_client = object()
        client._request = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"conversations": []},
                request=httpx.Request("GET", "https://teams.cloud.microsoft/conversations"),
            )
        )

        await client.validate()

        client._request.assert_awaited_once()

    def test_parse_message_preserves_real_rest_root_message_id(self):
        client = _TeamsAPIClient(region="emea")

        parsed = client._parse_message(
            {
                "id": "reply-1",
                "rootMessageId": "root-1",
                "conversationid": "19:channel@example",
                "imdisplayname": "Alice",
                "content": "<p>Confirmed</p>",
                "messagetype": "RichText/Html",
                "composetime": "2026-04-15T12:00:00Z",
            }
        )

        assert parsed is not None
        assert parsed["id"] == "reply-1"
        assert parsed["rootMessageId"] == "root-1"
        assert parsed["conversationid"] == "19:channel@example"

    @pytest.mark.asyncio
    async def test_get_messages_until_keeps_newer_messages_after_old_row_in_same_page(self):
        client = _TeamsAPIClient(region="emea")
        client._ensure_clients = AsyncMock()
        client._chat_client = object()

        old_time = NOW - timedelta(days=2)
        new_time = NOW
        cutoff = NOW - timedelta(days=1)

        response = MagicMock()
        response.json.return_value = {
            "_metadata": {
                "backwardLink": "https://teams.cloud.microsoft/opaque/backward",
            },
            "messages": [
                {
                    "id": "old",
                    "conversationid": "19:conversation@thread.tacv2",
                    "content": "<p>too old</p>",
                    "messagetype": "RichText/Html",
                    "composetime": old_time.isoformat(),
                },
                {
                    "id": "new",
                    "conversationid": "19:conversation@thread.tacv2",
                    "content": "<p>keep me</p>",
                    "messagetype": "RichText/Html",
                    "composetime": new_time.isoformat(),
                },
            ],
        }
        client._request = AsyncMock(return_value=response)

        messages = await client.get_messages_until("19:conversation@thread.tacv2", cutoff)

        assert [message["id"] for message in messages] == ["new"]
        audits = client.get_poll_audits()
        assert audits[0]["raw_messages_seen"] == 2
        assert audits[0]["selected_message_keys_seen"] == 1
        assert audits[0]["parse_filtered_messages"] == 1
        assert audits[0]["stop_reason"] == "cutoff_reached"

    def test_poll_audit_records_raw_page_counts_without_message_content(self):
        client = _TeamsAPIClient(region="emea")

        parsed_messages = [
            {
                "id": "1783500000001",
                "conversationid": "19:conversation@thread.tacv2",
                "rootMessageId": "1783500000000",
                "content": "normalized content",
                "time": NOW,
            },
            {
                "id": "1783500000002",
                "conversationid": "19:conversation@thread.tacv2",
                "rootMessageId": "1783500000000",
                "content": "second normalized content",
                "time": NOW + timedelta(minutes=1),
            },
        ]
        client._record_message_poll_page(
            "19:conversation@thread.tacv2",
            {
                "_metadata": {"backwardLink": "https://teams.cloud.microsoft/opaque/backward"},
                "messages": [
                    {
                        "id": "1783500000001",
                        "conversationid": "19:conversation@thread.tacv2",
                        "rootMessageId": "1783500000000",
                        "content": "<p>normalized content</p>",
                        "composetime": NOW.isoformat(),
                    },
                    {
                        "id": "1783500000001",
                        "conversationid": "19:conversation@thread.tacv2",
                        "rootMessageId": "1783500000000",
                        "content": "<p>duplicate page row</p>",
                        "composetime": NOW.isoformat(),
                    },
                    {
                        "id": "1783500000002",
                        "conversationid": "19:conversation@thread.tacv2",
                        "rootMessageId": "1783500000000",
                        "content": "<p>second normalized content</p>",
                        "composetime": (NOW + timedelta(minutes=1)).isoformat(),
                    },
                ],
            },
            parsed_messages,
        )
        client.record_poll_ledger_actions(
            "19:conversation@thread.tacv2",
            {"new": 1, "updated": 0, "unchanged": 1},
        )
        client.mark_poll_complete("19:conversation@thread.tacv2", stop_reason="gap_boundary")

        audits = client.get_poll_audits()

        assert len(audits) == 1
        assert audits[0]["raw_conversation_id"] == "19:conversation@thread.tacv2"
        assert audits[0]["raw_messages_seen"] == 3
        assert audits[0]["unique_message_keys_seen"] == 2
        assert audits[0]["duplicate_raw_messages"] == 1
        assert audits[0]["page_count"] == 1
        assert audits[0]["metadata_backward_link"] == "https://teams.cloud.microsoft/opaque/backward"
        assert audits[0]["upsert_new"] == 1
        assert audits[0]["upsert_unchanged"] == 1
        assert "normalized content" not in json.dumps(audits[0])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscover:
    @pytest.fixture
    def gene(self):
        gene = TeamsGene(
            config={"channels": "Engineering/architecture"},
            source_id="teams-test",
        )
        gene._client = MagicMock(spec=_TeamsAPIClient)
        return gene

    @pytest.mark.asyncio
    async def test_skips_inactive_conversations(self, gene):
        """Conversations with no activity since last sync are skipped entirely."""
        gene._client.list_conversations = AsyncMock(return_value=[{
            "id": "19:abc@thread.tacv2",
            "topic": "architecture",
            "lastActivity": NOW - timedelta(hours=2),
            "type": "channel",
        }])
        gene._resolve_channel = AsyncMock(return_value="19:abc@thread.tacv2")

        since = NOW - timedelta(hours=1)
        items = []
        async for item in gene.discover(since=since):
            items.append(item)

        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_discovers_threads_from_channel(self, gene):
        """Messages with rootMessageId are grouped into threads."""
        conv_id = "19:abc@thread.tacv2"
        gene._client.list_conversations = AsyncMock(return_value=[{
            "id": conv_id,
            "topic": "architecture",
            "lastActivity": NOW,
            "type": "channel",
            "channel_name": "architecture",
            "team_name": "Engineering",
        }])
        gene._resolve_channel = AsyncMock(return_value=conv_id)

        root = _msg("root1", "Alice", "Should we use gRPC?", NOW - timedelta(minutes=30))
        root["rootMessageId"] = "root1"
        reply1 = _msg("r1", "Bob", "Yes, gRPC fits well", NOW - timedelta(minutes=20))
        reply1["rootMessageId"] = "root1"
        reply2 = _msg("r2", "Alice", "Confirmed, going with gRPC", NOW - timedelta(minutes=10))
        reply2["rootMessageId"] = "root1"

        gene._fetch_with_context = AsyncMock(return_value=[root, reply1, reply2])

        items = []
        async for item in gene.discover(since=None):
            items.append(item)

        assert len(items) == 1
        assert "root1" in items[0].item_id
        assert items[0].extra["is_thread"] is True
        assert items[0].extra["message_count"] == 3

    @pytest.mark.asyncio
    async def test_discovers_blocks_from_flat_messages(self, gene):
        """Unthreaded messages are grouped into conversation blocks."""
        conv_id = "19:abc@thread.tacv2"
        gene._client.list_conversations = AsyncMock(return_value=[{
            "id": conv_id,
            "topic": "architecture",
            "lastActivity": NOW,
            "type": "channel",
            "channel_name": "architecture",
            "team_name": "Engineering",
        }])
        gene._resolve_channel = AsyncMock(return_value=conv_id)

        msgs = [
            _msg("1", "Alice", "Morning standup notes", NOW - timedelta(hours=5)),
            _msg("2", "Bob", "I'll take the migration task", NOW - timedelta(hours=5) + timedelta(minutes=5)),
            # Gap of 4 hours
            _msg("3", "Carol", "Afternoon: deploy looks good", NOW - timedelta(hours=1)),
            _msg("4", "Alice", "Confirmed, merging", NOW - timedelta(minutes=30)),
        ]
        gene._fetch_with_context = AsyncMock(return_value=msgs)

        items = []
        async for item in gene.discover(since=None):
            items.append(item)

        assert len(items) == 2  # two blocks split by 4-hour gap
        assert all(item.extra["is_thread"] is False for item in items)

    @pytest.mark.asyncio
    async def test_discover_times_out_stuck_conversation_fetch(self, gene):
        conv_id = "19:abc@thread.tacv2"
        gene.config = {"group_chats": [conv_id]}
        gene._conversation_fetch_timeout_seconds = 1
        gene._client.list_conversations = AsyncMock(return_value=[{
            "id": conv_id,
            "topic": "architecture",
            "lastActivity": NOW,
            "type": "group_chat",
        }])
        gene._client.mark_poll_complete = MagicMock()

        async def never_returns(conv_id, since):
            await asyncio.sleep(30)
            return []

        gene._fetch_with_context = never_returns

        with pytest.raises(RuntimeError, match="Teams sync could not fetch any configured conversations"):
            async for _ in gene.discover(since=None):
                pass

        gene._client.mark_poll_complete.assert_called_once_with(
            conv_id,
            stop_reason="fetch_timeout",
        )

    @pytest.mark.asyncio
    async def test_discover_skips_one_timed_out_conversation_and_continues(self, gene):
        stuck_id = "19:stuck@thread.v2"
        ok_id = "19:ok@thread.v2"
        gene.config = {"group_chats": [stuck_id, ok_id]}
        gene._conversation_fetch_timeout_seconds = 1
        gene._client.list_conversations = AsyncMock(return_value=[
            {
                "id": stuck_id,
                "topic": "stuck",
                "lastActivity": NOW,
                "type": "group_chat",
            },
            {
                "id": ok_id,
                "topic": "ok",
                "lastActivity": NOW,
                "type": "group_chat",
            },
        ])
        gene._client.mark_poll_complete = MagicMock()

        async def fetch(conv_id, since):
            if conv_id == stuck_id:
                await asyncio.sleep(30)
                return []
            return [_msg("m1", "Alice", "Ship the change", NOW)]

        gene._fetch_with_context = fetch

        items = []
        async for item in gene.discover(since=None):
            items.append(item)

        assert len(items) == 1
        assert items[0].extra["conversation_id"] == ok_id
        gene._client.mark_poll_complete.assert_called_once_with(
            stuck_id,
            stop_reason="fetch_timeout",
        )

    @pytest.mark.asyncio
    async def test_discovers_blocks_with_persisted_ledger_anchor_for_late_messages(self, tmp_path):
        conv_id = "19:abc@thread.tacv2"
        config = {
            "group_chats": [conv_id],
            "conversation_gap_minutes": 60,
            "ledger_state_path": str(tmp_path / "teams-ledger.json"),
        }
        gene = TeamsGene(config=config, source_id="teams-test")
        gene._client = MagicMock(spec=_TeamsAPIClient)
        gene._client.list_conversations = AsyncMock(return_value=[{
            "id": conv_id,
            "topic": "architecture",
            "lastActivity": NOW,
            "type": "group_chat",
        }])

        first_messages = [
            _msg("m2", "Alice", "Anchor", NOW),
            _msg("m3", "Bob", "Follow-up", NOW + timedelta(minutes=30)),
        ]
        gene._fetch_with_context = AsyncMock(return_value=first_messages)
        first_items = [item async for item in gene.discover(since=None)]
        assert len(first_items) == 1
        first_item = first_items[0]

        second_messages = [
            _msg("m1", "Carol", "Late earlier", NOW - timedelta(minutes=30)),
            _msg("m2", "Alice", "Anchor", NOW),
            _msg("m3", "Bob", "Follow-up", NOW + timedelta(minutes=30)),
        ]
        gene._fetch_with_context = AsyncMock(return_value=second_messages)
        second_items = [item async for item in gene.discover(since=None)]

        assert len(second_items) == 1
        assert second_items[0].item_id == first_item.item_id
        assert second_items[0].extra["window_id"] == first_item.extra["window_id"]
        assert second_items[0].extra["root_message_id"] == "m2"
        assert second_items[0].extra["block_start"] == (NOW - timedelta(minutes=30)).isoformat()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_thread_returns_json(self):
        gene = TeamsGene(config={"channels": "T/C"}, source_id="test")
        gene._client = MagicMock(spec=_TeamsAPIClient)

        thread_msgs = [
            _msg("root", "Alice", "Question about API", NOW - timedelta(minutes=10)),
            _msg("r1", "Bob", "Here's the answer", NOW),
        ]
        gene._client.get_thread_messages = AsyncMock(return_value=thread_msgs)

        from memforge.models import ContentItem
        item = ContentItem(
            item_id="teams-conv#root",
            title="Test thread",
            source_url="https://teams.microsoft.com/l/message/conv/root",
            last_modified=NOW,
            content_type="application/json",
            extra={
                "conversation_id": "conv",
                "root_message_id": "root",
                "conversation_type": "channel",
                "is_thread": True,
            },
        )

        raw = await gene.fetch(item)
        assert raw.content_type == "application/json"
        data = json.loads(raw.body)
        assert len(data["messages"]) == 2
        assert "Alice" in data["participants"]
        assert "Bob" in data["participants"]


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    @pytest.mark.asyncio
    async def test_normalize_markdown_format(self):
        gene = TeamsGene(config={"channels": "T/C"}, source_id="test")
        gene._client = MagicMock()

        from memforge.models import ContentItem, RawContent

        thread_data = {
            "conversation_type": "channel",
            "title": "#architecture: Should we use gRPC?",
            "channel_name": "architecture",
            "team_name": "Engineering",
            "messages": [
                {"id": "root", "from": "Alice", "content": "Should we use gRPC?", "time": "2026-03-15T10:23:00+00:00", "is_root": True, "mentions": [], "attachments": []},
                {"id": "r1", "from": "Bob", "content": "Yes, gRPC fits our streaming needs", "time": "2026-03-15T10:45:00+00:00", "is_root": False, "mentions": [], "attachments": []},
            ],
            "participants": ["Alice", "Bob"],
            "first_message_time": "2026-03-15T10:23:00+00:00",
            "last_message_time": "2026-03-15T10:45:00+00:00",
        }

        item = ContentItem(
            item_id="teams-conv#root",
            title="#architecture: Should we use gRPC?",
            source_url="",
            last_modified=NOW,
        )
        raw = RawContent(
            item=item,
            body=json.dumps(thread_data).encode("utf-8"),
            content_type="application/json",
        )

        result = await gene.normalize(raw)

        # Check header
        assert "# #architecture: Should we use gRPC?" in result.markdown_body
        assert "**Channel**: architecture (Engineering)" in result.markdown_body
        assert "**Participants**: Alice, Bob" in result.markdown_body
        assert "**Messages**: 2" in result.markdown_body

        # Check messages
        assert "**Alice** (2026-03-15T10:23" in result.markdown_body
        assert "Should we use gRPC?" in result.markdown_body
        assert "> **Bob** (2026-03-15T10:45" in result.markdown_body
        assert "gRPC fits our streaming needs" in result.markdown_body

    @pytest.mark.asyncio
    async def test_normalize_source_semantics(self):
        gene = TeamsGene(config={"channels": "T/C"}, source_id="test")
        gene._client = MagicMock()

        from memforge.models import ContentItem, RawContent

        thread_data = {
            "conversation_type": "channel",
            "title": "Test",
            "channel_name": "general",
            "team_name": "PAY",
            "messages": [
                {"id": "1", "from": "Alice", "content": "Hello https://example.com", "time": "2026-04-15T10:00:00+00:00", "is_root": True, "mentions": [], "attachments": []},
            ],
            "participants": ["Alice"],
            "first_message_time": "2026-04-15T10:00:00+00:00",
            "last_message_time": "2026-04-15T10:00:00+00:00",
        }

        item = ContentItem(item_id="test", title="Test", source_url="", last_modified=NOW)
        raw = RawContent(item=item, body=json.dumps(thread_data).encode(), content_type="application/json")

        result = await gene.normalize(raw)
        sem = result.source_semantics

        assert sem["conversation_type"] == "channel"
        assert sem["channel_name"] == "general"
        assert sem["team_name"] == "PAY"
        assert sem["participants"] == ["Alice"]
        assert sem["message_count"] == 1
        assert sem["has_links"] is True
        assert sem["has_code_blocks"] is False


# ---------------------------------------------------------------------------
# Content hash (unchanged thread = same normalized markdown = same hash)
# ---------------------------------------------------------------------------

class TestContentHash:
    @pytest.mark.asyncio
    async def test_same_messages_produce_same_markdown(self):
        gene = TeamsGene(config={"channels": "T/C"}, source_id="test")
        gene._client = MagicMock()

        from memforge.models import ContentItem, RawContent

        thread_data = {
            "conversation_type": "channel", "title": "Test", "channel_name": "ch",
            "team_name": "T", "participants": ["A"],
            "first_message_time": "2026-04-15T10:00:00+00:00",
            "last_message_time": "2026-04-15T10:00:00+00:00",
            "messages": [{"id": "1", "from": "A", "content": "Hello", "time": "2026-04-15T10:00:00+00:00", "is_root": True, "mentions": [], "attachments": []}],
        }
        item = ContentItem(item_id="test", title="Test", source_url="", last_modified=NOW)
        raw = RawContent(item=item, body=json.dumps(thread_data).encode(), content_type="application/json")

        result1 = await gene.normalize(raw)
        result2 = await gene.normalize(raw)

        assert result1.markdown_body == result2.markdown_body
