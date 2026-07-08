"""Microsoft Teams Gene — syncs channel messages, group chats, and DMs.

Wraps the Teams Chat API and Microsoft Graph API to discover, fetch, and
normalize conversation threads and message blocks into comprehensive markdown
for memory extraction.

Authentication extracts OAuth tokens from Chrome browser cookies via the
``memforge auth teams`` CLI command. Tokens are cached at
``~/.memforge/tokens/teams.json``.

Document granularity:
- Threaded channel messages (rootMessageId) → one thread = one document
- Unthreaded messages → grouped into conversation blocks by time gaps
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx

from memforge.genes.base import Gene
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)
from memforge.pipeline.normalizer_utils import html_to_markdown

logger = logging.getLogger(__name__)

__all__ = ["TeamsGene"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_API_AUDIENCE = "https://ic3.teams.office.com"
_GRAPH_API_AUDIENCE = "https://graph.microsoft.com"

_REGION_URLS = {
    "emea": "https://teams.cloud.microsoft/api/chatsvc/emea/v1/users/ME",
    "amer": "https://teams.cloud.microsoft/api/chatsvc/amer/v1/users/ME",
    "apac": "https://teams.cloud.microsoft/api/chatsvc/apac/v1/users/ME",
}
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_MAX_PAGE_SIZE = 200


# ============================================================================
# Internal API Client
# ============================================================================

class _TeamsAPIClient:
    """Async HTTP client for Teams Chat API + Microsoft Graph API.

    Loads OAuth tokens via TeamsAuthenticator and provides typed methods
    for conversations, messages, teams, channels, and user resolution.
    """

    def __init__(self, region: str = "emea") -> None:
        self._region = region
        self._chat_base = _REGION_URLS.get(region, _REGION_URLS["emea"])
        self._chat_client: httpx.AsyncClient | None = None
        self._graph_client: httpx.AsyncClient | None = None

    async def _load_tokens(self) -> tuple[str, str]:
        """Load chat and graph tokens.

        Search order:
        1. ~/.memforge/tokens/teams.json (cached)
        2. Chrome cookies (live extraction)
        """
        from memforge.auth.teams_auth import (
            TeamsAuthenticator,
        )

        tokens = TeamsAuthenticator.load_tokens()
        if not tokens:
            raise AuthenticationError(
                "No Teams tokens found. Run: memforge auth teams"
            )

        now = datetime.now(timezone.utc).timestamp()
        chat_token = None
        graph_token = None

        for audience, entry in tokens.items():
            token_str = entry.get("token") if isinstance(entry, dict) else entry
            expires_at = entry.get("expiresAt", 0) if isinstance(entry, dict) else 0

            if expires_at and expires_at < now:
                continue  # skip expired tokens

            if not token_str:
                continue

            if _CHAT_API_AUDIENCE in audience:
                chat_token = token_str
            elif audience in ("https://graph.microsoft.com", "00000003-0000-0ff1-ce00-000000000000"):
                graph_token = token_str

        if not chat_token:
            raise AuthenticationError(
                "Teams Chat API token not found or expired. Run: memforge auth teams"
            )
        if not graph_token:
            # Graph is optional — some operations work without it
            logger.warning("Graph API token not found — team/channel resolution may fail")
            graph_token = chat_token  # fallback: use chat token (may fail for Graph calls)

        return chat_token, graph_token

    async def _ensure_clients(self) -> None:
        """Create HTTP clients if not already initialized."""
        if self._chat_client and self._graph_client:
            return

        chat_token, graph_token = await self._load_tokens()

        self._chat_client = httpx.AsyncClient(
            base_url=self._chat_base,
            headers={"Authorization": f"Bearer {chat_token}"},
            timeout=30.0,
        )
        self._graph_client = httpx.AsyncClient(
            base_url=_GRAPH_BASE,
            headers={"Authorization": f"Bearer {graph_token}"},
            timeout=30.0,
        )

    async def validate(self) -> None:
        """Verify tokens are valid with a lightweight probe."""
        await self._ensure_clients()
        try:
            resp = await self._chat_client.get("/conversations", params={"pageSize": 1})
            if resp.status_code == 401:
                raise AuthenticationError(
                    "Teams API returned 401. Run: memforge auth teams"
                )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise AuthenticationError(f"Teams API probe failed: {e}") from e

    async def close(self) -> None:
        """Close HTTP clients."""
        if self._chat_client:
            await self._chat_client.aclose()
        if self._graph_client:
            await self._graph_client.aclose()

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry on 429/5xx."""
        from memforge.pipeline.retry import retry_async

        async def _do_request() -> httpx.Response:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    "Rate limited", request=resp.request, response=resp,
                )
            resp.raise_for_status()
            return resp

        return await retry_async(
            _do_request,
            max_retries=3,
            retryable_exceptions=(httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout),
            description=f"Teams API {method} {url}",
        )

    # --- Chat API methods ---

    async def get_user_favorites(self) -> set[str]:
        """Get conversation IDs pinned as favorites from user properties."""
        await self._ensure_clients()
        try:
            resp = await self._request(
                self._chat_client, "GET", "/properties",
            )
            data = resp.json()
            fav_raw = data.get("favorites", {})
            if isinstance(fav_raw, str):
                fav_raw = json.loads(fav_raw)
            return set(fav_raw.keys()) if isinstance(fav_raw, dict) else set()
        except Exception:
            logger.debug("Failed to fetch user favorites", exc_info=True)
            return set()

    async def list_conversations(self) -> list[dict]:
        """List all conversations (channels + chats)."""
        await self._ensure_clients()
        resp = await self._request(
            self._chat_client, "GET", "/conversations",
            params={"view": "msnp24Equivalent", "pageSize": _MAX_PAGE_SIZE},
        )
        data = resp.json()
        return self._parse_conversations(data)

    async def list_channels(self) -> list[dict]:
        """List all channel conversations grouped by team.

        Returns list of team dicts: {id, displayName, channels: [{id, displayName}]}
        Uses the Chat API conversationType=channel filter and extracts
        team names from spaceThreadTopic.
        """
        await self._ensure_clients()
        resp = await self._request(
            self._chat_client, "GET", "/conversations",
            params={
                "view": "msnp24Equivalent",
                "pageSize": _MAX_PAGE_SIZE,
                "conversationType": "channel",
            },
        )
        data = resp.json()
        raw_list = data.get("conversations", data) if isinstance(data, dict) else data
        if not isinstance(raw_list, list):
            return []

        # Group channels by groupId, extract team name from spaceThreadTopic
        teams: dict[str, dict] = {}
        for c in raw_list:
            conv_id = c.get("id", "")
            if not conv_id:
                continue
            tp = c.get("threadProperties", {})
            props = c.get("properties", {})
            group_id = tp.get("groupId", "")
            if not group_id:
                continue

            topic = tp.get("topic", "") or tp.get("topicThreadTopic", "")
            space_topic = tp.get("spaceThreadTopic", "")
            is_favorite = str(props.get("favorite", "")).lower() == "true"

            if group_id not in teams:
                teams[group_id] = {
                    "id": group_id,
                    "displayName": space_topic or "Unknown Team",
                    "channels": [],
                    "favorite": False,
                }
            elif space_topic and teams[group_id]["displayName"] == "Unknown Team":
                teams[group_id]["displayName"] = space_topic

            # A team is favorite if any of its channels (typically General) is marked favorite
            if is_favorite:
                teams[group_id]["favorite"] = True

            if topic:
                teams[group_id]["channels"].append({
                    "id": conv_id,
                    "displayName": topic,
                    "favorite": is_favorite,
                })

        return list(teams.values())

    async def get_messages(
        self, conversation_id: str, page_size: int = _MAX_PAGE_SIZE,
    ) -> list[dict]:
        """Fetch all messages for a conversation, paginated."""
        await self._ensure_clients()
        all_messages: list[dict] = []
        url = f"/conversations/{conversation_id}/messages"
        params = {"pageSize": page_size}

        while url:
            resp = await self._request(self._chat_client, "GET", url, params=params)
            data = resp.json()
            messages = data.get("messages", data) if isinstance(data, dict) else data

            if isinstance(messages, list):
                for m in messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        all_messages.append(parsed)

            # Pagination: check for next link
            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link

        return all_messages

    async def get_messages_until(
        self, conversation_id: str, cutoff: datetime,
    ) -> list[dict]:
        """Fetch messages backward until hitting cutoff timestamp."""
        await self._ensure_clients()
        all_messages: list[dict] = []
        url = f"/conversations/{conversation_id}/messages"
        params: dict = {"pageSize": _MAX_PAGE_SIZE}
        page_count = 0

        while url:
            # Rate-limit: Teams allows 15 req/10s — pace at ~1 req/s
            if page_count > 0:
                await asyncio.sleep(1.0)
            resp = await self._request(self._chat_client, "GET", url, params=params)
            page_count += 1
            data = resp.json()
            messages = data.get("messages", data) if isinstance(data, dict) else data

            hit_cutoff = False
            if isinstance(messages, list):
                for m in messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        if parsed["time"] < cutoff:
                            hit_cutoff = True
                            break
                        all_messages.append(parsed)

            if hit_cutoff:
                break

            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link

        return all_messages

    async def get_thread_messages(
        self, conversation_id: str, root_message_id: str,
    ) -> list[dict]:
        """Fetch all messages in a thread (root + replies)."""
        await self._ensure_clients()
        url = f"/conversations/{conversation_id};messageid={root_message_id}/messages"
        resp = await self._request(
            self._chat_client, "GET", url,
            params={"pageSize": _MAX_PAGE_SIZE},
        )

        data = resp.json()
        messages = data.get("messages", data) if isinstance(data, dict) else data
        result = []
        if isinstance(messages, list):
            for m in messages:
                parsed = self._parse_message(m)
                if parsed:
                    result.append(parsed)

        return result

    async def paginate_messages_backward(
        self, conversation_id: str,
    ) -> AsyncIterator[list[dict]]:
        """Yield pages of messages from newest to oldest."""
        await self._ensure_clients()
        url = f"/conversations/{conversation_id}/messages"
        params: dict = {"pageSize": _MAX_PAGE_SIZE}
        page_count = 0

        while url:
            if page_count > 0:
                await asyncio.sleep(1.0)
            resp = await self._request(self._chat_client, "GET", url, params=params)
            page_count += 1
            data = resp.json()
            raw_messages = data.get("messages", data) if isinstance(data, dict) else data

            page: list[dict] = []
            if isinstance(raw_messages, list):
                for m in raw_messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        page.append(parsed)

            if page:
                yield page

            # Next page via backward link
            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link

    # --- Graph API methods ---

    async def get_joined_teams(self) -> list[dict]:
        """List teams the authenticated user belongs to."""
        await self._ensure_clients()
        resp = await self._graph_client.get("/me/joinedTeams")
        resp.raise_for_status()
        return resp.json().get("value", [])

    async def get_team_channels(self, team_id: str) -> list[dict]:
        """List channels in a team."""
        await self._ensure_clients()
        resp = await self._graph_client.get(f"/teams/{team_id}/channels")
        resp.raise_for_status()
        return resp.json().get("value", [])

    async def resolve_user(self, query: str) -> dict | None:
        """Resolve a user by display name via Graph API people search."""
        await self._ensure_clients()
        try:
            resp = await self._graph_client.get(
                "/me/people",
                params={"$search": f'"{query}"', "$top": 5},
            )
            resp.raise_for_status()
            results = resp.json().get("value", [])
            if results:
                person = results[0]
                return {
                    "id": person.get("id", ""),
                    "displayName": person.get("displayName", query),
                }

            # Fallback: directory search
            resp = await self._graph_client.get(
                "/users",
                params={"$search": f'"displayName:{query}"', "$top": 5},
                headers={"ConsistencyLevel": "eventual"},
            )
            resp.raise_for_status()
            users = resp.json().get("value", [])
            if users:
                return {
                    "id": users[0].get("id", ""),
                    "displayName": users[0].get("displayName", query),
                }
        except Exception:
            logger.warning("Failed to resolve user '%s' via Graph API", query, exc_info=True)

        return None

    # --- Parsing helpers ---

    def _parse_conversations(self, data: dict) -> list[dict]:
        """Parse raw conversation list into clean dicts."""
        raw_list = data.get("conversations", data) if isinstance(data, dict) else data
        if not isinstance(raw_list, list):
            return []

        result = []
        for c in raw_list:
            conv_id = c.get("id", "")
            if not conv_id:
                continue
            topic = c.get("threadProperties", {}).get("topic", "")
            last_activity_str = (
                c.get("lastMessage", {}).get("composetime")
                or c.get("lastActivity", "")
            )
            last_activity = self._parse_timestamp(last_activity_str)

            # Extract last message sender info (useful for DM name resolution)
            last_msg = c.get("lastMessage", {})
            sender_display = last_msg.get("imdisplayname", "")
            sender_id = last_msg.get("from", "")

            result.append({
                "id": conv_id,
                "topic": topic or "Untitled",
                "lastActivity": last_activity,
                "type": self._infer_conversation_type(conv_id),
                "lastMessageSender": sender_display,
                "lastMessageSenderId": sender_id,
                "favorite": str(c.get("properties", {}).get("favorite", "")).lower() == "true",
            })
        return result

    def _parse_message(self, m: dict) -> dict | None:
        """Parse a raw Teams message into a clean dict."""
        msg_type = m.get("messagetype", m.get("messageType", ""))
        # Skip system messages (topic changes, member additions, etc.)
        if msg_type and msg_type not in (
            "Text", "RichText/Html", "RichText", "RichText/Media_GenericCard",
        ):
            return None

        content = m.get("content", "")
        # Convert HTML to clean text via normalizer_utils
        if "<" in content:
            content = html_to_markdown(content).strip()
        if not content:
            return None

        # Extract sender display name — prefer imdisplayname (human-readable)
        # over from (which is often a raw API URL like https://teams.cloud.microsoft/.../contacts/8:orgid:...)
        from_display = m.get("imdisplayname", "") or ""
        if not from_display:
            sender = m.get("from", "")
            if isinstance(sender, str) and not sender.startswith("http"):
                from_display = sender
            elif isinstance(sender, dict):
                from_display = sender.get("displayName", sender.get("name", ""))

        composetime = m.get("composetime", m.get("originalarrivaltime", ""))
        root_message_id = (
            m.get("rootMessageId")
            or m.get("properties", {}).get("rootMessageId")
            or m.get("properties", {}).get("parentMessageId")
        )

        return {
            "id": m.get("id", m.get("amsreferencesid", "")),
            "conversationid": m.get("conversationid"),
            "from": from_display or "Unknown",
            "content": content,
            "time": self._parse_timestamp(composetime),
            "rootMessageId": root_message_id,
            "parentMessageId": m.get("properties", {}).get("parentMessageId"),
            "mentions": m.get("properties", {}).get("mentions", []),
            "attachments": m.get("amsreferences", []),
        }

    @staticmethod
    def _parse_timestamp(ts: str | None) -> datetime:
        """Parse an ISO timestamp, returning epoch on failure."""
        if not ts:
            return datetime(2000, 1, 1, tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime(2000, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _infer_conversation_type(conv_id: str) -> str:
        """Infer conversation type from ID pattern."""
        if "@thread.tacv2" in conv_id:
            return "channel"
        elif "@unq.gbl.spaces" in conv_id:
            return "individual_chat"
        else:
            return "group_chat"


# ============================================================================
# Error types
# ============================================================================

class AuthenticationError(Exception):
    """Raised when Teams authentication fails."""


# ============================================================================
# Teams Gene
# ============================================================================

class TeamsGene(Gene):
    """Microsoft Teams data source gene.

    Syncs channel messages, group chats, and 1:1 chats via the Teams Chat API.
    Normalizes conversation threads and message blocks into comprehensive
    markdown for memory extraction.
    """

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="teams",
            display_name="Microsoft Teams",
            description="Channel messages, group chats, and direct messages",
            default_sync_interval_minutes=60,
            auth_method="browser_cookie",
            data_shape="message",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Connection", order=0),
                ConfigGroup(key="scope", label="What to Sync", order=1),
                ConfigGroup(key="sync", label="Sync Settings", order=2),
            ],
            fields=[
                ConfigField(
                    key="region", label="Teams Region",
                    field_type=ConfigFieldType.SELECT, required=False,
                    default="emea",
                    options=["emea", "amer", "apac"],
                    help_text="Teams API regional endpoint",
                    group="connection", order=0,
                ),
                ConfigField(
                    key="channels", label="Team Channels",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="TeamName/ChannelName or conversation IDs",
                    help_text="Comma-separated list of channels to sync",
                    group="scope", order=0,
                ),
                ConfigField(
                    key="group_chats", label="Group Chats",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="Chat topic or conversation ID",
                    help_text="Comma-separated group chat names or IDs",
                    group="scope", order=1,
                ),
                ConfigField(
                    key="individual_chats", label="Individual Chats",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="Person name or user ID",
                    help_text="Comma-separated names or IDs for 1:1 chats",
                    group="scope", order=2,
                ),
                ConfigField(
                    key="max_age_days", label="Max Age (days)",
                    field_type=ConfigFieldType.INTEGER, required=False,
                    default="90",
                    help_text="How far back to fetch on initial sync",
                    group="sync", order=0,
                ),
                ConfigField(
                    key="conversation_gap_minutes", label="Conversation Gap (minutes)",
                    field_type=ConfigFieldType.INTEGER, required=False,
                    default="60",
                    help_text="Minutes of silence that starts a new conversation block",
                    group="sync", order=1,
                ),
                ConfigField(
                    key="max_block_messages", label="Max Block Messages",
                    field_type=ConfigFieldType.INTEGER, required=False,
                    default="100",
                    help_text="Maximum messages per conversation block",
                    group="sync", order=2,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._client: _TeamsAPIClient | None = None
        self._gap_minutes = int(config.get("conversation_gap_minutes", 60))
        self._max_block = int(config.get("max_block_messages", 100))
        self._max_age_days = int(config.get("max_age_days", 90))
        self._message_cache: dict[str, list[dict]] = {}  # conv_id → messages (per-sync)

        # Validate scope
        channels = config.get("channels", [])
        group_chats = config.get("group_chats", [])
        individual_chats = config.get("individual_chats", [])
        if not any([channels, group_chats, individual_chats]):
            raise ValueError(
                "At least one of channels, group_chats, or individual_chats must be configured"
            )

    async def authenticate(self) -> None:
        """Authenticate using tokens from Chrome cookies."""
        region = self.config.get("region", "emea")
        self._client = _TeamsAPIClient(region=region)
        await self._client.validate()
        self._log.info("Teams authenticated (region=%s)", region)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Discover threads and conversation blocks from configured sources."""
        self._message_cache.clear()  # fresh cache per sync run
        conversations = await self._client.list_conversations()
        conv_lookup = {c["id"]: c for c in conversations}
        self._log.info("Listed %d conversations, since=%s", len(conversations), since)

        # Resolve configured conversations to IDs + metadata
        configured = await self._resolve_configured_conversations(conv_lookup)
        self._log.info("Resolved %d configured conversations", len(configured))

        for conv_id, conv_meta in configured:
            # Skip if no activity since last sync
            if since and conv_meta["lastActivity"] < since:
                self._log.debug("Skipping %s — no activity since %s", conv_id[:30], since)
                continue

            # Fetch messages with full context
            self._log.info("Fetching messages for %s (topic=%s)", conv_id[:30], conv_meta.get("topic", "?"))
            messages = await self._fetch_with_context(conv_id, since)
            self._log.info("Got %d messages for %s", len(messages), conv_id[:30])
            if not messages:
                continue

            # Cache for fetch() reuse — avoids re-fetching the same conversation
            self._message_cache[conv_id] = messages

            # Partition by threading
            threaded: dict[str, list[dict]] = defaultdict(list)
            unthreaded: list[dict] = []

            for msg in messages:
                root_id = msg.get("rootMessageId") or msg.get("parentMessageId")
                if root_id and root_id != msg["id"]:
                    threaded[root_id].append(msg)
                else:
                    unthreaded.append(msg)

            # Yield threads (channels mostly — group/1:1 rarely have these)
            for root_id, replies in threaded.items():
                # Find root message in unthreaded
                root_msgs = [m for m in unthreaded if m["id"] == root_id]
                thread_msgs = (root_msgs + replies) if root_msgs else replies
                thread_msgs.sort(key=lambda m: m["time"])

                if since and max(m["time"] for m in thread_msgs) < since:
                    continue

                yield self._make_content_item(
                    conv_meta, root_id, thread_msgs, is_thread=True,
                )

            # Yield conversation blocks from unthreaded messages
            # (exclude messages that are thread roots — already yielded above)
            thread_root_ids = set(threaded.keys())
            block_messages = [m for m in unthreaded if m["id"] not in thread_root_ids]
            block_messages.sort(key=lambda m: m["time"])

            for block in _group_into_blocks(block_messages, self._gap_minutes, self._max_block):
                if since and max(m["time"] for m in block) < since:
                    continue
                yield self._make_content_item(
                    conv_meta, block[0]["id"], block, is_thread=False,
                )

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full thread/block content."""
        conv_id = item.extra["conversation_id"]
        root_msg_id = item.extra["root_message_id"]
        conv_type = item.extra["conversation_type"]

        if item.extra.get("is_thread"):
            messages = await self._client.get_thread_messages(conv_id, root_msg_id)
        else:
            # Use cached messages from discover() if available, else re-fetch
            cached = self._message_cache.get(conv_id)
            if cached:
                messages = cached
            else:
                messages = await self._fetch_with_context(conv_id, since=None)

            # Filter to this block's time range
            block_start = item.extra.get("block_start")
            block_end = item.extra.get("block_end")
            if block_start and block_end:
                start = datetime.fromisoformat(block_start)
                end = datetime.fromisoformat(block_end)
                messages = [m for m in messages if start <= m["time"] <= end]

        if not messages:
            messages = [{"id": root_msg_id, "from": "Unknown", "content": "", "time": item.last_modified}]

        participants = sorted({m["from"] for m in messages if m.get("from")})
        first_time = min(m["time"] for m in messages)
        last_time = max(m["time"] for m in messages)

        thread_data = {
            "conversation_type": conv_type,
            "title": item.title,
            "channel_name": item.extra.get("channel_name", ""),
            "team_name": item.space_or_project,
            "messages": [
                {
                    "id": m["id"],
                    "from": m["from"],
                    "content": m["content"],
                    "time": m["time"].isoformat() if isinstance(m["time"], datetime) else m["time"],
                    "mentions": m.get("mentions", []),
                    "attachments": m.get("attachments", []),
                    "is_root": m["id"] == root_msg_id,
                }
                for m in sorted(messages, key=lambda x: x["time"])
            ],
            "participants": participants,
            "first_message_time": first_time.isoformat(),
            "last_message_time": last_time.isoformat(),
        }

        return RawContent(
            item=item,
            body=json.dumps(thread_data, default=str).encode("utf-8"),
            content_type="application/json",
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert Teams thread/block JSON to comprehensive markdown."""
        data = json.loads(raw.body.decode("utf-8"))
        messages = data.get("messages", [])
        participants = data.get("participants", [])
        conv_type = data.get("conversation_type", "")
        channel_name = data.get("channel_name", "")
        team_name = data.get("team_name", "")
        first_time = data.get("first_message_time", "")
        last_time = data.get("last_message_time", "")

        # Build header
        header_lines = [f"# {raw.item.title}", ""]
        if channel_name:
            header_lines.append(f"**Channel**: {channel_name}" + (f" ({team_name})" if team_name else ""))
        elif conv_type == "group_chat":
            header_lines.append(f"**Group Chat**: {team_name or 'Unnamed'}")
        elif conv_type == "individual_chat":
            header_lines.append("**Direct Message**")
        if participants:
            header_lines.append(f"**Participants**: {', '.join(participants)}")
        if first_time and last_time:
            header_lines.append(f"**Date range**: {first_time[:16]} – {last_time[:16]}")
        header_lines.append(f"**Messages**: {len(messages)}")
        header_lines.extend(["", "---", ""])

        # Build message body
        body_lines: list[str] = []
        has_code_blocks = False
        has_links = False

        for i, msg in enumerate(messages):
            content = msg.get("content", "").strip()
            if not content:
                continue

            if "```" in content:
                has_code_blocks = True
            if "http://" in content or "https://" in content:
                has_links = True

            author = msg.get("from", "Unknown")
            time_str = msg.get("time", "")[:16]  # trim to minute precision
            is_root = msg.get("is_root", i == 0)

            if is_root:
                body_lines.append(f"**{author}** ({time_str}):")
                body_lines.append(content)
                body_lines.append("")
            else:
                # Replies / subsequent messages as blockquotes
                body_lines.append(f"> **{author}** ({time_str}):")
                for line in content.split("\n"):
                    body_lines.append(f"> {line}")
                body_lines.append("")

        # Attachments
        all_attachments = []
        for msg in messages:
            for att in msg.get("attachments", []):
                name = att.get("fileName", att.get("name", "attachment"))
                author = msg.get("from", "")
                all_attachments.append(f"- {name} (shared by {author})")

        if all_attachments:
            body_lines.extend(["---", "", "**Attachments**:"])
            body_lines.extend(all_attachments)

        markdown = "\n".join(header_lines) + "\n\n" + "\n".join(body_lines)

        return NormalizedContent(
            item=raw.item,
            markdown_body=markdown,
            source_semantics={
                "conversation_type": conv_type,
                "channel_name": channel_name,
                "team_name": team_name,
                "participants": participants,
                "message_count": len(messages),
                "date_range": {"start": first_time, "end": last_time},
                "has_code_blocks": has_code_blocks,
                "has_links": has_links,
            },
        )

    async def health_check(self) -> dict:
        """Check Teams API connectivity."""
        try:
            await self._client.validate()
            return {"healthy": True}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _resolve_configured_conversations(
        self, conv_lookup: dict[str, dict],
    ) -> list[tuple[str, dict]]:
        """Resolve configured channels/chats to (conversation_id, metadata) pairs."""
        result: list[tuple[str, dict]] = []

        # Channels
        channels = self.config.get("channels", [])
        if isinstance(channels, str):
            channels = [c.strip() for c in channels.split(",") if c.strip()]

        for channel_spec in channels:
            conv_id = await self._resolve_channel(channel_spec, conv_lookup)
            if conv_id and conv_id in conv_lookup:
                meta = {**conv_lookup[conv_id]}  # copy to avoid mutating shared dict
                meta["channel_name"] = channel_spec.split("/")[-1] if "/" in channel_spec else channel_spec
                meta["team_name"] = channel_spec.split("/")[0] if "/" in channel_spec else ""
                result.append((conv_id, meta))
            elif conv_id:
                result.append((conv_id, {
                    "id": conv_id, "topic": channel_spec, "type": "channel",
                    "lastActivity": datetime.now(timezone.utc),
                    "channel_name": channel_spec, "team_name": "",
                }))

        # Group chats
        group_chats = self.config.get("group_chats", [])
        if isinstance(group_chats, str):
            group_chats = [g.strip() for g in group_chats.split(",") if g.strip()]

        for chat_spec in group_chats:
            # Direct ID
            if ":" in chat_spec or "@" in chat_spec:
                if chat_spec in conv_lookup:
                    result.append((chat_spec, conv_lookup[chat_spec]))
                else:
                    # ID not in active conversation list — use it directly
                    # (the user selected it in the picker, so it's valid)
                    self._log.info("Group chat %s not in conversation list, using ID directly", chat_spec)
                    result.append((chat_spec, {
                        "id": chat_spec, "topic": "Group Chat", "type": "group_chat",
                        "lastActivity": datetime.now(timezone.utc),
                    }))
                continue
            # Match by topic
            for conv_id, meta in conv_lookup.items():
                if meta.get("topic", "").lower() == chat_spec.lower():
                    result.append((conv_id, meta))
                    break
            else:
                self._log.warning("Group chat not found: %s", chat_spec)

        # Individual chats
        individual_chats = self.config.get("individual_chats", [])
        if isinstance(individual_chats, str):
            individual_chats = [i.strip() for i in individual_chats.split(",") if i.strip()]

        for person_spec in individual_chats:
            user = await self._client.resolve_user(person_spec)
            if not user:
                self._log.warning("Could not resolve user: %s", person_spec)
                continue

            user_id = user["id"]
            for conv_id, meta in conv_lookup.items():
                if meta.get("type") == "individual_chat" and user_id in conv_id:
                    meta["person_name"] = user.get("displayName", person_spec)
                    result.append((conv_id, meta))
                    break
            else:
                self._log.warning("No 1:1 conversation found with user: %s", person_spec)

        return result

    async def _resolve_channel(
        self, channel_spec: str, conv_lookup: dict[str, dict],
    ) -> str | None:
        """Resolve a channel spec (TeamName/ChannelName or direct ID) to conversation ID."""
        # Direct conversation ID
        if "@thread.tacv2" in channel_spec:
            return channel_spec

        if "/" not in channel_spec:
            self._log.warning("Channel spec must be TeamName/ChannelName or a conversation ID: %s", channel_spec)
            return None

        team_name, channel_name = channel_spec.split("/", 1)

        # Resolve via Graph API
        teams = await self._client.get_joined_teams()
        team = next(
            (t for t in teams if t.get("displayName", "").lower() == team_name.strip().lower()),
            None,
        )
        if not team:
            self._log.warning("Team not found: %s", team_name)
            return None

        channels = await self._client.get_team_channels(team["id"])
        channel = next(
            (c for c in channels if c.get("displayName", "").lower() == channel_name.strip().lower()),
            None,
        )
        if not channel:
            self._log.warning("Channel not found: %s in team %s", channel_name, team_name)
            return None

        # The channel ID from Graph is the conversation ID for Chat API
        return channel.get("id")

    async def _fetch_with_context(
        self, conv_id: str, since: datetime | None,
    ) -> list[dict]:
        """Fetch messages with full conversation block context.

        Initial sync (since=None): all messages within max_age_days.
        Incremental sync: backward from newest until a gap > gap_minutes,
        ensuring the full active conversation block is captured.
        """
        if since is None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
            return await self._client.get_messages_until(conv_id, cutoff)

        # Incremental: paginate backward, stop at gap boundary
        gap_seconds = self._gap_minutes * 60
        all_messages: list[dict] = []
        found_boundary = False

        async for page in self._client.paginate_messages_backward(conv_id):
            all_messages.extend(page)
            all_messages.sort(key=lambda m: m["time"])

            # Scan for a gap > gap_minutes where the earlier side is before `since`
            for i in range(len(all_messages) - 1):
                gap = (all_messages[i + 1]["time"] - all_messages[i]["time"]).total_seconds()
                if gap > gap_seconds and all_messages[i]["time"] < since:
                    all_messages = all_messages[i + 1:]
                    found_boundary = True
                    break

            if found_boundary:
                break

            # Stop if we've gone past since + one full gap window
            if all_messages and all_messages[0]["time"] < since - timedelta(minutes=self._gap_minutes):
                break

        return all_messages

    def _make_content_item(
        self,
        conv_meta: dict,
        first_msg_id: str,
        messages: list[dict],
        is_thread: bool,
    ) -> ContentItem:
        """Build a ContentItem from a thread or conversation block."""
        conv_id = conv_meta["id"]
        conv_type = conv_meta.get("type", "channel")
        channel_name = conv_meta.get("channel_name", conv_meta.get("topic", ""))
        team_name = conv_meta.get("team_name", "")
        person_name = conv_meta.get("person_name", "")

        first_msg = messages[0] if messages else {}

        max_time = max((m["time"] for m in messages), default=datetime.now(timezone.utc))
        min_time = min((m["time"] for m in messages), default=datetime.now(timezone.utc))

        # Title by type — use date range for blocks, first message for threads
        if conv_type == "channel":
            if is_thread:
                preview = first_msg.get("content", "")[:80]
                title = f"#{channel_name}: {preview}"
            else:
                title = f"#{channel_name}: {_format_date_range(min_time, max_time)}"
        elif conv_type == "group_chat":
            topic = conv_meta.get("topic", "Group")
            title = f"Group: {topic} -- {_format_date_range(min_time, max_time)}"
        else:
            title = f"DM with {person_name or 'Unknown'} -- {_format_date_range(min_time, max_time)}"

        # Build Teams deep link
        source_url = f"https://teams.microsoft.com/l/message/{conv_id}/{first_msg_id}"

        return ContentItem(
            item_id=f"teams-{conv_id}#{first_msg_id}",
            title=title[:200],
            source_url=source_url,
            last_modified=max_time,
            content_type="application/json",
            space_or_project=team_name or conv_meta.get("topic", ""),
            author=first_msg.get("from"),
            labels=[conv_type, channel_name or person_name or conv_meta.get("topic", "")],
            extra={
                "conversation_id": conv_id,
                "root_message_id": first_msg_id,
                "conversation_type": conv_type,
                "channel_name": channel_name,
                "message_count": len(messages),
                "is_thread": is_thread,
                "block_start": min_time.isoformat() if not is_thread else None,
                "block_end": max_time.isoformat() if not is_thread else None,
            },
        )


# ============================================================================
# Helpers
# ============================================================================

def _format_date_range(min_time: datetime, max_time: datetime) -> str:
    """Format a compact date range string for document titles."""
    if min_time.date() == max_time.date():
        return f"{min_time.strftime('%b %d')}, {min_time.strftime('%H:%M')}-{max_time.strftime('%H:%M')}"
    return f"{min_time.strftime('%b %d %H:%M')} - {max_time.strftime('%b %d %H:%M')}"


# ============================================================================
# Block grouping (module-level for testability)
# ============================================================================

def _group_into_blocks(
    messages: list[dict],
    gap_minutes: int,
    max_messages: int,
) -> list[list[dict]]:
    """Split unthreaded messages into conversation blocks by time gaps.

    A gap of >gap_minutes between consecutive messages starts a new block.
    Blocks exceeding max_messages are split at the largest internal gap.
    """
    if not messages:
        return []

    sorted_msgs = sorted(messages, key=lambda m: m["time"])
    blocks: list[list[dict]] = []
    current_block: list[dict] = [sorted_msgs[0]]
    gap_seconds = gap_minutes * 60

    for i in range(1, len(sorted_msgs)):
        gap = (sorted_msgs[i]["time"] - sorted_msgs[i - 1]["time"]).total_seconds()
        if gap > gap_seconds:
            blocks.append(current_block)
            current_block = [sorted_msgs[i]]
        else:
            current_block.append(sorted_msgs[i])

    if current_block:
        blocks.append(current_block)

    # Split oversized blocks at largest internal gap
    final_blocks: list[list[dict]] = []
    for block in blocks:
        if len(block) <= max_messages:
            final_blocks.append(block)
            continue

        # Find largest gap and split there
        while len(block) > max_messages:
            max_gap = 0.0
            split_idx = len(block) // 2  # fallback: split in half

            for i in range(1, len(block)):
                gap = (block[i]["time"] - block[i - 1]["time"]).total_seconds()
                if gap > max_gap:
                    max_gap = gap
                    split_idx = i

            final_blocks.append(block[:split_idx])
            block = block[split_idx:]

        if block:
            final_blocks.append(block)

    return final_blocks
