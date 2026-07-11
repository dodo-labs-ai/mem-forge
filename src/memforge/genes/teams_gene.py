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
from pathlib import Path

import httpx

from memforge.genes.base import Gene
from memforge.genes.local_adapter_packages import read_package_body
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
LOCAL_AGENT_TEAMS_PACKAGE_KIND = "teams_window_document"


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
        self._poll_audits: dict[str, dict] = {}

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
                "No Teams session found. Connect Teams from the source wizard."
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
                "Teams session expired. Connect Teams from the source wizard."
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
            await self._request(
                self._chat_client,
                "GET",
                "/conversations",
                params={"pageSize": 1},
            )
        except AuthenticationError:
            raise
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
            if resp.status_code == 401:
                raise AuthenticationError(
                    "Teams session expired. Connect Teams from the source wizard."
                )
            if resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    "Rate limited", request=resp.request, response=resp,
                )
            resp.raise_for_status()
            return resp

        return await retry_async(
            _do_request,
            max_retries=3,
            retryable_exceptions=(
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ),
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
        requested_urls: set[str] = set()
        seen_message_keys: set[tuple[str, str, str]] = set()

        while url:
            if url in requested_urls:
                self.mark_poll_complete(conversation_id, stop_reason="repeated_backward_link")
                break
            requested_urls.add(url)
            resp = await self._request(self._chat_client, "GET", url, params=params)
            data = resp.json()
            messages = data.get("messages", data) if isinstance(data, dict) else data

            parsed_page: list[dict] = []
            if isinstance(messages, list):
                for m in messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        parsed_page.append(parsed)
                        message_key = _parsed_message_key(conversation_id, parsed)
                        if message_key in seen_message_keys:
                            continue
                        seen_message_keys.add(message_key)
                        all_messages.append(parsed)
            self._record_message_poll_page(conversation_id, data, parsed_page)

            # Pagination: check for next link
            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link
            if not url:
                self.mark_poll_complete(conversation_id, stop_reason="no_backward_link")

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
        requested_urls: set[str] = set()
        seen_message_keys: set[tuple[str, str, str]] = set()

        while url:
            if url in requested_urls:
                self.mark_poll_complete(conversation_id, stop_reason="repeated_backward_link")
                break
            requested_urls.add(url)
            # Rate-limit: Teams allows 15 req/10s — pace at ~1 req/s
            if page_count > 0:
                await asyncio.sleep(1.0)
            resp = await self._request(self._chat_client, "GET", url, params=params)
            page_count += 1
            data = resp.json()
            messages = data.get("messages", data) if isinstance(data, dict) else data

            hit_cutoff = False
            parsed_page: list[dict] = []
            if isinstance(messages, list):
                for m in messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        if parsed["time"] < cutoff:
                            hit_cutoff = True
                            continue
                        parsed_page.append(parsed)
                        message_key = _parsed_message_key(conversation_id, parsed)
                        if message_key in seen_message_keys:
                            continue
                        seen_message_keys.add(message_key)
                        all_messages.append(parsed)
            self._record_message_poll_page(conversation_id, data, parsed_page)

            if hit_cutoff:
                self.mark_poll_complete(conversation_id, stop_reason="cutoff_reached")
                break

            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link
            if not url:
                self.mark_poll_complete(conversation_id, stop_reason="no_backward_link")

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
        requested_urls: set[str] = set()
        seen_message_keys: set[tuple[str, str, str]] = set()

        while url:
            if url in requested_urls:
                self.mark_poll_complete(conversation_id, stop_reason="repeated_backward_link")
                break
            requested_urls.add(url)
            if page_count > 0:
                await asyncio.sleep(1.0)
            resp = await self._request(self._chat_client, "GET", url, params=params)
            page_count += 1
            data = resp.json()
            raw_messages = data.get("messages", data) if isinstance(data, dict) else data

            page: list[dict] = []
            parsed_page: list[dict] = []
            if isinstance(raw_messages, list):
                for m in raw_messages:
                    parsed = self._parse_message(m)
                    if parsed:
                        parsed_page.append(parsed)
                        message_key = _parsed_message_key(conversation_id, parsed)
                        if message_key in seen_message_keys:
                            continue
                        seen_message_keys.add(message_key)
                        page.append(parsed)
            self._record_message_poll_page(conversation_id, data, parsed_page)

            if page:
                yield page

            # Next page via backward link
            url = None
            params = {}
            if isinstance(data, dict):
                next_link = data.get("_metadata", {}).get("backwardLink")
                if next_link:
                    url = next_link
            if not url:
                self.mark_poll_complete(conversation_id, stop_reason="no_backward_link")

    def _record_message_poll_page(
        self,
        conversation_id: str,
        data: dict,
        parsed_messages: list[dict],
    ) -> None:
        raw_messages = data.get("messages", data) if isinstance(data, dict) else data
        if not isinstance(raw_messages, list):
            raw_messages = []
        audit = self._poll_audits.setdefault(conversation_id, _new_poll_audit(conversation_id))
        audit["page_count"] += 1
        audit["raw_messages_seen"] += len(raw_messages)
        audit["parse_filtered_messages"] += max(0, len(raw_messages) - len(parsed_messages))

        metadata = data.get("_metadata", {}) if isinstance(data, dict) else {}
        if isinstance(metadata, dict):
            if metadata.get("syncState"):
                audit["metadata_sync_state"] = metadata["syncState"]
            if metadata.get("backwardLink"):
                audit["metadata_backward_link"] = metadata["backwardLink"]

        seen_keys = audit["_seen_message_keys"]
        for message in raw_messages:
            key = _raw_message_key(conversation_id, message)
            if key in seen_keys:
                audit["duplicate_raw_messages"] += 1
            else:
                seen_keys.add(key)
            timestamp = _raw_message_timestamp(message)
            if timestamp:
                audit["covered_created_from"] = _min_iso(audit.get("covered_created_from"), timestamp)
                audit["covered_created_to"] = _max_iso(audit.get("covered_created_to"), timestamp)

        selected_keys = audit["_selected_message_keys"]
        for message in parsed_messages:
            selected_keys.add(_parsed_message_key(conversation_id, message))

    def record_poll_ledger_actions(self, conversation_id: str, counts: dict[str, int]) -> None:
        audit = self._poll_audits.setdefault(conversation_id, _new_poll_audit(conversation_id))
        audit["upsert_new"] += _int_or_zero(counts.get("new"))
        audit["upsert_updated"] += _int_or_zero(counts.get("updated"))
        audit["upsert_unchanged"] += _int_or_zero(counts.get("unchanged"))
        audit["explicit_delete_markers"] += _int_or_zero(counts.get("deleted"))
        audit["missing_once_candidates"] += _int_or_zero(counts.get("missing_once"))
        audit["ledger_action_basis"] = "message_receipt"

    def mark_poll_complete(self, conversation_id: str, *, stop_reason: str) -> None:
        audit = self._poll_audits.setdefault(conversation_id, _new_poll_audit(conversation_id))
        audit["pagination_complete"] = True
        audit["stop_reason"] = stop_reason

    def get_poll_audits(self) -> list[dict]:
        result: list[dict] = []
        for conversation_id, audit in sorted(self._poll_audits.items()):
            unique_count = len(audit.get("_seen_message_keys", set()))
            selected_count = len(audit.get("_selected_message_keys", set()))
            clean = {
                key: value
                for key, value in audit.items()
                if key not in {"_seen_message_keys", "_selected_message_keys"}
            }
            clean["raw_conversation_id"] = conversation_id
            clean["unique_message_keys_seen"] = unique_count
            clean["selected_message_keys_seen"] = selected_count
            if not clean.get("ledger_action_basis"):
                clean["upsert_new"] = selected_count
                clean["ledger_action_basis"] = "selected_without_message_receipt"
            result.append(clean)
        return result

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
                    key="conversation_ids", label="Teams Conversations",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="Conversation IDs selected through Browse Teams",
                    help_text="Direct Teams conversation IDs selected through Browse Teams",
                    group="scope", order=0,
                ),
                ConfigField(
                    key="max_age_days", label="Max Age (days)",
                    field_type=ConfigFieldType.INTEGER, required=False,
                    default="14",
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
        self._max_age_days = int(config.get("max_age_days", 14))
        self._conversation_fetch_timeout_seconds = int(
            config.get("conversation_fetch_timeout_seconds", 300)
        )
        self._message_cache: dict[str, list[dict]] = {}  # conv_id → messages (per-sync)

        # Local-agent Teams sources read already-captured raw window packages
        # from the server-side inbox. They do not need remote Teams selectors.
        if self._local_agent_documents_dir() is None and not self._local_agent_package_manifest():
            conversation_ids = config.get("conversation_ids", [])
            channels = config.get("channels", [])
            group_chats = config.get("group_chats", [])
            individual_chats = config.get("individual_chats", [])
            if not any([conversation_ids, channels, group_chats, individual_chats]):
                raise ValueError(
                    "At least one Teams conversation ID must be configured"
                )

    async def authenticate(self) -> None:
        """Authenticate using tokens from Chrome cookies."""
        local_documents_dir = self._local_agent_documents_dir()
        if self._local_agent_package_manifest():
            return
        if local_documents_dir is not None:
            local_documents_dir.mkdir(parents=True, exist_ok=True)
            return
        region = self.config.get("region", "emea")
        self._client = _TeamsAPIClient(region=region)
        await self._client.validate()
        self._log.info("Teams authenticated (region=%s)", region)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Discover threads and conversation blocks from configured sources."""
        local_documents_dir = self._local_agent_documents_dir()
        manifest = self._local_agent_package_manifest()
        if manifest:
            async for item in self._discover_local_agent_package_manifest(manifest, since):
                yield item
            return
        if local_documents_dir is not None:
            async for item in self._discover_local_agent_packages(local_documents_dir, since):
                yield item
            return

        self._message_cache.clear()  # fresh cache per sync run
        conversations = await self._client.list_conversations()
        conv_lookup = {c["id"]: c for c in conversations}
        self._log.info("Listed %d conversations, since=%s", len(conversations), since)

        # Resolve configured conversations to IDs + metadata
        configured = await self._resolve_configured_conversations(conv_lookup)
        self._log.info("Resolved %d configured conversations", len(configured))

        successful_polls = 0
        conversation_failures: list[str] = []
        for conv_id, conv_meta in configured:
            # Skip if no activity since last sync
            if since and conv_meta["lastActivity"] < since:
                self._log.debug("Skipping %s — no activity since %s", conv_id[:30], since)
                continue

            # Fetch messages with full context
            self._log.info("Fetching messages for %s (topic=%s)", conv_id[:30], conv_meta.get("topic", "?"))
            try:
                messages = await self._fetch_with_timeout(conv_id, since)
            except TimeoutError as exc:
                self._log.warning("Skipping Teams conversation %s after timeout: %s", conv_id[:30], exc)
                conversation_failures.append(str(exc))
                continue
            self._log.info("Got %d messages for %s", len(messages), conv_id[:30])
            successful_polls += 1
            self._record_message_ledger_actions(conv_id, conv_meta, messages)
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

            for item in self._project_unthreaded_content_items(conv_meta, block_messages, since=since):
                yield item

        if conversation_failures and successful_polls == 0:
            first_failure = conversation_failures[0]
            remaining = len(conversation_failures) - 1
            suffix = f" (+{remaining} more)" if remaining else ""
            raise RuntimeError(f"Teams sync could not fetch any configured conversations: {first_failure}{suffix}")

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full thread/block content."""
        if item.extra.get("package_uri") or item.extra.get("package_path"):
            return RawContent(
                item=item,
                body=read_package_body(self, item, source_label="Teams"),
                content_type="application/json",
            )

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
                messages = await self._fetch_with_timeout(conv_id, since=None)

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
        package = json.loads(raw.body.decode("utf-8"))
        data = package
        package_semantics = {}
        if package.get("package_kind") == LOCAL_AGENT_TEAMS_PACKAGE_KIND:
            raw_payload = package.get("raw_payload")
            if not isinstance(raw_payload, dict):
                raise ValueError("Teams local-agent package is missing raw_payload")
            data = raw_payload
            package_semantics = {
                "source_kind": "teams",
                "conversation_id": package.get("conversation_id"),
                "root_message_id": package.get("root_message_id"),
                "window_id": package.get("window_id"),
                "window_type": package.get("window_type"),
                "revision_hash": package.get("revision_hash"),
                "raw_hash": package.get("raw_hash"),
                "submitted_at": package.get("submitted_at"),
                "submitted_by": package.get("submitted_by"),
            }

        raw_messages = data.get("messages", [])
        messages = [msg for msg in raw_messages if isinstance(msg, dict)] if isinstance(raw_messages, list) else []
        participants = _teams_string_values(data.get("participants", []))
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
            content = str(msg.get("content") or "").strip()
            if not content:
                continue

            if "```" in content:
                has_code_blocks = True
            if "http://" in content or "https://" in content:
                has_links = True

            author = str(msg.get("from") or "Unknown")
            time_str = str(msg.get("time") or "")[:16]  # trim to minute precision
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
            for name in _teams_attachment_names(msg.get("attachments", [])):
                author = str(msg.get("from") or "")
                all_attachments.append(f"- {name} (shared by {author})")

        if all_attachments:
            body_lines.extend(["---", "", "**Attachments**:"])
            body_lines.extend(all_attachments)

        markdown = "\n".join(header_lines) + "\n\n" + "\n".join(body_lines)

        return NormalizedContent(
            item=raw.item,
            markdown_body=markdown,
            source_semantics={
                **package_semantics,
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

        conversation_ids = self.config.get("conversation_ids", [])
        if isinstance(conversation_ids, str):
            conversation_ids = [item.strip() for item in conversation_ids.split(",") if item.strip()]
        for conversation_id in conversation_ids:
            if conversation_id in conv_lookup:
                result.append((conversation_id, conv_lookup[conversation_id]))
            else:
                result.append((conversation_id, {
                    "id": conversation_id,
                    "topic": "Teams conversation",
                    "type": _TeamsAPIClient._infer_conversation_type(conversation_id),
                    "lastActivity": datetime.now(timezone.utc),
                }))

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

        deduped: list[tuple[str, dict]] = []
        seen_ids: set[str] = set()
        for conversation_id, metadata in result:
            if conversation_id in seen_ids:
                continue
            seen_ids.add(conversation_id)
            deduped.append((conversation_id, metadata))
        return deduped

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
                    self._client.mark_poll_complete(conv_id, stop_reason="gap_boundary")
                    break

            if found_boundary:
                break

            # Stop if we've gone past since + one full gap window
            if all_messages and all_messages[0]["time"] < since - timedelta(minutes=self._gap_minutes):
                self._client.mark_poll_complete(conv_id, stop_reason="incremental_context_boundary")
                break

        return all_messages

    async def _fetch_with_timeout(
        self, conv_id: str, since: datetime | None,
    ) -> list[dict]:
        timeout = self._conversation_fetch_timeout_seconds
        if timeout <= 0:
            return await self._fetch_with_context(conv_id, since)
        try:
            return await asyncio.wait_for(
                self._fetch_with_context(conv_id, since),
                timeout=timeout,
            )
        except TimeoutError as exc:
            if self._client is not None:
                self._client.mark_poll_complete(conv_id, stop_reason="fetch_timeout")
            raise TimeoutError(
                f"Teams message fetch timed out after {timeout}s for conversation {conv_id[:30]}"
            ) from exc

    def _local_agent_documents_dir(self) -> Path | None:
        configured = str(self.config.get("local_agent_documents_dir") or "").strip()
        return Path(configured).expanduser() if configured else None

    def _local_agent_package_manifest(self) -> list[dict]:
        manifest = self.config.get("local_agent_package_manifest")
        if not isinstance(manifest, list):
            return []
        return [entry for entry in manifest if isinstance(entry, dict)]

    def _teams_ledger_state_path(self) -> Path | None:
        configured = str(self.config.get("ledger_state_path") or "").strip()
        return Path(configured).expanduser() if configured else None

    def get_poll_audits(self) -> list[dict]:
        if self._client is None or not hasattr(self._client, "get_poll_audits"):
            return []
        return self._client.get_poll_audits()

    def _record_message_ledger_actions(
        self,
        conv_id: str,
        conv_meta: dict,
        messages: list[dict],
    ) -> None:
        if self._client is None or not hasattr(self._client, "record_poll_ledger_actions"):
            return
        ledger_state_path = self._teams_ledger_state_path()
        if ledger_state_path is None:
            self._client.record_poll_ledger_actions(conv_id, {"new": len(messages)})
            return

        from memforge.local_agent.teams_ledger import (
            TeamsLedgerMessage,
            TeamsLedgerStateStore,
        )

        ledger_messages = [
            TeamsLedgerMessage(
                source_id=self.source_id,
                conversation_id=conv_id,
                conversation_type=str(conv_meta.get("type") or "group_chat"),
                message_id=str(message["id"]),
                created_at=message["time"],
                body_normalized=str(message.get("content") or ""),
                root_message_id=str(message.get("rootMessageId") or message["id"]),
                parent_message_id=message.get("parentMessageId"),
            )
            for message in messages
        ]
        counts = TeamsLedgerStateStore(ledger_state_path).observe_messages(
            source_id=self.source_id,
            conversation_id=conv_id,
            messages=ledger_messages,
        )
        self._client.record_poll_ledger_actions(conv_id, counts)

    def _project_unthreaded_content_items(
        self,
        conv_meta: dict,
        block_messages: list[dict],
        *,
        since: datetime | None,
    ) -> list[ContentItem]:
        ledger_state_path = self._teams_ledger_state_path()
        if ledger_state_path is None:
            items: list[ContentItem] = []
            for block in _group_into_blocks(block_messages, self._gap_minutes, self._max_block):
                if since and max(m["time"] for m in block) < since:
                    continue
                items.append(self._make_content_item(
                    conv_meta, block[0]["id"], block, is_thread=False,
                ))
            return items

        from memforge.local_agent.teams_ledger import (
            TeamsLedgerMessage,
            TeamsLedgerProjector,
            TeamsLedgerStateStore,
        )

        conv_id = str(conv_meta["id"])
        messages_by_id = {str(message["id"]): message for message in block_messages}
        ledger_messages = [
            TeamsLedgerMessage(
                source_id=self.source_id,
                conversation_id=conv_id,
                conversation_type=str(conv_meta.get("type") or "group_chat"),
                message_id=str(message["id"]),
                created_at=message["time"],
                body_normalized=str(message.get("content") or ""),
                root_message_id=str(message.get("rootMessageId") or message["id"]),
                parent_message_id=message.get("parentMessageId"),
            )
            for message in block_messages
        ]
        store = TeamsLedgerStateStore(ledger_state_path)
        previous = store.load_projection(source_id=self.source_id, conversation_id=conv_id)
        projection = TeamsLedgerProjector(gap_minutes=self._gap_minutes).project_unthreaded(
            ledger_messages,
            previous=previous,
        )
        store.save_projection(source_id=self.source_id, conversation_id=conv_id, projection=projection)

        items = []
        for block in projection.blocks:
            messages = [
                messages_by_id[message_id]
                for message_id in block.member_message_ids
                if message_id in messages_by_id
            ]
            if not messages:
                continue
            if since and block.member_max_created_at < since:
                continue
            items.append(self._make_content_item(
                conv_meta,
                block.frozen_anchor_message_id,
                sorted(messages, key=lambda m: m["time"]),
                is_thread=False,
                window_id=block.window_id,
                block_start=block.member_min_created_at,
                block_end=block.member_max_created_at,
                revision_hash=block.revision_hash,
                block_membership_fingerprint=block.block_membership_fingerprint,
                assignment_generation=block.assignment_generation,
                rebuild_generation=block.rebuild_generation,
                bridge_not_merged=block.bridge_not_merged,
            ))
        return items

    async def _discover_local_agent_packages(
        self,
        documents_dir: Path,
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        for package_path in sorted(documents_dir.rglob("*.json")):
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable Teams local-agent package: %s", package_path)
                continue
            if package.get("package_kind") != LOCAL_AGENT_TEAMS_PACKAGE_KIND:
                continue
            last_modified = _TeamsAPIClient._parse_timestamp(str(package.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            window_id = str(package.get("window_id") or package.get("doc_id") or "")
            yield ContentItem(
                item_id=str(package.get("doc_id") or window_id),
                title=str(package.get("title") or window_id),
                source_url=str(package.get("source_url") or ""),
                last_modified=last_modified,
                content_type="application/json",
                space_or_project=str(package.get("space_or_project") or ""),
                version=str(package.get("revision_hash") or package.get("version") or ""),
                author=package.get("submitted_by"),
                labels=["teams", str(package.get("window_type") or "")],
                extra={
                    "package_path": str(package_path),
                    "conversation_id": package.get("conversation_id"),
                    "root_message_id": package.get("root_message_id"),
                    "window_id": window_id,
                    "window_type": package.get("window_type"),
                    "revision_hash": package.get("revision_hash"),
                },
            )

    async def _discover_local_agent_package_manifest(
        self,
        manifest: list[dict],
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        for entry in sorted(
            manifest,
            key=lambda item: (str(item.get("last_modified") or ""), str(item.get("doc_id") or "")),
        ):
            package_uri = str(entry.get("package_uri") or "").strip()
            if not package_uri:
                continue
            last_modified = _TeamsAPIClient._parse_timestamp(str(entry.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            window_id = str(entry.get("window_id") or entry.get("doc_id") or "")
            yield ContentItem(
                item_id=str(entry.get("doc_id") or window_id),
                title=str(entry.get("title") or window_id),
                source_url=str(entry.get("source_url") or ""),
                last_modified=last_modified,
                content_type="application/json",
                space_or_project=str(entry.get("space_or_project") or ""),
                version=str(entry.get("revision_hash") or entry.get("version") or ""),
                author=entry.get("submitted_by"),
                labels=["teams", str(entry.get("window_type") or "")],
                extra={
                    "package_uri": package_uri,
                    "package_path": entry.get("package_path"),
                    "conversation_id": entry.get("conversation_id"),
                    "root_message_id": entry.get("root_message_id"),
                    "window_id": window_id,
                    "window_type": entry.get("window_type"),
                    "revision_hash": entry.get("revision_hash"),
                },
            )

    def _make_content_item(
        self,
        conv_meta: dict,
        first_msg_id: str,
        messages: list[dict],
        is_thread: bool,
        *,
        window_id: str | None = None,
        block_start: datetime | None = None,
        block_end: datetime | None = None,
        revision_hash: str | None = None,
        block_membership_fingerprint: str | None = None,
        assignment_generation: int | None = None,
        rebuild_generation: int | None = None,
        bridge_not_merged: bool | None = None,
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

        item_id = window_id or f"teams-{conv_id}#{first_msg_id}"
        return ContentItem(
            item_id=item_id,
            title=title[:200],
            source_url=source_url,
            last_modified=max_time,
            content_type="application/json",
            space_or_project=team_name or conv_meta.get("topic", ""),
            version=revision_hash or "",
            author=first_msg.get("from"),
            labels=[conv_type, channel_name or person_name or conv_meta.get("topic", "")],
            extra={
                "conversation_id": conv_id,
                "root_message_id": first_msg_id,
                "conversation_type": conv_type,
                "channel_name": channel_name,
                "message_count": len(messages),
                "is_thread": is_thread,
                "window_id": window_id,
                "block_start": (block_start or min_time).isoformat() if not is_thread else None,
                "block_end": (block_end or max_time).isoformat() if not is_thread else None,
                "revision_hash": revision_hash,
                "block_membership_fingerprint": block_membership_fingerprint,
                "assignment_generation": assignment_generation,
                "rebuild_generation": rebuild_generation,
                "bridge_not_merged": bridge_not_merged,
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


def _new_poll_audit(conversation_id: str) -> dict:
    return {
        "raw_conversation_id": conversation_id,
        "field_contract_version": "teams_chatsvc_rest_v1",
        "access_probe_status": "ok",
        "pagination_complete": False,
        "stop_reason": "",
        "page_count": 0,
        "raw_messages_seen": 0,
        "unique_message_keys_seen": 0,
        "selected_message_keys_seen": 0,
        "duplicate_raw_messages": 0,
        "parse_filtered_messages": 0,
        "upsert_new": 0,
        "upsert_updated": 0,
        "upsert_unchanged": 0,
        "explicit_delete_markers": 0,
        "missing_once_candidates": 0,
        "ledger_action_basis": "",
        "_seen_message_keys": set(),
        "_selected_message_keys": set(),
    }


def _raw_message_key(conversation_id: str, message: dict) -> tuple[str, str, str]:
    message_id = str(message.get("id") or message.get("clientmessageid") or "")
    root_id = str(
        message.get("rootMessageId")
        or message.get("properties", {}).get("rootMessageId")
        or message.get("properties", {}).get("parentMessageId")
        or message_id
    )
    return (str(message.get("conversationid") or conversation_id), root_id, message_id)


def _parsed_message_key(conversation_id: str, message: dict) -> tuple[str, str, str]:
    message_id = str(message.get("id") or "")
    root_id = str(message.get("rootMessageId") or message.get("parentMessageId") or message_id)
    return (str(message.get("conversationid") or conversation_id), root_id, message_id)


def _raw_message_timestamp(message: dict) -> str | None:
    timestamp = message.get("composetime") or message.get("originalarrivaltime")
    if not timestamp:
        return None
    parsed = _TeamsAPIClient._parse_timestamp(str(timestamp))
    if parsed.year <= 2000:
        return None
    return parsed.isoformat()


def _min_iso(current: str | None, candidate: str) -> str:
    if not current:
        return candidate
    return min(current, candidate)


def _max_iso(current: str | None, candidate: str) -> str:
    if not current:
        return candidate
    return max(current, candidate)


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _teams_string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("displayName") or item.get("name") or item.get("id") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _teams_attachment_names(value: object) -> list[str]:
    if isinstance(value, dict):
        value = [value]
    elif isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        return []

    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = str(
                item.get("fileName")
                or item.get("name")
                or item.get("title")
                or item.get("contentUrl")
                or "attachment"
            ).strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return names


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
