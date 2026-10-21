"""Microsoft Teams Chat Service client.

Teams is read through the user's browser session (ADR 0002): the client uses
the Teams Access Token the ``memforge auth teams`` flow keeps in the OS
keychain, and never starts authentication UI itself. It reads conversations
and messages from the Chat Service and hosted images from Microsoft Graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urlsplit

import httpx

from memforge.conversations.port import PollOutcome
from memforge.genes.base import SourceConfigurationError
from memforge.local_agent.teams_contract import (
    TeamsMessageEvidenceError,
    parse_teams_edit_time,
    parse_teams_source_timestamp,
    validate_teams_provider_message,
)
from memforge.pipeline.normalizer_utils import html_to_markdown
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    SourceArtifactContractError,
    normalize_source_artifact_media_type,
    parse_source_artifact_content_length,
)

logger = logging.getLogger(__name__)

_CHAT_API_AUDIENCE = "https://ic3.teams.office.com"
_GRAPH_API_AUDIENCE = "https://graph.microsoft.com"

_REGION_URLS = {
    "emea": "https://teams.cloud.microsoft/api/chatsvc/emea/v1/users/ME",
    "amer": "https://teams.cloud.microsoft/api/chatsvc/amer/v1/users/ME",
    "apac": "https://teams.cloud.microsoft/api/chatsvc/apac/v1/users/ME",
}
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_MAX_PAGE_SIZE = 200


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

class _TeamsHostedContentHTMLParser(HTMLParser):
    """Collect exact Graph hosted-content identities from Teams HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        source = next((value for key, value in attrs if key.lower() == "src"), None)
        if not source:
            return
        parts = [unquote(part) for part in urlsplit(source).path.split("/") if part]
        for index, part in enumerate(parts[:-2]):
            if part.lower() != "hostedcontents" or parts[index + 2] != "$value":
                continue
            hosted_id = parts[index + 1].strip()
            if hosted_id and hosted_id not in self.ids:
                self.ids.append(hosted_id)


def _teams_hosted_content_ids(content: str) -> list[str]:
    """Return stable inline-image IDs without following arbitrary URLs."""

    if "<" not in content or "hostedcontent" not in content.lower():
        return []
    parser = _TeamsHostedContentHTMLParser()
    parser.feed(content)
    parser.close()
    return parser.ids


class AuthenticationError(SourceConfigurationError):
    """Raised when Teams authentication fails."""


class TeamsMessagePageSchemaError(ValueError):
    """Raised when a Chatsvc page cannot establish its message collection."""


class TeamsMessageRecordSchemaError(ValueError):
    """Raised when a memory-relevant message lacks stable source identity."""


class TeamsMessagePaginationError(ValueError):
    """Raised when Chatsvc pagination cannot prove forward progress."""


@dataclass(frozen=True, slots=True)
class TeamsMessagePoll:
    """The parsed messages of one poll and how its paging ended."""

    messages: tuple[dict, ...]
    outcome: PollOutcome
    earliest_complete_page_time: datetime | None = None
    stop_reason: str = ""


class TeamsChatClient:
    """Async HTTP client for Teams Chat API + Microsoft Graph API.

    Loads OAuth tokens via TeamsAuthenticator and provides typed methods
    for conversations, messages, teams, channels, and user resolution.
    """

    def __init__(self, region: str = "emea") -> None:
        self._region = region
        self._chat_base = _REGION_URLS.get(region, _REGION_URLS["emea"])
        self._chat_client: httpx.AsyncClient | None = None
        self._graph_client: httpx.AsyncClient | None = None

    async def _load_tokens(self) -> tuple[str, str | None]:
        """Load chat and graph tokens.

        The collection path is read-only and never starts authentication UI.
        """
        from memforge.auth.teams_auth import (
            TeamsAuthenticator,
        )

        tokens = TeamsAuthenticator.load_tokens()
        if not tokens:
            raise AuthenticationError("No Teams session found. Connect Teams from the source wizard.")

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
            raise AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
        return chat_token, graph_token

    async def _ensure_clients(self) -> None:
        """Create HTTP clients if not already initialized."""
        if self._chat_client:
            return

        chat_token, graph_token = await self._load_tokens()

        self._chat_client = httpx.AsyncClient(
            base_url=self._chat_base,
            headers={"Authorization": f"Bearer {chat_token}"},
            timeout=30.0,
        )
        if graph_token:
            self._graph_client = httpx.AsyncClient(
                base_url=_GRAPH_BASE,
                headers={"Authorization": f"Bearer {graph_token}"},
                timeout=30.0,
            )

    def _require_graph_client(self, capability: str) -> httpx.AsyncClient:
        if self._graph_client is None:
            raise AuthenticationError(
                f"{capability} requires Microsoft Graph access. "
                "Reconnect Teams after Graph API access is available."
            )
        return self._graph_client

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
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry on 429/5xx."""
        from memforge.pipeline.retry import retry_async

        async def _do_request() -> httpx.Response:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code == 401:
                raise AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
            if resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=resp.request,
                    response=resp,
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

    async def _request_bytes(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        max_bytes: int,
    ) -> tuple[bytes, Mapping[str, str]]:
        """Read one provider body incrementally under an explicit byte limit."""

        from memforge.pipeline.retry import retry_async

        async def _do_request() -> tuple[bytes, Mapping[str, str]]:
            async with client.stream("GET", url) as response:
                if response.status_code == 401:
                    raise AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "Rate limited",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                declared_size = parse_source_artifact_content_length(response.headers.get("content-length"))
                if declared_size is not None and declared_size > max_bytes:
                    raise SourceArtifactContractError("Teams hosted-content exceeds the Source Artifact storage limit")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(content) + len(chunk) > max_bytes:
                        raise SourceArtifactContractError(
                            "Teams hosted-content exceeds the Source Artifact storage limit"
                        )
                    content.extend(chunk)
                if declared_size is not None and declared_size != len(content):
                    raise SourceArtifactContractError(
                        "Teams hosted-content transport length does not match the response body"
                    )
                return bytes(content), dict(response.headers)

        return await retry_async(
            _do_request,
            max_retries=3,
            retryable_exceptions=(
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ),
            description=f"Teams API GET {url}",
        )

    # --- Chat API methods ---

    async def list_conversations(self) -> list[dict]:
        """List all conversations (channels + chats)."""
        await self._ensure_clients()
        resp = await self._request(
            self._chat_client,
            "GET",
            "/conversations",
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
            self._chat_client,
            "GET",
            "/conversations",
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
                teams[group_id]["channels"].append(
                    {
                        "id": conv_id,
                        "displayName": topic,
                        "favorite": is_favorite,
                    }
                )

        return list(teams.values())

    async def read_messages(
        self,
        conversation_id: str,
        *,
        start: datetime,
        end: datetime | None,
        budget_seconds: float,
    ) -> TeamsMessagePoll:
        """Read one conversation's messages created in ``[start, end)``, newest first.

        Paging stops when a page shows a message created before ``start``,
        when the conversation has nothing older, or when the time budget runs
        out; the result says which, so the reader knows what the poll proved.
        Pages overlap by one millisecond, so messages sharing a page boundary's
        time are never skipped.
        """

        await self._ensure_clients()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget_seconds
        url = f"/conversations/{conversation_id}/messages"
        end_time = _last_millisecond_before(end) if end is not None else None
        params = _teams_message_page_params(page_size=_MAX_PAGE_SIZE, end_time=end_time)
        messages: dict[str, dict] = {}
        earliest_complete_page_time: datetime | None = None
        previous_end_time: int | None = None
        pages = 0
        while True:
            if pages:
                if loop.time() >= deadline:
                    return TeamsMessagePoll(
                        messages=tuple(messages.values()),
                        outcome=PollOutcome.STOPPED_EARLY,
                        earliest_complete_page_time=earliest_complete_page_time,
                        stop_reason="time_budget_exhausted",
                    )
                # Teams allows 15 requests in 10 seconds.
                await asyncio.sleep(1.0)
            response = await self._request(self._chat_client, "GET", url, params=params)
            pages += 1
            page = _validated_teams_message_page(response.json())
            for raw in page:
                parsed = self._parse_message(raw)
                if parsed is None or parsed["time"] < start or (end is not None and parsed["time"] >= end):
                    continue
                messages.setdefault(parsed["id"], parsed)
            if page:
                earliest_complete_page_time = min(_teams_message_timestamp(raw) for raw in page)
            if page and earliest_complete_page_time < start:
                return TeamsMessagePoll(messages=tuple(messages.values()), outcome=PollOutcome.REACHED_START,
                                        stop_reason="reached_start")
            next_end_time = _next_teams_message_end_time(page, page_size=_MAX_PAGE_SIZE,
                                                         previous_end_time=previous_end_time)
            if next_end_time is None:
                return TeamsMessagePoll(messages=tuple(messages.values()), outcome=PollOutcome.HISTORY_EXHAUSTED,
                                        stop_reason="history_exhausted")
            previous_end_time = next_end_time
            params = _teams_message_page_params(page_size=_MAX_PAGE_SIZE, end_time=next_end_time)

    # --- Graph API methods ---

    async def get_hosted_content(
        self,
        *,
        conversation_id: str,
        message_id: str,
        hosted_content_id: str,
        conversation_type: str,
        team_id: str | None,
        root_message_id: str | None = None,
    ) -> tuple[bytes, str]:
        """Download one exact inline image through the documented Graph route."""

        await self._ensure_clients()
        graph_client = self._require_graph_client("Teams hosted-content")
        conversation = str(conversation_id or "").strip()
        message = str(message_id or "").strip()
        hosted = str(hosted_content_id or "").strip()
        if not conversation or not message or not hosted:
            raise SourceArtifactContractError("Teams hosted-content identity is incomplete")
        if conversation_type == "channel":
            normalized_team_id = str(team_id or "").strip()
            if not normalized_team_id:
                raise SourceArtifactContractError("Teams channel hosted-content requires an exact team identity")
            normalized_root_id = str(root_message_id or "").strip()
            message_path = f"/messages/{quote(normalized_root_id or message, safe='')}"
            if normalized_root_id and normalized_root_id != message:
                message_path += f"/replies/{quote(message, safe='')}"
            path = (
                f"/teams/{quote(normalized_team_id, safe='')}"
                f"/channels/{quote(conversation, safe='')}"
                f"{message_path}"
                f"/hostedContents/{quote(hosted, safe='')}/$value"
            )
        else:
            path = (
                f"/chats/{quote(conversation, safe='')}"
                f"/messages/{quote(message, safe='')}"
                f"/hostedContents/{quote(hosted, safe='')}/$value"
            )
        content, headers = await self._request_bytes(
            graph_client,
            path,
            max_bytes=MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
        )
        media_type = normalize_source_artifact_media_type(headers.get("content-type"))
        if media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES or not media_type.startswith("image/"):
            raise SourceArtifactContractError(
                f"Teams hosted-content media type is unsupported: {media_type or 'missing'}"
            )
        return content, media_type

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
            last_activity_str = c.get("lastMessage", {}).get("composetime") or c.get("lastActivity", "")
            last_activity = self._parse_timestamp(last_activity_str)

            # Extract last message sender info (useful for DM name resolution)
            last_msg = c.get("lastMessage", {})
            sender_display = last_msg.get("imdisplayname", "")
            sender_id = last_msg.get("from", "")

            result.append(
                {
                    "id": conv_id,
                    "topic": topic or "Untitled",
                    "team_name": str(c.get("threadProperties", {}).get("spaceThreadTopic") or "").strip() or None,
                    "lastActivity": last_activity,
                    "type": infer_conversation_type(conv_id),
                    "team_id": str(c.get("threadProperties", {}).get("groupId") or "").strip() or None,
                    "lastMessageSender": sender_display,
                    "lastMessageSenderId": sender_id,
                    "favorite": str(c.get("properties", {}).get("favorite", "")).lower() == "true",
                }
            )
        return result

    def _parse_message(self, m: dict) -> dict | None:
        """Parse a raw Teams message into a clean dict."""
        try:
            memory_relevant = validate_teams_provider_message(m)
        except TeamsMessageEvidenceError as exc:
            raise TeamsMessageRecordSchemaError(str(exc)) from exc
        if not memory_relevant:
            return None

        content = m["content"]
        hosted_content_ids = _teams_hosted_content_ids(content)
        # Convert HTML to clean text via normalizer_utils
        if "<" in content:
            content = html_to_markdown(content).strip()

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
        source_time = parse_teams_source_timestamp(composetime)
        message_id = str(m.get("id") or m.get("amsreferencesid") or "").strip()
        if not message_id or source_time is None:
            raise TeamsMessageRecordSchemaError("Teams message record is missing a stable id or source timestamp")
        root_message_id = (
            m.get("rootMessageId")
            or m.get("properties", {}).get("rootMessageId")
            or m.get("properties", {}).get("parentMessageId")
        )

        return {
            "id": message_id,
            "conversationid": m.get("conversationid"),
            "from": from_display or "Unknown",
            "content": content,
            "time": source_time,
            "edited_time": parse_teams_edit_time(m),
            "rootMessageId": root_message_id,
            "parentMessageId": m.get("properties", {}).get("parentMessageId"),
            "mentions": m.get("properties", {}).get("mentions", []),
            "attachments": m.get("amsreferences", []),
            "hosted_content_ids": hosted_content_ids,
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



# ============================================================================
# Error types
# ============================================================================


def _last_millisecond_before(end: datetime) -> int:
    """The latest whole millisecond, since the epoch, strictly before ``end``: the inclusive cursor of ``[start, end)``.

    An end just after a message's time still reads that message's millisecond.
    """

    microseconds = (end - _EPOCH) // timedelta(microseconds=1)
    return -(-microseconds // 1000) - 1


def infer_conversation_type(conversation_id: str) -> str:
    """The kind of a Teams conversation, as its id shows it."""
    if "@thread.tacv2" in conversation_id:
        return "channel"
    if "@unq.gbl.spaces" in conversation_id:
        return "individual_chat"
    return "group_chat"


def _validated_teams_message_page(data: object) -> list[dict]:
    """Return one Chat Service message page, or fail before the page proves anything."""

    messages: object
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    else:
        raise TeamsMessagePageSchemaError("Teams message page is missing an explicit messages list")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise TeamsMessagePageSchemaError("Teams message page has an invalid messages list")
    for message in messages:
        try:
            validate_teams_provider_message(message)
        except TeamsMessageEvidenceError as exc:
            raise TeamsMessageRecordSchemaError(str(exc)) from exc
    return messages


def _teams_message_page_params(*, page_size: int, end_time: int | None = None) -> dict:
    params: dict[str, int | str] = {
        "pageSize": page_size,
        "view": "msnp24Equivalent",
        "startTime": 0,
    }
    if end_time is not None:
        params["endTime"] = end_time
    return params


def _next_teams_message_end_time(
    messages: list[dict],
    *,
    page_size: int,
    previous_end_time: int | None,
) -> int | None:
    """The inclusive millisecond cursor of the next newest-first page; ``None`` when the conversation has no more.

    The cursor is the time of the page's oldest message, so the next page
    repeats the messages sharing that millisecond rather than skipping any.
    """

    if len(messages) < page_size:
        return None
    end_time = int(min(_teams_message_timestamp(message) for message in messages).timestamp() * 1000)
    if previous_end_time is not None and end_time >= previous_end_time:
        raise TeamsMessagePaginationError("Teams message time cursor did not advance")
    return end_time


def _teams_message_timestamp(message: Mapping[str, object]) -> datetime:
    timestamp = parse_teams_source_timestamp(
        message.get("composetime") or message.get("originalarrivaltime")
    )
    if timestamp is None:
        raise TeamsMessageRecordSchemaError(
            "Teams text message is missing a stable id or source timestamp"
        )
    return timestamp
