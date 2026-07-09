"""Teams local-agent auth and conversation browse helpers."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
from typing import Any

import httpx

from memforge.auth.teams_auth import TeamsAuthenticator
from memforge.genes.teams_gene import AuthenticationError, _TeamsAPIClient

_CHAT_API_AUDIENCE = "https://ic3.teams.office.com"


def teams_auth_status() -> dict[str, Any]:
    """Return cached Teams auth status without exposing tokens."""
    tokens = TeamsAuthenticator.load_tokens()
    if not tokens:
        return {
            "authenticated": False,
            "expires_in_minutes": None,
            "error": "No Teams session found.",
        }

    validity = TeamsAuthenticator.check_token_expiry(tokens)
    if not any(validity.values()):
        return {
            "authenticated": False,
            "expires_in_minutes": None,
            "error": "Teams session expired.",
        }

    now = datetime.now(timezone.utc).timestamp()
    min_minutes = None
    for audience, is_valid in validity.items():
        entry = tokens.get(audience)
        if is_valid and isinstance(entry, dict):
            expires_at = entry.get("expiresAt", 0)
            if expires_at > 0:
                remaining = int((expires_at - now) / 60)
                if min_minutes is None or remaining < min_minutes:
                    min_minutes = remaining

    return {
        "authenticated": True,
        "expires_in_minutes": min_minutes,
        "error": None,
    }


async def browse_teams_conversations(*, region: str = "emea") -> dict[str, Any]:
    """Browse selectable Teams conversations using local cached chatsvc tokens."""
    tokens = TeamsAuthenticator.load_tokens()
    if not tokens:
        raise ValueError("No Teams session found.")

    chat_token = TeamsAuthenticator.get_token_for_audience(tokens, _CHAT_API_AUDIENCE)
    if not chat_token:
        raise ValueError("Missing Chat API token.")

    client = _TeamsAPIClient(region=region)
    client._chat_client = httpx.AsyncClient(
        base_url=client._chat_base,
        headers={"Authorization": f"Bearer {chat_token}"},
        timeout=30.0,
    )
    client._graph_client = httpx.AsyncClient(base_url="https://localhost", timeout=1.0)
    try:
        raw_channel_teams, raw_convos = await asyncio.gather(
            client.list_channels(),
            client.list_conversations(),
            return_exceptions=True,
        )
        auth_error = _first_auth_error(raw_channel_teams, raw_convos)
        if auth_error is not None:
            raise auth_error
        if isinstance(raw_channel_teams, Exception):
            raw_channel_teams = []
        if isinstance(raw_convos, Exception):
            raw_convos = []

        teams = [
            {
                "id": team["id"],
                "displayName": team["displayName"],
                "channels": [
                    {"id": channel["id"], "displayName": channel["displayName"]}
                    for channel in team["channels"]
                ],
            }
            for team in raw_channel_teams
            if team.get("channels")
        ]

        user_convos = [conversation for conversation in raw_convos if conversation.get("id", "").startswith("19:")]
        my_oid = _jwt_oid(chat_token)
        guid_to_name = _sender_guid_name_map(user_convos, my_oid)

        group_chats = sorted(
            [
                {
                    "id": conversation["id"],
                    "topic": conversation.get("topic", "Untitled"),
                    "lastActivity": _iso_or_none(conversation.get("lastActivity")),
                }
                for conversation in user_convos
                if conversation.get("type") == "group_chat"
            ],
            key=lambda item: item["lastActivity"] or "",
            reverse=True,
        )
        individual_chats = _dedupe_chats(
            sorted(
                [
                    {
                        "id": conversation["id"],
                        "topic": name,
                        "lastActivity": _iso_or_none(conversation.get("lastActivity")),
                    }
                    for conversation in user_convos
                    if conversation.get("type") == "individual_chat"
                    if (name := _resolve_dm_name(conversation, my_oid, guid_to_name))
                ],
                key=lambda item: item["lastActivity"] or "",
                reverse=True,
            )
        )
        favorites = _favorite_chats(raw_channel_teams)
        return {
            "favorites": favorites,
            "teams": teams,
            "group_chats": group_chats,
            "individual_chats": individual_chats,
        }
    finally:
        await client.close()


def _jwt_oid(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        jwt_data = json.loads(base64.b64decode(payload))
        return str(jwt_data.get("oid") or "")
    except Exception:
        return ""


def _first_auth_error(*results: Any) -> Exception | None:
    for result in results:
        if isinstance(result, AuthenticationError):
            return result
        if isinstance(result, httpx.HTTPStatusError) and result.response.status_code == 401:
            return AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
    return None


def _sender_guid_name_map(conversations: list[dict[str, Any]], my_oid: str) -> dict[str, str]:
    if not my_oid:
        return {}
    guid_to_name: dict[str, str] = {}
    for conversation in conversations:
        sender_id = str(conversation.get("lastMessageSenderId") or "")
        sender_name = str(conversation.get("lastMessageSender") or "")
        if sender_name and "orgid:" in sender_id:
            guid = sender_id.split("orgid:")[-1].split("/")[0]
            if guid and guid != my_oid:
                guid_to_name[guid] = sender_name
    return guid_to_name


def _resolve_dm_name(conversation: dict[str, Any], my_oid: str, guid_to_name: dict[str, str]) -> str:
    sender_id = str(conversation.get("lastMessageSenderId") or "")
    if not (my_oid and my_oid in sender_id):
        return str(conversation.get("lastMessageSender") or "")
    id_part = str(conversation.get("id") or "").replace("19:", "").split("@")[0]
    other_guid = next((guid for guid in id_part.split("_") if guid != my_oid), None)
    return guid_to_name.get(other_guid, "") if other_guid else ""


def _dedupe_chats(chats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for chat in chats:
        if chat["id"] in seen_ids:
            continue
        seen_ids.add(chat["id"])
        deduped.append(chat)
    return deduped


def _favorite_chats(raw_channel_teams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    favorites: list[dict[str, Any]] = []
    for team in raw_channel_teams:
        if not team.get("favorite"):
            continue
        favorite_channels = [channel for channel in team["channels"] if channel.get("favorite")]
        if favorite_channels:
            for channel in favorite_channels:
                favorites.append({"id": channel["id"], "topic": f"{team['displayName']} / {channel['displayName']}"})
        elif team["channels"]:
            favorites.append({"id": team["channels"][0]["id"], "topic": team["displayName"]})
    return favorites


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)
