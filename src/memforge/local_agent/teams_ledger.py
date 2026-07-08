"""Teams canonical message and window projection primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

TEAMS_LEDGER_STATE_VERSION = 1


@dataclass(frozen=True)
class TeamsMessageKey:
    source_id: str
    conversation_id: str
    root_message_id: str
    message_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "conversation_id": self.conversation_id,
            "root_message_id": self.root_message_id,
            "message_id": self.message_id,
        }


@dataclass(frozen=True)
class TeamsLedgerMessage:
    source_id: str
    conversation_id: str
    conversation_type: str
    message_id: str
    created_at: datetime
    body_normalized: str
    root_message_id: str | None = None
    parent_message_id: str | None = None
    modified_at: datetime | None = None
    deleted_state: str = "observed"

    @property
    def key(self) -> TeamsMessageKey:
        return TeamsMessageKey(
            source_id=self.source_id,
            conversation_id=self.conversation_id,
            root_message_id=self.root_message_id or self.message_id,
            message_id=self.message_id,
        )

    @property
    def body_hash(self) -> str:
        return _sha256(self.body_normalized)


@dataclass(frozen=True)
class TeamsBlockProjection:
    source_id: str
    conversation_id: str
    window_id: str
    frozen_anchor_message_id: str
    anchor_created_at: datetime
    member_min_created_at: datetime
    member_max_created_at: datetime
    member_message_ids: tuple[str, ...]
    block_membership_fingerprint: str
    revision_hash: str
    assignment_generation: int = 0
    rebuild_generation: int = 0
    bridge_not_merged: bool = False


@dataclass(frozen=True)
class TeamsProjectionResult:
    blocks: tuple[TeamsBlockProjection, ...]


class TeamsLedgerProjector:
    """Project unthreaded Teams messages into stable 60-minute windows."""

    def __init__(self, *, gap_minutes: int = 60) -> None:
        self.gap_minutes = gap_minutes
        self._gap = timedelta(minutes=gap_minutes)

    def project_unthreaded(
        self,
        messages: list[TeamsLedgerMessage],
        *,
        previous: TeamsProjectionResult | None = None,
    ) -> TeamsProjectionResult:
        ordered = sorted(messages, key=lambda msg: (msg.created_at, msg.message_id))
        if not ordered:
            return TeamsProjectionResult(blocks=())
        if previous is None or not previous.blocks:
            return TeamsProjectionResult(blocks=tuple(self._project_new_blocks(ordered)))
        return TeamsProjectionResult(blocks=tuple(self._project_against_snapshot(ordered, previous)))

    def _project_new_blocks(self, ordered: list[TeamsLedgerMessage]) -> list[TeamsBlockProjection]:
        groups: list[list[TeamsLedgerMessage]] = []
        current: list[TeamsLedgerMessage] = []
        for message in ordered:
            if not current:
                current = [message]
                continue
            gap = message.created_at - current[-1].created_at
            if gap > self._gap:
                groups.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            groups.append(current)
        return [self._build_block(group[0], group) for group in groups]

    def _project_against_snapshot(
        self,
        ordered: list[TeamsLedgerMessage],
        previous: TeamsProjectionResult,
    ) -> list[TeamsBlockProjection]:
        previous_by_member = {
            message_id: block
            for block in previous.blocks
            for message_id in block.member_message_ids
        }
        assigned: dict[str, list[TeamsLedgerMessage]] = {block.window_id: [] for block in previous.blocks}
        new_messages: list[TeamsLedgerMessage] = []
        for message in ordered:
            block = previous_by_member.get(message.message_id)
            if block is None:
                new_messages.append(message)
            else:
                assigned[block.window_id].append(message)

        for message in sorted(new_messages, key=lambda msg: (msg.created_at, msg.message_id)):
            block = self._select_snapshot_block(message, previous.blocks)
            if block is None:
                block = self._new_late_block(message)
                previous = replace(previous, blocks=(*previous.blocks, block))
                assigned[block.window_id] = []
            assigned[block.window_id].append(message)

        rebuilt: list[TeamsBlockProjection] = []
        for block in sorted(previous.blocks, key=lambda item: (item.anchor_created_at, item.window_id)):
            members = sorted(assigned.get(block.window_id, []), key=lambda msg: (msg.created_at, msg.message_id))
            if not members:
                continue
            rebuilt.append(self._build_block(members[0], members, previous_block=block))
        return rebuilt

    def _select_snapshot_block(
        self,
        message: TeamsLedgerMessage,
        blocks: tuple[TeamsBlockProjection, ...],
    ) -> TeamsBlockProjection | None:
        containing = [
            block for block in blocks
            if block.member_min_created_at <= message.created_at <= block.member_max_created_at
        ]
        if containing:
            return sorted(
                containing,
                key=lambda block: (abs((message.created_at - block.anchor_created_at).total_seconds()), block.window_id),
            )[0]

        before = [
            block for block in blocks
            if block.member_max_created_at < message.created_at
            and message.created_at - block.member_max_created_at <= self._gap
        ]
        after = [
            block for block in blocks
            if message.created_at < block.member_min_created_at
            and block.member_min_created_at - message.created_at <= self._gap
        ]
        if before and after:
            return sorted(before, key=lambda block: (message.created_at - block.member_max_created_at, block.window_id))[0]
        if before:
            return sorted(before, key=lambda block: (message.created_at - block.member_max_created_at, block.window_id))[0]
        if after:
            return sorted(after, key=lambda block: (block.member_min_created_at - message.created_at, block.window_id))[0]
        return None

    def _new_late_block(self, message: TeamsLedgerMessage) -> TeamsBlockProjection:
        return self._build_block(message, [message])

    def _build_block(
        self,
        anchor: TeamsLedgerMessage,
        members: list[TeamsLedgerMessage],
        *,
        previous_block: TeamsBlockProjection | None = None,
    ) -> TeamsBlockProjection:
        if previous_block is None:
            frozen_anchor_message_id = anchor.message_id
            anchor_created_at = anchor.created_at
            window_id = build_teams_window_id(
                source_id=anchor.source_id,
                conversation_id=anchor.conversation_id,
                root_or_anchor_message_id=frozen_anchor_message_id,
                window_type="time_block",
            )
            assignment_generation = 0
            rebuild_generation = 0
            bridge_not_merged = False
        else:
            frozen_anchor_message_id = previous_block.frozen_anchor_message_id
            anchor_created_at = previous_block.anchor_created_at
            window_id = previous_block.window_id
            assignment_generation = previous_block.assignment_generation + 1
            rebuild_generation = previous_block.rebuild_generation
            bridge_not_merged = previous_block.bridge_not_merged

        member_message_ids = tuple(message.message_id for message in members)
        member_min_created_at = min(message.created_at for message in members)
        member_max_created_at = max(message.created_at for message in members)
        membership_fingerprint = _block_membership_fingerprint(window_id, members)
        revision_hash = _revision_hash(
            window_id=window_id,
            rebuild_generation=rebuild_generation,
            block_membership_fingerprint=membership_fingerprint,
            messages=members,
            bridge_not_merged=bridge_not_merged,
        )
        return TeamsBlockProjection(
            source_id=anchor.source_id,
            conversation_id=anchor.conversation_id,
            window_id=window_id,
            frozen_anchor_message_id=frozen_anchor_message_id,
            anchor_created_at=anchor_created_at,
            member_min_created_at=member_min_created_at,
            member_max_created_at=member_max_created_at,
            member_message_ids=member_message_ids,
            block_membership_fingerprint=membership_fingerprint,
            revision_hash=revision_hash,
            assignment_generation=assignment_generation,
            rebuild_generation=rebuild_generation,
            bridge_not_merged=bridge_not_merged,
        )


class TeamsLedgerStateStore:
    """Durable JSON store for Teams block anchors and memberships."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def load_projection(self, *, source_id: str, conversation_id: str) -> TeamsProjectionResult | None:
        payload = self._load()
        record = payload.get("conversations", {}).get(_conversation_state_key(source_id, conversation_id))
        if not isinstance(record, dict):
            return None
        blocks = record.get("blocks")
        if not isinstance(blocks, list):
            return None
        return TeamsProjectionResult(blocks=tuple(_block_from_json(block) for block in blocks if isinstance(block, dict)))

    def save_projection(
        self,
        *,
        source_id: str,
        conversation_id: str,
        projection: TeamsProjectionResult,
    ) -> dict[str, Any]:
        payload = self._load()
        conversations = payload.setdefault("conversations", {})
        conversations[_conversation_state_key(source_id, conversation_id)] = {
            "source_id": source_id,
            "conversation_id": conversation_id,
            "blocks": [_block_to_json(block) for block in projection.blocks],
        }
        return self._save(payload)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": TEAMS_LEDGER_STATE_VERSION, "conversations": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": TEAMS_LEDGER_STATE_VERSION, "conversations": {}}
        if not isinstance(payload, dict) or payload.get("version") != TEAMS_LEDGER_STATE_VERSION:
            return {"version": TEAMS_LEDGER_STATE_VERSION, "conversations": {}}
        conversations = payload.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
        return {"version": TEAMS_LEDGER_STATE_VERSION, "conversations": conversations}

    def _save(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            "version": TEAMS_LEDGER_STATE_VERSION,
            "conversations": payload.get("conversations") if isinstance(payload.get("conversations"), dict) else {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(cleaned, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return cleaned


def build_teams_window_id(
    *,
    source_id: str,
    conversation_id: str,
    root_or_anchor_message_id: str,
    window_type: str,
) -> str:
    prefix = "teams-thread" if window_type == "thread" else "teams-block"
    payload = {
        "source_id": source_id,
        "conversation_id": conversation_id,
        "root_or_anchor_message_id": root_or_anchor_message_id,
        "window_type": window_type,
    }
    encoded = _urlsafe_json(payload)
    return f"{prefix}:v1:{encoded}"


def decode_teams_window_id(window_id: str) -> dict[str, str]:
    try:
        _prefix, version, encoded = window_id.split(":", 2)
    except ValueError as exc:
        raise ValueError("invalid Teams window id") from exc
    if version != "v1":
        raise ValueError(f"unsupported Teams window id version: {version}")
    decoded = _decode_urlsafe_json(encoded)
    return {
        "source_id": str(decoded["source_id"]),
        "conversation_id": str(decoded["conversation_id"]),
        "root_or_anchor_message_id": str(decoded["root_or_anchor_message_id"]),
        "window_type": str(decoded["window_type"]),
    }


def build_teams_receipt_key(*, source_id: str, window_id: str, revision_hash: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "window_id": window_id,
        "revision_hash": revision_hash,
    }


def _block_to_json(block: TeamsBlockProjection) -> dict[str, Any]:
    return {
        "source_id": block.source_id,
        "conversation_id": block.conversation_id,
        "window_id": block.window_id,
        "frozen_anchor_message_id": block.frozen_anchor_message_id,
        "anchor_created_at": block.anchor_created_at.isoformat(),
        "member_min_created_at": block.member_min_created_at.isoformat(),
        "member_max_created_at": block.member_max_created_at.isoformat(),
        "member_message_ids": list(block.member_message_ids),
        "block_membership_fingerprint": block.block_membership_fingerprint,
        "revision_hash": block.revision_hash,
        "assignment_generation": block.assignment_generation,
        "rebuild_generation": block.rebuild_generation,
        "bridge_not_merged": block.bridge_not_merged,
    }


def _block_from_json(value: dict[str, Any]) -> TeamsBlockProjection:
    return TeamsBlockProjection(
        source_id=str(value["source_id"]),
        conversation_id=str(value["conversation_id"]),
        window_id=str(value["window_id"]),
        frozen_anchor_message_id=str(value["frozen_anchor_message_id"]),
        anchor_created_at=_parse_dt(value["anchor_created_at"]),
        member_min_created_at=_parse_dt(value["member_min_created_at"]),
        member_max_created_at=_parse_dt(value["member_max_created_at"]),
        member_message_ids=tuple(str(item) for item in value.get("member_message_ids", [])),
        block_membership_fingerprint=str(value["block_membership_fingerprint"]),
        revision_hash=str(value["revision_hash"]),
        assignment_generation=int(value.get("assignment_generation") or 0),
        rebuild_generation=int(value.get("rebuild_generation") or 0),
        bridge_not_merged=bool(value.get("bridge_not_merged")),
    )


def _conversation_state_key(source_id: str, conversation_id: str) -> str:
    return _sha256_json({"source_id": source_id, "conversation_id": conversation_id})


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def _block_membership_fingerprint(window_id: str, messages: list[TeamsLedgerMessage]) -> str:
    payload = [
        {"window_id": window_id, **message.key.as_dict()}
        for message in sorted(messages, key=lambda msg: (msg.created_at, msg.message_id))
    ]
    return _sha256_json(payload)


def _revision_hash(
    *,
    window_id: str,
    rebuild_generation: int,
    block_membership_fingerprint: str,
    messages: list[TeamsLedgerMessage],
    bridge_not_merged: bool,
) -> str:
    payload: dict[str, Any] = {
        "window_id": window_id,
        "rebuild_generation": rebuild_generation,
        "block_membership_fingerprint": block_membership_fingerprint,
        "bridge_not_merged": bridge_not_merged,
        "messages": [
            {
                **message.key.as_dict(),
                "created_at": message.created_at.isoformat(),
                "body_hash": message.body_hash,
                "modified_at": message.modified_at.isoformat() if message.modified_at else None,
                "deleted_state": message.deleted_state,
            }
            for message in sorted(messages, key=lambda msg: (msg.created_at, msg.message_id))
        ],
    }
    return _sha256_json(payload)


def _urlsafe_json(payload: dict[str, str]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_urlsafe_json(value: str) -> dict[str, Any]:
    padding = "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8"))


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(raw)
