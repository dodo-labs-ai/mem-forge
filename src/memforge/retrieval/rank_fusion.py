"""Storage-neutral rank fusion primitives shared by retrieval workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RankedChannelItem:
    """One item ranked inside an independent retrieval channel.

    ``score`` determines rank when ``rank`` is absent.  ``rank`` exists for
    channels such as entity-graph traversal whose rank was computed before
    channel contributions were coalesced.  ``multiplier`` keeps channel-local
    specificity without requiring raw scores to be comparable across channels.
    """

    item_id: str
    score: float
    rank: int | None = None
    multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class RankContribution:
    channel: str
    rank: int
    channel_weight: float
    multiplier: float
    weighted_score: float


@dataclass(frozen=True, slots=True)
class FusedRankedItem:
    item_id: str
    score: float
    contributions: tuple[RankContribution, ...]


def weighted_reciprocal_rank_fusion(
    *,
    channels: Mapping[str, Sequence[RankedChannelItem]],
    weights: Mapping[str, float],
    k: int = 60,
) -> tuple[FusedRankedItem, ...]:
    """Fuse independently ranked channels with deterministic weighted RRF.

    Backend scores only order items *within* a channel.  They never cross the
    channel boundary, which keeps vector, lexical, and graph score scales from
    being compared accidentally.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")

    contributions_by_id: dict[str, list[RankContribution]] = {}
    channel_order = {channel: index for index, channel in enumerate(channels)}
    for channel, raw_items in channels.items():
        channel_weight = float(weights.get(channel, 1.0))
        has_explicit_ranks = any(item.rank is not None for item in raw_items)
        if has_explicit_ranks and any(item.rank is None for item in raw_items):
            raise ValueError(f"channel {channel!r} mixes explicit and implicit ranks")
        ranked_items = sorted(
            raw_items,
            key=(
                lambda item: (int(item.rank or 0), item.item_id)
                if has_explicit_ranks
                else (-float(item.score), item.item_id)
            ),
        )
        seen_item_ids: set[str] = set()
        unique_ranked_items: list[RankedChannelItem] = []
        for item in ranked_items:
            if item.item_id in seen_item_ids:
                continue
            seen_item_ids.add(item.item_id)
            unique_ranked_items.append(item)
        for fallback_rank, item in enumerate(unique_ranked_items, start=1):
            rank = item.rank if item.rank is not None else fallback_rank
            if rank < 1:
                raise ValueError("RRF ranks must be one-based")
            multiplier = max(0.0, float(item.multiplier))
            weighted_score = channel_weight * multiplier / (k + rank)
            contributions_by_id.setdefault(item.item_id, []).append(
                RankContribution(
                    channel=channel,
                    rank=rank,
                    channel_weight=channel_weight,
                    multiplier=multiplier,
                    weighted_score=weighted_score,
                )
            )

    fused: list[FusedRankedItem] = []
    for item_id, contributions in contributions_by_id.items():
        ordered_contributions = tuple(
            sorted(contributions, key=lambda part: channel_order[part.channel])
        )
        fused.append(
            FusedRankedItem(
                item_id=item_id,
                score=sum(part.weighted_score for part in ordered_contributions),
                contributions=ordered_contributions,
            )
        )
    return tuple(sorted(fused, key=lambda item: (-item.score, item.item_id)))
