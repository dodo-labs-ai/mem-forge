from memforge.retrieval.rank_fusion import (
    RankedChannelItem,
    weighted_reciprocal_rank_fusion,
)


def test_weighted_rrf_fuses_ranked_channels_and_exposes_contributions() -> None:
    fused = weighted_reciprocal_rank_fusion(
        channels={
            "vector": (
                RankedChannelItem("mem-a", score=0.90),
                RankedChannelItem("mem-b", score=0.80),
            ),
            "lexical": (
                RankedChannelItem("mem-b", score=8.0),
                RankedChannelItem("mem-c", score=7.0),
            ),
        },
        weights={"vector": 1.0, "lexical": 1.0},
        k=60,
    )

    assert [item.item_id for item in fused] == ["mem-b", "mem-a", "mem-c"]
    assert [part.channel for part in fused[0].contributions] == ["vector", "lexical"]
    assert [part.rank for part in fused[0].contributions] == [2, 1]


def test_weighted_rrf_accepts_explicit_rank_and_multiplier_for_graph_channel() -> None:
    fused = weighted_reciprocal_rank_fusion(
        channels={
            "graph": (
                RankedChannelItem("mem-specific", score=0.1, rank=2, multiplier=1.0),
                RankedChannelItem("mem-generic", score=0.9, rank=1, multiplier=0.2),
            ),
        },
        weights={"graph": 2.0},
        k=60,
    )

    assert [item.item_id for item in fused] == ["mem-specific", "mem-generic"]
    assert fused[0].contributions[0].weighted_score == 2.0 / 62.0
    assert fused[1].contributions[0].weighted_score == 2.0 * 0.2 / 61.0


def test_weighted_rrf_breaks_score_and_rank_ties_by_id() -> None:
    fused = weighted_reciprocal_rank_fusion(
        channels={
            "vector": (
                RankedChannelItem("mem-b", score=0.5),
                RankedChannelItem("mem-a", score=0.5),
            )
        },
        weights={"vector": 1.0},
        k=60,
    )

    assert [item.item_id for item in fused] == ["mem-a", "mem-b"]


def test_weighted_rrf_counts_an_item_at_most_once_per_channel() -> None:
    fused = weighted_reciprocal_rank_fusion(
        channels={
            "vector": (
                RankedChannelItem("mem-a", score=0.9),
                RankedChannelItem("mem-a", score=0.8),
                RankedChannelItem("mem-b", score=0.7),
            ),
            "lexical": (RankedChannelItem("mem-a", score=5.0),),
        },
        weights={"vector": 1.0, "lexical": 1.0},
        k=60,
    )

    mem_a = next(item for item in fused if item.item_id == "mem-a")
    assert [part.channel for part in mem_a.contributions] == ["vector", "lexical"]
