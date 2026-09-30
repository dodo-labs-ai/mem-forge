import { Link } from "react-router-dom";
import { formatCount } from "@/lib/format";
import { StatCard } from "@/patterns";
import { otherRelationsSummary, type RelationCounts } from "./model/relations";
import type { MemoryStats } from "./model/types";

function statusCount(stats: MemoryStats, status: string): number {
  return stats.by_status.find((entry) => entry.key === status)?.count ?? 0;
}

function countValue(count: number | undefined): string | undefined {
  return count === undefined ? undefined : formatCount(count);
}

interface MemoryStatCardsProps {
  stats: MemoryStats | undefined;
  openReviews: number | undefined;
  relationCounts: RelationCounts | undefined;
  reviewQueuePath: string;
  conflictsPath: string;
}

export function MemoryStatCards({ stats, openReviews, relationCounts, reviewQueuePath, conflictsPath }: MemoryStatCardsProps) {
  const historical = stats ? statusCount(stats, "superseded") + statusCount(stats, "retired") : undefined;
  const conflicts = relationCounts?.contradicts;
  return (
    <section aria-label="Overview" className="grid gap-3 md:grid-cols-3">
      <StatCard
        label="Active memories"
        value={countValue(stats ? statusCount(stats, "active") : undefined)}
        detail={historical === undefined ? null : `${formatCount(historical)} superseded or retired`}
      />
      <StatCard
        label="Waiting for review"
        value={countValue(openReviews)}
        tone={openReviews ? "warn" : undefined}
        detail="Current memories stay searchable until you decide"
        link={{ label: "Review queue", render: <Link to={reviewQueuePath} /> }}
      />
      <StatCard
        label="Conflicts between documents"
        value={countValue(conflicts)}
        tone={conflicts ? "danger" : undefined}
        detail={relationCounts ? otherRelationsSummary(relationCounts) : null}
        link={{ label: conflicts === 1 ? "Show conflict" : "Show conflicts", render: <Link to={conflictsPath} /> }}
      />
    </section>
  );
}
