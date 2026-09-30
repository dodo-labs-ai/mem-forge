import { ArrowUpRight, Link2 } from "lucide-react";
import type { MouseEvent } from "react";
import { Link } from "react-router-dom";
import { formatCount, formatRelative, pluralize } from "@/lib/format";
import { MemoryStatusLabel, MemoryTypeBadge, OnlyMeLabel, ProjectLabel, SourceLabel } from "./MemoryBadges";
import { useMemoryTable } from "./memoryTableContext";
import { sourceName, sourceSummary, type MemoryRow, type RelationHint } from "./model/memoryRows";
import { projectBadge } from "./model/projects";

/** Links inside a row open their own target instead of the row's. */
function stopRowClick(event: MouseEvent) {
  event.stopPropagation();
}

function RelationHintLink({ hint }: { hint: RelationHint }) {
  if (hint.kind === "conflicts") {
    return (
      <Link
        to={hint.target}
        onClick={stopRowClick}
        className="inline-flex items-center gap-1 text-xs font-medium text-tone-danger-foreground hover:underline"
      >
        <Link2 aria-hidden className="size-3" />
        {pluralize(hint.count, "conflict")}
      </Link>
    );
  }
  return (
    <Link
      to={hint.target}
      onClick={stopRowClick}
      className="inline-flex items-center gap-1 text-xs font-medium text-tone-live-foreground hover:underline"
    >
      <ArrowUpRight aria-hidden className="size-3" />
      Updated by a newer memory
    </Link>
  );
}

export function MemoryCell({ row }: { row: MemoryRow }) {
  const { projects, typeLabels } = useMemoryTable();
  const project = row.context ? projectBadge(row.context.projectKey, projects) : null;
  const sources = row.context ? sourceSummary(row.context.sources) : null;
  return (
    <div className="flex min-w-0 flex-col gap-1.5 whitespace-normal">
      <Link to={row.target} onClick={stopRowClick} className="font-medium text-foreground hover:underline">
        {row.content}
      </Link>
      <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1">
        {project ? <ProjectLabel project={project} /> : null}
        {sources ? (
          <span className="inline-flex min-w-0 items-center gap-1">
            <SourceLabel source={sources.first} name={sourceName(sources.first, typeLabels)} />
            {sources.more > 0 ? (
              <span className="text-xs text-muted-foreground">and {formatCount(sources.more)} more</span>
            ) : null}
          </span>
        ) : null}
        {row.context?.isPrivate ? <OnlyMeLabel /> : null}
        {row.relationHints.map((hint) => (
          <RelationHintLink key={hint.kind} hint={hint} />
        ))}
      </div>
    </div>
  );
}

export function TypeCell({ row }: { row: MemoryRow }) {
  return <MemoryTypeBadge type={row.memoryType} />;
}

export function StatusCell({ row }: { row: MemoryRow }) {
  return <MemoryStatusLabel status={row.status} />;
}

export function SupportCell({ row }: { row: MemoryRow }) {
  return <div className="text-right font-mono text-subtle-foreground tabular-nums">{formatCount(row.supportCount)}</div>;
}

export function AgeCell({ row }: { row: MemoryRow }) {
  return <div className="text-right text-muted-foreground">{row.seenAt ? formatRelative(row.seenAt) : null}</div>;
}
