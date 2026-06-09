import type { ReactNode } from "react";
import { ChevronDown, ChevronRight, FolderTree } from "lucide-react";
import type { SourceProjectGroup } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  SHARED_PROJECT_KEY,
  UNSORTED_PROJECT_KEY,
} from "@/api/projectKeys";
import { projectGroupKey } from "./projectGrouping";

const UNMAPPED_GROUP_TITLE = "Unmapped";
const UNMAPPED_GROUP_DESCRIPTION =
  "No project assigned yet. Memories stay searchable in Unsorted until configured.";
const UNSORTED_GROUP_DESCRIPTION =
  "Searchable catch-all for memories without a project assignment.";

export function ProjectGroup({
  group,
  expanded,
  onToggle,
  children,
}: {
  group: SourceProjectGroup;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  const project = group.project;
  const isUnmapped = project === null;
  const isShared = project?.key === SHARED_PROJECT_KEY;
  const isUnsorted = project?.key === UNSORTED_PROJECT_KEY;
  const title = project ? project.name : UNMAPPED_GROUP_TITLE;
  const description = isUnmapped
    ? UNMAPPED_GROUP_DESCRIPTION
    : isUnsorted
      ? UNSORTED_GROUP_DESCRIPTION
      : null;
  const sourceCount = group.sources.length;

  return (
    <section className="border-b last:border-b-0">
      <header>
        <Button
          type="button"
          variant="ghost"
          aria-expanded={expanded}
          aria-controls={`project-group-${projectGroupKey(group)}`}
          onClick={onToggle}
          className="flex h-auto w-full items-start justify-between gap-3 rounded-none px-4 py-3 text-left hover:bg-muted/40"
        >
          <span className="flex min-w-0 items-start gap-2">
            {expanded ? (
              <ChevronDown className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRight className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            )}
            <FolderTree className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <span className="min-w-0">
              <span className="flex flex-wrap items-center gap-2">
                <span className="truncate text-sm font-semibold">{title}</span>
                {isShared && (
                  <Badge variant="secondary" className="text-[11px]">
                    team-wide
                  </Badge>
                )}
                {isUnmapped && (
                  <Badge variant="outline" className="text-[11px]">
                    needs setup
                  </Badge>
                )}
              </span>
              {description && (
                <span className="mt-0.5 block whitespace-normal text-xs font-normal text-muted-foreground">
                  {description}
                </span>
              )}
            </span>
          </span>
          <span className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
            <span>
              <span className="font-medium text-foreground">{sourceCount}</span>{" "}
              {sourceCount === 1 ? "source" : "sources"}
            </span>
            <span>
              <span className="font-medium text-foreground">{group.docCount}</span> docs
            </span>
            <span>
              <span className="font-medium text-foreground">{group.memoryCount}</span> memories
            </span>
          </span>
        </Button>
      </header>
      {expanded && (
        <div id={`project-group-${projectGroupKey(group)}`} className="divide-y border-t">
          {children}
        </div>
      )}
    </section>
  );
}
