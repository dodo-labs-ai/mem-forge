import { CircleAlert } from "lucide-react";
import { useProjects } from "@/api";
import { errorMessage } from "@/lib/errors";
import { formatCount, formatDateTime, formatRelative, pluralize } from "@/lib/format";
import { DrawerSection, ErrorNotice, Notice, PropertyList } from "@/patterns";
import { Skeleton } from "@/ui/skeleton";
import { useAgentSessionCompleteness, useSourceProjects } from "./api";
import type { Source } from "./model/types";

/**
 * The drawer's "Session coverage" for an agent-session Source: how the
 * session windows its plugin sent turned out (kept as memory, skipped as
 * low-signal, failed), and the projects the kept ones were filed under.
 */
export function AgentSessionCoverage({ source }: { source: Source }) {
  const completeness = useAgentSessionCompleteness(source.id);
  const projects = useSourceProjects(source.id);
  const projectNames = new Map((useProjects().data ?? []).map((project) => [project.key, project.name]));
  const failure = completeness.data?.latest_failure ?? null;

  return (
    <DrawerSection title="Session coverage">
      {completeness.isError ? (
        <ErrorNotice
          title="Could not load session coverage"
          message={errorMessage(completeness.error)}
          onRetry={() => void completeness.refetch()}
        />
      ) : completeness.data ? (
        <>
          <PropertyList
            layout="row"
            items={[
              { label: "Windows processed", value: formatCount(completeness.data.processed_total) },
              { label: "Kept as memory", value: formatCount(completeness.data.counts.knowledge_patched) },
              { label: "Skipped as low-signal", value: formatCount(completeness.data.counts.no_output) },
            ]}
          />
          <p className="text-sm text-muted-foreground">
            Most session windows are conversation or short-lived work. MemForge keeps the ones likely to help future
            work and skips the rest.
          </p>
          {failure ? (
            <Notice tone="warn" icon={CircleAlert} title={`${pluralize(failure.count, "window")} failed`}>
              {failure.reason ?? "MemForge could not process them."}
              {failure.last_seen_at ? ` Last failure ${formatRelative(failure.last_seen_at)}.` : null}
            </Notice>
          ) : null}
        </>
      ) : (
        <Skeleton aria-label="Loading session coverage" className="h-12" />
      )}

      <h4 className="text-sm font-medium">Projects with memories from sessions</h4>
      {projects.isError ? (
        <ErrorNotice title="Could not load the projects" message={errorMessage(projects.error)} onRetry={() => void projects.refetch()} />
      ) : !projects.data ? (
        <Skeleton aria-label="Loading projects" className="h-12" />
      ) : projects.data.length === 0 ? (
        <p className="text-sm text-muted-foreground">No session has been kept yet.</p>
      ) : (
        <ul aria-label="Projects with memories from sessions" className="divide-y rounded-lg border">
          {projects.data.map((project) => (
            <li key={project.project} className="flex flex-wrap items-baseline gap-x-4 gap-y-0.5 p-3 text-sm">
              <span className="min-w-0 flex-1 font-medium text-foreground">{projectNames.get(project.project) ?? project.project}</span>
              <span className="text-muted-foreground">{pluralize(project.document_count, "summary", "summaries")}</span>
              <span className="text-muted-foreground">{pluralize(project.memory_count, "memory", "memories")}</span>
              {project.last_observed_at ? (
                <span className="text-xs text-muted-foreground" title={formatDateTime(project.last_observed_at)}>
                  {formatRelative(project.last_observed_at)}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </DrawerSection>
  );
}
