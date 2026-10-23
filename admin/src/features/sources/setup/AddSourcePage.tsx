import { Link } from "react-router-dom";
import { localSyncState, useLocalSyncStatus, type LocalSyncState } from "@/features/local-sync";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";
import { newSourcePath, SOURCES_PATH } from "@/lib/paths";
import { BackLink, ErrorNotice, PageHeader, toneDot } from "@/patterns";
import { Badge } from "@/ui/badge";
import { Skeleton } from "@/ui/skeleton";
import { useSourceTypes } from "../api";
import { SourceIcon } from "../SourceIcon";
import { addableSourceTypes, canRunOnLocalSync, runBadge } from "./model/sourceTypePicker";

const LOCAL_SYNC_LABELS: Record<LocalSyncState, string> = {
  checking: "Checking local sync",
  online: "Local sync online",
  offline: "Local sync offline",
};

/** `/sources/new`: the source types a person can add, each opening its setup page. */
export function AddSourcePage() {
  const types = useSourceTypes();
  const localSync = localSyncState(useLocalSyncStatus());
  const addable = addableSourceTypes(types.data ?? []);

  return (
    <div className="space-y-6">
      <BackLink label="Sources" render={<Link to={SOURCES_PATH} />} />
      <PageHeader title="Add source" description="Choose where MemForge reads from. Each sync turns new and changed content into memories." />
      {types.isError ? (
        <ErrorNotice title="Could not load source types" message={errorMessage(types.error)} onRetry={() => void types.refetch()} />
      ) : null}
      {types.isPending ? (
        <div aria-busy="true" aria-label="Loading source types" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
        </div>
      ) : (
        <ul aria-label="Source types" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {addable.map((type) => (
            <li key={type.name}>
              <Link
                to={newSourcePath(type.name)}
                className="flex h-full flex-col gap-2 rounded-lg border p-4 outline-none transition-colors hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50"
              >
                <span className="flex items-center gap-2.5">
                  <SourceIcon type={type.name} className="size-5" />
                  <span className="font-medium text-foreground">{type.display_name}</span>
                </span>
                <span className="flex-1 text-sm text-muted-foreground">{type.description}</span>
                <span className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">{runBadge(type.execution_kinds)}</Badge>
                  {canRunOnLocalSync(type.execution_kinds) ? (
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span
                        aria-hidden
                        className={cn(
                          "size-2 rounded-full",
                          localSync === "online" ? toneDot.ok : localSync === "offline" ? toneDot.idle : toneDot.live,
                        )}
                      />
                      {LOCAL_SYNC_LABELS[localSync]}
                    </span>
                  ) : null}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
