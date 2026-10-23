import { useState } from "react";
import { ArrowRight, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { formatDateTime, formatRelative } from "@/lib/format";
import { ConfirmDialog, DrawerSection, Notice, PropertyList } from "@/patterns";
import { Button } from "@/ui/button";
import { useRestorePreviousScope, useSourceConfigLabels } from "./api";
import {
  presentSourceScopeTransition,
  scopeChangeErrorMessage,
  transitionFailedItemsTitle,
} from "./model/sourceScopeTransition";
import type { SourceSyncActivity } from "./model/sourceSyncActivity";
import type { Source } from "./model/types";

interface SourceScopeTransitionSectionProps {
  source: Source;
  activity: SourceSyncActivity | undefined;
}

/**
 * The scope change a source is applying: what it changes, how many of the
 * items its new scope lists are synced, the items whose failure keeps it
 * open, and the way back to the previous scope.
 */
export function SourceScopeTransitionSection({ source, activity }: SourceScopeTransitionSectionProps) {
  const labels = useSourceConfigLabels(source.scope_transition ? source.type : null);
  const restore = useRestorePreviousScope();
  const [confirming, setConfirming] = useState(false);
  const view = presentSourceScopeTransition(source, activity, labels);
  if (!view || !source.scope_transition) return null;
  const { restoreConfig } = view;
  const startedAt = source.scope_transition.created_at;

  return (
    <DrawerSection title="Scope change in progress">
      <Notice
        tone="live"
        icon={RefreshCw}
        title={view.status}
        action={
          restoreConfig ? (
            <Button size="sm" variant="outline" onClick={() => setConfirming(true)}>
              Restore the previous scope
            </Button>
          ) : null
        }
      >
        Nothing is removed until every item the new scope lists is synced. Each sync retries the items that failed.
      </Notice>

      <PropertyList
        items={[
          ...view.changes.map((change) => ({
            label: change.label,
            value: (
              <span className="inline-flex flex-wrap items-center gap-1.5">
                <span className="text-muted-foreground">{change.from}</span>
                <ArrowRight aria-label="changes to" className="size-3.5 shrink-0 text-muted-foreground" />
                <span>{change.to}</span>
              </span>
            ),
          })),
          {
            label: "Progress",
            value: view.progressLabel,
            hint: view.reason ? `Last sync: ${view.reason}` : undefined,
          },
          { label: "Started", value: formatRelative(startedAt), hint: formatDateTime(startedAt) },
        ]}
      />

      {view.failedItems ? (
        <div className="space-y-2">
          <p className="text-sm font-medium text-foreground">{transitionFailedItemsTitle(view.failedItems.count)}</p>
          <ul className="divide-y rounded-lg border">
            {view.failedItems.items.map((item) => (
              <li key={item.documentId} className="space-y-0.5 p-3">
                <p className="truncate font-medium text-foreground">{item.title}</p>
                <p className="text-xs text-muted-foreground">
                  {item.stage ? <span className="font-medium text-subtle-foreground">{item.stage}: </span> : null}
                  {item.reason}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {restoreConfig ? (
        <ConfirmDialog
          open={confirming}
          onOpenChange={setConfirming}
          title="Restore the previous scope?"
          description={
            <>
              This undoes the scope change in progress and saves {source.name}'s previous scope again, keeping its
              other settings. Nothing the previous scope synced is removed.
            </>
          }
          confirmLabel="Restore"
          onConfirm={async () => {
            try {
              await restore.mutateAsync({ sourceId: source.id, config: restoreConfig });
            } catch (error) {
              throw new Error(scopeChangeErrorMessage(error));
            }
            toast.success(`${source.name} returns to its previous scope`);
          }}
        />
      ) : null}
    </DrawerSection>
  );
}
