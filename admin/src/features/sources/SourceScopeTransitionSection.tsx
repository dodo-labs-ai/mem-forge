import { useState } from "react";
import { ArrowRight, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { formatDateTime, formatRelative } from "@/lib/format";
import { ConfirmDialog, DrawerSection, Notice, PropertyList } from "@/patterns";
import { Button } from "@/ui/button";
import { useRestorePreviousScope, useScopeChangePreview, useSourceConfigLabels } from "./api";
import {
  presentRestore,
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
 * The scope change a source is applying: what it changes, what it removes
 * and when, how far it is, the items whose failure keeps it open, and the way
 * back to the previous scope.
 */
export function SourceScopeTransitionSection({ source, activity }: SourceScopeTransitionSectionProps) {
  const labels = useSourceConfigLabels(source.scope_transition ? source.type : null);
  const restore = useRestorePreviousScope();
  const [confirming, setConfirming] = useState(false);
  const transition = source.scope_transition;
  const restoreConfig = transition?.restore_config ?? null;
  const preview = useScopeChangePreview(source.id, restoreConfig, confirming);
  const view = presentSourceScopeTransition(source, activity, labels);
  if (!view || !transition) return null;
  const restoring = presentRestore(source.name, preview.data, preview.isError);
  const startedAt = transition.created_at;

  return (
    <DrawerSection title="Scope change in progress">
      <Notice
        tone="live"
        icon={RefreshCw}
        title={view.returning ? "Returning to the previous scope" : view.status}
        action={
          restoreConfig ? (
            <Button size="sm" variant="outline" onClick={() => setConfirming(true)}>
              Restore the previous scope
            </Button>
          ) : null
        }
      >
        {view.summary}
      </Notice>

      <PropertyList
        items={[
          ...view.changes.map((change) => ({
            label: change.label,
            value: (
              <span className="inline-flex flex-wrap items-center gap-1.5">
                <span className="text-muted-foreground">{change.from}</span>
                <ArrowRight aria-hidden className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="sr-only">changes to</span>
                <span>{change.to}</span>
              </span>
            ),
          })),
          ...(view.returning ? [{ label: "Status", value: view.status }] : []),
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
          description={restoring.description}
          ready={restoring.ready}
          confirmLabel="Restore"
          onConfirm={async () => {
            try {
              await restore.mutateAsync({ sourceId: source.id, transitionId: transition.id, config: restoreConfig });
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
