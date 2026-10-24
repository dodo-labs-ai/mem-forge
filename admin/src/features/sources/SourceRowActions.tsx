import { MoreHorizontal } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { sourceEvaluationPath, sourceSettingsPath } from "@/lib/paths";
import { Button } from "@/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/ui/dropdown-menu";
import { canDeleteSourceType, canConfigureSourceType } from "./model/managedSources";
import type { SourceRow } from "./model/sourceRows";
import type { Source } from "./model/types";
import type { SourceActions } from "./useSourceActions";

interface SourceRowActionsProps {
  row: SourceRow;
  actions: SourceActions;
  onViewDetails: (source: Source) => void;
  onDelete: (source: Source) => void;
  onReRead: (source: Source) => void;
}

/** One primary button (the fix when the source needs the user) and a menu of the rest. */
export function SourceRowActions({ row, actions, onViewDetails, onDelete, onReRead }: SourceRowActionsProps) {
  const navigate = useNavigate();
  const { source, attention, activity } = row;
  const capabilities = source.capabilities;
  const sync = actions.syncControl(source, activity);
  const canSync = Boolean(capabilities?.can_sync) && source.status === "active";
  const canStop = actions.canStopSync(source, activity);
  const syncIsLive = activity !== undefined && ["queued", "active", "recovering"].includes(activity.state);
  // While a sync runs, the row's button stops it; a queued sync keeps its own button.
  const stopIsPrimary = canStop && (activity?.state === "active" || activity?.state === "recovering");
  const control = attention && actions.attentionControl(attention.action, source, attention.actionLabel);

  return (
    <div className="flex items-center justify-end gap-1" onClick={(event) => event.stopPropagation()}>
      {attention ? (
        <Button size="sm" disabled={!control?.enabled} onClick={() => actions.runAttention(attention.action, source)}>
          {control?.label}
        </Button>
      ) : stopIsPrimary ? (
        <Button size="sm" variant="outline" onClick={() => actions.stopSync(source)}>
          Stop sync
        </Button>
      ) : canSync ? (
        <Button size="sm" variant="outline" disabled={!sync.enabled} onClick={() => actions.syncNow(source, sync.retryTarget)}>
          {sync.label}
        </Button>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`More actions for ${source.name}`} />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuItem onClick={() => onViewDetails(source)}>View details</DropdownMenuItem>
          <DropdownMenuItem disabled={!canSync || !sync.enabled} onClick={() => actions.syncNow(source, sync.retryTarget)}>
            {sync.label}
          </DropdownMenuItem>
          {canStop ? (
            <DropdownMenuItem className="items-start" onClick={() => actions.stopSync(source)}>
              <span className="flex flex-col gap-0.5">
                <span>Stop sync</span>
                <span className="text-xs text-muted-foreground">
                  Stop this sync. What it saved stays, and the next sync runs as usual.
                </span>
              </span>
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuItem
            className="items-start"
            disabled={!capabilities?.can_configure || source.status !== "active" || !sync.enabled}
            onClick={() => onReRead(source)}
          >
            <span className="flex flex-col gap-0.5">
              <span>Re-read all documents</span>
              <span className="text-xs text-muted-foreground">Process every document again with the current models.</span>
            </span>
          </DropdownMenuItem>
          <DropdownMenuItem className="items-start" onClick={() => navigate(sourceEvaluationPath(source.id))}>
            <span className="flex flex-col gap-0.5">
              <span>Pipeline checks</span>
              <span className="text-xs text-muted-foreground">
                Open this source on the Evaluation page. It checks how the source is processed, not search quality.
              </span>
            </span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            checked={source.enabled_for_me ?? true}
            disabled={!capabilities?.can_subscribe}
            onCheckedChange={() => actions.toggleUsedInSearches(source)}
          >
            Use in my searches
          </DropdownMenuCheckboxItem>
          <DropdownMenuItem onClick={() => actions.togglePinned(source)}>
            {source.pinned_for_me ? "Unpin" : "Pin to top"}
          </DropdownMenuItem>
          {canConfigureSourceType(source.type) ? (
            <>
              <DropdownMenuItem
                className="items-start"
                // Pausing stops a sync, so it stays available during one; resuming waits for the sync to end.
                disabled={!capabilities?.can_configure || (source.status === "paused" && syncIsLive)}
                onClick={() => actions.togglePaused(source)}
              >
                {source.status === "paused" ? (
                  "Resume syncing"
                ) : (
                  <span className="flex flex-col gap-0.5">
                    <span>Pause syncing</span>
                    <span className="text-xs text-muted-foreground">
                      Stop any sync in progress and skip syncs until you resume.
                    </span>
                  </span>
                )}
              </DropdownMenuItem>
              <DropdownMenuItem disabled={!capabilities?.can_configure} onClick={() => navigate(sourceSettingsPath(source.id))}>
                Configure
              </DropdownMenuItem>
            </>
          ) : null}
          {canDeleteSourceType(source.type) ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                disabled={!capabilities?.can_delete}
                onClick={() => onDelete(source)}
              >
                Delete source
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
