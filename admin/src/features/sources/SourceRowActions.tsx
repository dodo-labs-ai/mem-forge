import { MoreHorizontal } from "lucide-react";
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
import { sourceSyncActivityBlocksActions } from "./model/sourceSyncActivity";
import type { Source } from "./model/types";
import type { SourceActions } from "./useSourceActions";

interface SourceRowActionsProps {
  row: SourceRow;
  actions: SourceActions;
  onViewDetails: (source: Source) => void;
  onDelete: (source: Source) => void;
}

/** One primary button (the fix when the source needs the user) and a menu of the rest. */
export function SourceRowActions({ row, actions, onViewDetails, onDelete }: SourceRowActionsProps) {
  const { source, attention, activity } = row;
  const capabilities = source.capabilities;
  const busy = sourceSyncActivityBlocksActions(activity) || actions.syncPending(source.id);
  const canSync = Boolean(capabilities?.can_sync) && source.status === "active";

  return (
    <div className="flex items-center justify-end gap-1" onClick={(event) => event.stopPropagation()}>
      {attention ? (
        <Button size="sm" onClick={() => actions.runAttention(attention.action, source)}>
          {attention.actionLabel}
        </Button>
      ) : canSync ? (
        <Button size="sm" variant="outline" disabled={busy} onClick={() => actions.syncNow(source)}>
          {busy ? "Syncing" : "Sync now"}
        </Button>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`More actions for ${source.name}`} />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuItem onClick={() => onViewDetails(source)}>View details</DropdownMenuItem>
          <DropdownMenuItem disabled={!canSync || busy} onClick={() => actions.syncNow(source)}>
            Sync now
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
              <DropdownMenuItem disabled={!capabilities?.can_configure} onClick={() => actions.togglePaused(source)}>
                {source.status === "paused" ? "Resume syncing" : "Pause syncing"}
              </DropdownMenuItem>
              <DropdownMenuItem disabled={!capabilities?.can_configure} onClick={() => actions.openInV1()}>
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
