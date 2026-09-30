import { MoreHorizontal } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { sourceEvaluationPath } from "@/lib/paths";
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
}

/** One primary button (the fix when the source needs the user) and a menu of the rest. */
export function SourceRowActions({ row, actions, onViewDetails, onDelete }: SourceRowActionsProps) {
  const navigate = useNavigate();
  const { source, attention, activity } = row;
  const capabilities = source.capabilities;
  const sync = actions.syncControl(source, activity);
  const canSync = Boolean(capabilities?.can_sync) && source.status === "active";

  return (
    <div className="flex items-center justify-end gap-1" onClick={(event) => event.stopPropagation()}>
      {attention ? (
        <Button size="sm" onClick={() => actions.runAttention(attention.action, source)}>
          {attention.actionLabel}
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
