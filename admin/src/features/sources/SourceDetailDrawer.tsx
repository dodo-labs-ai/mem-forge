import { CircleAlert, Info, MoreHorizontal, RefreshCw } from "lucide-react";
import { useSharedWorkspace } from "@/api";
import { formatCount, formatDateTime, formatInterval, formatRelative, formatUntil, pluralize } from "@/lib/format";
import { ActionCard, DetailDrawer, DrawerSection, Notice, PropertyList, StatusBadge } from "@/patterns";
import { Button } from "@/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/ui/dropdown-menu";
import { AgentSessionCoverage } from "./AgentSessionCoverage";
import { AGENT_SESSION_SOURCE_TYPE } from "./model/agentSessions";
import { isManagedSourceType } from "./model/managedSources";
import { canChangeSourceAccess } from "./model/sourceAccess";
import { BRAND_MARKS, sourceBrandKeysFor } from "./model/sourceBrand";
import { failedItemsTitle, retryingAfterUpgradeTitle } from "./model/sourceFailedItems";
import { presentSourceSyncActivity } from "./model/sourceSyncActivity";
import type { SourceRow } from "./model/sourceRows";
import type { Source } from "./model/types";
import { sourceRunsOn } from "./sourceRunsOn";
import { SourceAccessLabel, SourceAccessTransitionSection } from "./SourceAccess";
import { SourceEvaluationChip } from "./SourceEvaluationChip";
import { SourceIcon } from "./SourceIcon";
import { FullSyncMenuItem } from "./SourceRowActions";
import { SourceScopeTransitionSection } from "./SourceScopeTransitionSection";
import type { SourceActions } from "./useSourceActions";

interface SourceDetailDrawerProps {
  row: SourceRow | null;
  typeLabel: string;
  projectName: string;
  actions: SourceActions;
  onChangeAccess: (source: Source) => void;
  onFullSync: (source: Source) => void;
  onOpenChange: (open: boolean) => void;
}

function unrecognizedSessionLinesTitle(count: number, type: string, client: string | null | undefined): string {
  const brandKey = sourceBrandKeysFor(type, client)?.[0];
  const clientLabel = brandKey ? BRAND_MARKS[brandKey].label : "this client";
  return `${pluralize(count, "session line")} from ${clientLabel} ${count === 1 ? "was" : "were"} not recognized`;
}

export function SourceDetailDrawer({
  row,
  typeLabel,
  projectName,
  actions,
  onChangeAccess,
  onFullSync,
  onOpenChange,
}: SourceDetailDrawerProps) {
  const sharedWorkspace = useSharedWorkspace();
  if (row === null) return <DetailDrawer open={false} onOpenChange={onOpenChange} title="" children={null} />;
  const { source, activity, attention } = row;
  const presentation = activity ? presentSourceSyncActivity(activity, source.name, "items") : null;
  const schedule = source.sync_schedule;
  const { failedItems } = row;
  const unrecognizedLines = source.unrecognized_session_lines;
  const sync = actions.syncControl(source, activity);
  const canStop = actions.canStopSync(source, activity);
  const canSync = Boolean(source.capabilities?.can_sync) && source.status === "active";
  const canChangeAccess = canChangeSourceAccess(source, sharedWorkspace);
  const fullSync = actions.fullSyncControl(source, activity);
  const hasMenu = canChangeAccess || fullSync.visible;

  return (
    <DetailDrawer
      open
      onOpenChange={onOpenChange}
      title={source.name}
      subtitle={typeLabel}
      leading={
        <span className="flex size-9 shrink-0 items-center justify-center rounded-md border bg-surface-subtle">
          <SourceIcon type={source.type} client={source.client} className="size-5" />
        </span>
      }
      actions={
        canStop || canSync || hasMenu ? (
          <>
            {canStop ? (
              <Button size="sm" variant="outline" onClick={() => actions.stopSync(source)}>
                Stop sync
              </Button>
            ) : null}
            {canSync ? (
              <Button size="sm" variant="outline" disabled={!sync.enabled} onClick={() => actions.syncNow(source, sync.retryTarget)}>
                {sync.label}
              </Button>
            ) : null}
            {hasMenu ? (
              <DropdownMenu>
                <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`More actions for ${source.name}`} />}>
                  <MoreHorizontal />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-64">
                  {fullSync.visible ? (
                    <FullSyncMenuItem enabled={fullSync.enabled} onClick={() => onFullSync(source)} />
                  ) : null}
                  {canChangeAccess ? (
                    <DropdownMenuItem onClick={() => onChangeAccess(source)}>Change who can use this source</DropdownMenuItem>
                  ) : null}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null}
          </>
        ) : null
      }
    >
      <PropertyList
        layout="row"
        items={[
          { label: "Runs on", value: sourceRunsOn(row).label },
          ...(sharedWorkspace
            ? [{ label: "Access", value: <SourceAccessLabel source={source} /> }]
            : []),
          { label: "Saves to", value: projectName },
        ]}
      />

      <SourceEvaluationChip source={source} />

      {attention ? (
        <ActionCard
          tone={attention.reason === "overdue" || attention.reason === "unmapped" ? "warn" : "danger"}
          title={attention.title}
          description={attention.detail}
          action={
            attention.action === "view_details" ? null : (
              <Button size="sm" disabled={!actions.attentionControl(attention.action, source, attention.actionLabel).enabled} onClick={() => actions.runAttention(attention.action, source)}>
                {actions.attentionControl(attention.action, source, attention.actionLabel).label}
              </Button>
            )
          }
        />
      ) : null}

      {unrecognizedLines ? (
        <Notice
          tone="warn"
          icon={CircleAlert}
          title={unrecognizedSessionLinesTitle(unrecognizedLines.count, source.type, source.client)}
        >
          Update the MemForge plugin so new sessions are read in full.
        </Notice>
      ) : null}

      {sharedWorkspace ? <SourceAccessTransitionSection source={source} /> : null}

      <SourceScopeTransitionSection source={source} activity={activity} />

      {isManagedSourceType(source.type) ? (
        // Its plugin delivers the sessions, so the source has no sync runs or schedule.
        <DrawerSection title="Sessions">
          <PropertyList
            items={[
              {
                label: "Last session",
                value: source.last_sync ? formatRelative(source.last_sync) : "None yet",
                hint: source.last_sync ? formatDateTime(source.last_sync) : undefined,
              },
            ]}
          />
        </DrawerSection>
      ) : (
        <DrawerSection title="Sync">
          <PropertyList
            items={[
              {
                label: "Status",
                value: presentation ? (
                  <StatusBadge
                    tone={
                      activity?.state === "failed"
                        ? "danger"
                        : activity?.state === "success"
                          ? "ok"
                          : activity?.state === "stopped"
                            ? "idle"
                            : "live"
                    }
                  >
                    {presentation.message}
                  </StatusBadge>
                ) : (
                  "Not synced yet"
                ),
                // The attention card above already explains a problem in full.
                hint: attention ? undefined : presentation?.detail,
              },
              {
                label: "Last sync",
                value: source.last_sync ? formatRelative(source.last_sync) : "Never",
                hint: source.last_sync ? formatDateTime(source.last_sync) : undefined,
              },
              {
                label: "Schedule",
                value: schedule?.enabled ? formatInterval(schedule.interval_minutes) : "Manual only",
                hint:
                  schedule?.enabled && schedule.next_run_at && source.status === "active"
                    ? `Next run ${formatUntil(schedule.next_run_at)}`
                    : undefined,
              },
            ]}
          />
          {presentation?.notice ? <Notice tone="idle" icon={Info}>{presentation.notice}</Notice> : null}
        </DrawerSection>
      )}

      <DrawerSection title="Content">
        <PropertyList
          layout="row"
          items={[
            { label: "Items synced", value: formatCount(source.doc_count) },
            { label: "Memories", value: formatCount(source.memory_count ?? row.memoryCount) },
            { label: "Used in my searches", value: source.enabled_for_me ?? true ? "Yes" : "No" },
          ]}
        />
      </DrawerSection>

      {source.type === AGENT_SESSION_SOURCE_TYPE ? <AgentSessionCoverage source={source} /> : null}

      {failedItems ? (
        <DrawerSection title={failedItemsTitle(failedItems.count)}>
          {failedItems.retryingAfterUpgrade > 0 ? (
            <Notice tone="live" icon={RefreshCw} title={retryingAfterUpgradeTitle(failedItems.retryingAfterUpgrade)}>
              MemForge was upgraded. The sync on its way retries each item that failed before the upgrade once with the
              new version.
            </Notice>
          ) : null}
          <p className="text-sm text-muted-foreground">
            Syncing this source retries every failed item. If a reason points to access or sign-in, fix that first,
            then sync.
          </p>
          <ul className="divide-y rounded-lg border">
            {failedItems.items.map((item) => (
              <li key={item.documentId} className="space-y-0.5 p-3">
                <div className="flex items-center gap-2">
                  <p className="min-w-0 flex-1 truncate font-medium text-foreground">{item.title}</p>
                  {item.retryingAfterUpgrade ? <StatusBadge tone="live">Retrying</StatusBadge> : null}
                </div>
                <p className="text-xs text-muted-foreground">
                  {item.stage ? <span className="font-medium text-subtle-foreground">{item.stage}: </span> : null}
                  {item.reason}
                </p>
              </li>
            ))}
          </ul>
          {failedItems.count > failedItems.items.length ? (
            <p className="text-xs text-muted-foreground">
              Showing the {formatCount(failedItems.items.length)} most recent of {formatCount(failedItems.count)} failed
              items.
            </p>
          ) : null}
        </DrawerSection>
      ) : null}
    </DetailDrawer>
  );
}
