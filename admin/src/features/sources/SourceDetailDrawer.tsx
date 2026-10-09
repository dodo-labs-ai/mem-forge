import { formatCount, formatDateTime, formatInterval, formatRelative, formatUntil } from "@/lib/format";
import { ActionCard, DetailDrawer, DrawerSection, PropertyList, StatusBadge } from "@/patterns";
import { Button } from "@/ui/button";
import { presentSourceSyncActivity } from "./model/sourceSyncActivity";
import type { SourceRow } from "./model/sourceRows";
import { sourceRunsOn } from "./sourceRunsOn";
import { SourceIcon } from "./SourceIcon";
import type { SourceActions } from "./useSourceActions";

interface SourceDetailDrawerProps {
  row: SourceRow | null;
  typeLabel: string;
  projectName: string;
  actions: SourceActions;
  onOpenChange: (open: boolean) => void;
}

export function SourceDetailDrawer({ row, typeLabel, projectName, actions, onOpenChange }: SourceDetailDrawerProps) {
  if (row === null) return <DetailDrawer open={false} onOpenChange={onOpenChange} title="" children={null} />;
  const { source, activity, attention } = row;
  const presentation = activity ? presentSourceSyncActivity(activity, source.name, "items") : null;
  const schedule = source.sync_schedule;
  const failedItems = activity?.error?.items ?? [];
  const sync = actions.syncControl(source, activity);

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
        source.capabilities?.can_sync && source.status === "active" ? (
          <Button size="sm" variant="outline" disabled={!sync.enabled} onClick={() => actions.syncNow(source, sync.retryTarget)}>
            {sync.label}
          </Button>
        ) : null
      }
    >
      <PropertyList
        layout="row"
        items={[
          { label: "Runs on", value: sourceRunsOn(row).label },
          { label: "Access", value: source.access_policy === "private" ? "Only me" : "Workspace" },
          { label: "Saves to", value: projectName },
        ]}
      />

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

      <DrawerSection title="Sync">
        <PropertyList
          items={[
            {
              label: "Status",
              value: presentation ? (
                <StatusBadge tone={activity?.state === "failed" ? "danger" : activity?.state === "success" ? "ok" : "live"}>
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
      </DrawerSection>

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

      {failedItems.length > 0 ? (
        <DrawerSection title="Items that failed">
          <ul className="divide-y rounded-lg border">
            {failedItems.map((item) => (
              <li key={item.doc_id} className="space-y-0.5 p-3">
                <p className="truncate font-medium text-foreground">{item.title || item.doc_id}</p>
                <p className="text-xs text-muted-foreground">{item.error}</p>
              </li>
            ))}
          </ul>
        </DrawerSection>
      ) : null}
    </DetailDrawer>
  );
}
