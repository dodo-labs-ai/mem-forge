import type { components } from "@/api";
import type { AccessPolicy } from "../../model/sourceAccess";
import type { Source } from "../../model/types";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];
export type SyncSchedule = components["schemas"]["SourceSyncScheduleRequest"];

/**
 * The values of the steps the setup page owns around the type's groups:
 * Basics (`name`), Access (`access_policy`), Project (`project_binding`) and
 * Schedule (`sync_schedule`).
 */

/** In a workspace of one, every Source is the owner's own. */
export const PERSONAL_ACCESS_POLICY: AccessPolicy = "private";

/** How often an automatic sync can run. */
export const SCHEDULE_INTERVALS: readonly { minutes: number; label: string }[] = [
  { minutes: 30, label: "Every 30 minutes" },
  { minutes: 60, label: "Hourly" },
  { minutes: 360, label: "Every 6 hours" },
  { minutes: 720, label: "Every 12 hours" },
  { minutes: 1440, label: "Daily" },
  { minutes: 10080, label: "Weekly" },
];

/** The interval offered when the type names none. */
const FALLBACK_INTERVAL_MINUTES = 1440;

export function scheduleIntervalLabel(minutes: number): string {
  return SCHEDULE_INTERVALS.find((interval) => interval.minutes === minutes)?.label ?? `Every ${minutes} minutes`;
}

/** The intervals to choose from, with the current one added when it is not among them. */
export function scheduleIntervalChoices(current: number): { minutes: number; label: string }[] {
  if (SCHEDULE_INTERVALS.some((interval) => interval.minutes === current)) return [...SCHEDULE_INTERVALS];
  return [...SCHEDULE_INTERVALS, { minutes: current, label: scheduleIntervalLabel(current) }].sort((a, b) => a.minutes - b.minutes);
}

/** The schedule a form starts from: the Source's own, or off at the type's default interval. */
export function initialSchedule(source: Source | undefined, gene: Pick<GeneMetadata, "default_sync_interval_minutes">): SyncSchedule {
  if (source?.sync_schedule) {
    return { enabled: Boolean(source.sync_schedule.enabled), interval_minutes: source.sync_schedule.interval_minutes };
  }
  const interval = gene.default_sync_interval_minutes > 0 ? gene.default_sync_interval_minutes : FALLBACK_INTERVAL_MINUTES;
  return { enabled: false, interval_minutes: interval };
}

export function scheduleSummary(schedule: SyncSchedule): string {
  return schedule.enabled ? scheduleIntervalLabel(schedule.interval_minutes) : "Manual only";
}

/** The binding a fixed project choice saves; no project leaves the Source without one. */
export function fixedProjectBinding(projectKey: string | null): { mode: "fixed"; project_key: string } | null {
  return projectKey === null ? null : { mode: "fixed", project_key: projectKey };
}
