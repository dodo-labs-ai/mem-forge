import { isManagedSourceType } from "./managedSources";
import type { SourceReadiness } from "./sourceReadiness";
import { presentSourceSyncActivity, type SourceSyncActivity } from "./sourceSyncActivity";
import type { Source } from "./types";

/** A scheduled sync this late counts as overdue. */
export const OVERDUE_GRACE_MS = 15 * 60_000;

export type SourceAttentionReason =
  | "local_sync_unavailable"
  | "sign_in_required"
  | "account_mismatch"
  | "configuration_required"
  | "file_limit_exceeded"
  | "sync_failed"
  | "partially_synced"
  | "overdue"
  | "unmapped";

/** What resolves the problem. The page decides how each action is carried out. */
export type SourceAttentionAction =
  | "start_local_sync"
  | "sign_in"
  | "configure"
  | "configure_scope"
  | "retry"
  | "view_details"
  | "sync_now"
  | "assign_project";

export interface SourceAttention {
  reason: SourceAttentionReason;
  title: string;
  detail: string;
  action: SourceAttentionAction;
  actionLabel: string;
}

interface SourceAttentionInput {
  source: Source;
  readiness: SourceReadiness | null;
  activity: SourceSyncActivity | undefined;
  now?: Date;
}

function isActive(activity: SourceSyncActivity | undefined): boolean {
  return activity !== undefined && ["queued", "active", "recovering"].includes(activity.state);
}

function isOverdue(source: Source, activity: SourceSyncActivity | undefined, now: Date): boolean {
  const schedule = source.sync_schedule;
  if (!schedule?.enabled || !schedule.next_run_at || isActive(activity)) return false;
  return now.getTime() - new Date(schedule.next_run_at).getTime() > OVERDUE_GRACE_MS;
}

/**
 * The one problem on a source the user should act on, most blocking first,
 * or null when the source needs nothing. Paused sources need nothing.
 */
export function sourceAttention({ source, readiness, activity, now = new Date() }: SourceAttentionInput): SourceAttention | null {
  if (source.status === "paused") return null;

  switch (readiness) {
    case "local_sync_unavailable":
      return {
        reason: "local_sync_unavailable",
        title: "Local sync is offline",
        detail: "This source syncs from your computer. Start local sync to continue.",
        action: "start_local_sync",
        actionLabel: "Copy start command",
      };
    case "sign_in_required":
      return {
        reason: "sign_in_required",
        title: "Sign-in required",
        detail: "The connection to this source expired. Sign in again to resume syncing.",
        action: "sign_in",
        actionLabel: "Sign in",
      };
    case "account_mismatch":
      return {
        reason: "account_mismatch",
        title: "Signed in with a different account",
        detail: "The account used to sign in does not match the one this source was set up with.",
        action: "sign_in",
        actionLabel: "Sign in again",
      };
    case "configuration_required":
      return {
        reason: "configuration_required",
        title: "Setup incomplete",
        detail: "Some required settings are missing.",
        action: "configure",
        actionLabel: "Configure",
      };
    default:
      break;
  }

  if (activity?.state === "failed") {
    const presentation = presentSourceSyncActivity(activity, source.name, "items");
    if (presentation.configureScope) {
      return {
        reason: "file_limit_exceeded",
        title: presentation.message,
        detail: presentation.detail,
        action: "configure_scope",
        actionLabel: "Configure file scope",
      };
    }
    return {
      reason: "sync_failed",
      title: "Last sync failed",
      detail: presentation.detail,
      action: "retry",
      actionLabel: "Retry",
    };
  }

  if (activity?.state === "partial") {
    return {
      reason: "partially_synced",
      title: "Some items failed",
      detail: "The last sync finished, but some items could not be read.",
      action: "view_details",
      actionLabel: "View details",
    };
  }

  if (isOverdue(source, activity, now)) {
    return {
      reason: "overdue",
      title: "Scheduled sync is overdue",
      detail: "The scheduled sync did not start on time.",
      action: "sync_now",
      actionLabel: "Sync now",
    };
  }

  if (!source.project_binding && !isManagedSourceType(source.type)) {
    return {
      reason: "unmapped",
      title: "No project",
      detail: "New memories from this source are not assigned to a project.",
      action: "assign_project",
      actionLabel: "Choose a project",
    };
  }

  return null;
}
