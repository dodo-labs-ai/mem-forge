import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useApi } from "@/api";
import { LOCAL_SYNC_START_COMMAND } from "@/features/local-sync";
import { errorMessage } from "@/lib/errors";
import { sourceSettingsPath } from "@/lib/paths";
import { configSchemaQuery, useSetPaused, useSetPinned, useSetUsedInSearches, useStopSync, useSyncSource } from "./api";
import { firstGroupWithPurpose } from "./model/configGroups";
import type { SourceAttentionAction } from "./model/sourceAttention";
import {
  sourceSyncCanStop,
  sourceSyncControl,
  type SourceSyncActivity,
  type SourceSyncRetryTarget,
} from "./model/sourceSyncActivity";
import type { Source } from "./model/types";
import { useJiraSignIn } from "./useJiraSignIn";
import { useTeamsSignIn } from "./useTeamsSignIn";

/** Runs row and drawer actions, reporting the outcome in a toast. */
export function useSourceActions({ onViewDetails }: { onViewDetails: (source: Source) => void }) {
  const sync = useSyncSource();
  const setPinned = useSetPinned();
  const setUsedInSearches = useSetUsedInSearches();
  const setPaused = useSetPaused();
  const stopSync = useStopSync();
  const jiraSignIn = useJiraSignIn();
  const teamsSignIn = useTeamsSignIn();
  const navigate = useNavigate();
  const api = useApi();
  const queryClient = useQueryClient();

  /** Opens the settings step of the type's first content group, or the settings when its form cannot be read. */
  function openContentStep(source: Source) {
    queryClient.fetchQuery(configSchemaQuery(api, source.type)).then(
      (schema) => navigate(sourceSettingsPath(source.id, firstGroupWithPurpose(schema, "content") ?? undefined)),
      () => navigate(sourceSettingsPath(source.id)),
    );
  }

  function report(promise: Promise<unknown>, success: string, failure: string) {
    promise.then(
      () => toast.success(success),
      (error: unknown) => toast.error(failure, { description: errorMessage(error) }),
    );
  }

  /**
   * How a source's sign-in is renewed: in place through local sync for a Jira
   * browser session or Teams, and otherwise on the connection step, where
   * a token is entered. Jira's `auth_mode` is always in its configuration.
   */
  function signInRenewal(source: Source): typeof jiraSignIn | typeof teamsSignIn | null {
    if (source.type === "teams") return teamsSignIn;
    if (source.type === "jira" && source.config.auth_mode === "browser_cookie") return jiraSignIn;
    return null;
  }

  return {
    attentionControl(action: SourceAttentionAction, source: Source, label: string) {
      if (action !== "sign_in") return { label, enabled: true };
      const renewal = signInRenewal(source);
      return {
        label: renewal?.isPending && renewal.variables?.id === source.id ? "Waiting for sign-in…" : label,
        enabled: Boolean(source.capabilities?.can_configure_connection) && !renewal?.isPending,
      };
    },
    /** The sync button for a source, which reads Starting while this page's request for it is on its way. */
    syncControl(source: Source, activity: SourceSyncActivity | undefined) {
      return sourceSyncControl(activity, sync.isPending && sync.variables?.source.id === source.id);
    },
    syncNow(source: Source, retryTarget?: SourceSyncRetryTarget) {
      report(
        sync.mutateAsync({ source, retryTarget }),
        retryTarget ? `Retry started for ${source.name}` : `Sync started for ${source.name}`,
        retryTarget ? `Could not start the retry for ${source.name}` : `Could not start a sync for ${source.name}`,
      );
    },
    /** Whether Full sync is offered for the source, and whether it can start now: an active source with no sync on its way. */
    fullSyncControl(source: Source, activity: SourceSyncActivity | undefined) {
      const visible = Boolean(source.capabilities?.can_force_resync);
      return { visible, enabled: visible && source.status === "active" && this.syncControl(source, activity).enabled };
    },
    /** Whether the source's sync can be stopped from here, by someone who can sync or pause it. */
    canStopSync(source: Source, activity: SourceSyncActivity | undefined) {
      const capabilities = source.capabilities;
      return Boolean(capabilities?.can_sync || capabilities?.can_configure) && sourceSyncCanStop(activity)
        && !(stopSync.isPending && stopSync.variables === source.id);
    },
    stopSync(source: Source) {
      stopSync.mutateAsync(source.id).then(
        (stopped) =>
          toast.success(
            stopped.outcome === "stopping"
              ? `Stopping the sync of ${source.name}`
              : stopped.outcome === "stopped"
                ? `Sync of ${source.name} stopped`
                : `${source.name} was not syncing`,
          ),
        (error: unknown) => toast.error(`Could not stop the sync of ${source.name}`, { description: errorMessage(error) }),
      );
    },
    togglePinned(source: Source) {
      const pinned = !source.pinned_for_me;
      report(
        setPinned.mutateAsync({ sourceId: source.id, pinned }),
        pinned ? `${source.name} pinned` : `${source.name} unpinned`,
        "Could not change the pin",
      );
    },
    toggleUsedInSearches(source: Source) {
      const enabled = !(source.enabled_for_me ?? true);
      report(
        setUsedInSearches.mutateAsync({ sourceId: source.id, enabled }),
        enabled ? `${source.name} is used in your searches` : `${source.name} is no longer used in your searches`,
        "Could not change search use",
      );
    },
    togglePaused(source: Source) {
      const paused = source.status !== "paused";
      report(
        setPaused.mutateAsync({ sourceId: source.id, paused }),
        paused ? `${source.name} paused` : `${source.name} resumed`,
        paused ? "Could not pause the source" : "Could not resume the source",
      );
    },
    runAttention(action: SourceAttentionAction, source: Source) {
      switch (action) {
        case "sync_now":
        case "retry":
          this.syncNow(source);
          return;
        case "start_local_sync":
          report(navigator.clipboard.writeText(LOCAL_SYNC_START_COMMAND), "Start command copied", "Could not copy");
          return;
        case "view_details":
          onViewDetails(source);
          return;
        case "sign_in": {
          const renewal = signInRenewal(source);
          if (renewal === teamsSignIn) {
            report(teamsSignIn.mutateAsync(source), `Signed in to Teams for ${source.name}`, "Teams sign-in failed");
          } else if (renewal === jiraSignIn) {
            report(jiraSignIn.mutateAsync(source), `Signed in to ${source.name}`, "Jira sign-in failed");
          } else {
            navigate(sourceSettingsPath(source.id, "connection"));
          }
          return;
        }
        case "configure":
          navigate(sourceSettingsPath(source.id, "connection"));
          return;
        case "configure_scope":
          openContentStep(source);
          return;
        case "assign_project":
          navigate(sourceSettingsPath(source.id, "project"));
          return;
      }
    },
  };
}

export type SourceActions = ReturnType<typeof useSourceActions>;
