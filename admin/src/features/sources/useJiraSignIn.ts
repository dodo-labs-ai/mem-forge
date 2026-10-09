import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";
import { HTTP_STATUS, isApiErrorStatus, unwrap, useApi } from "@/api";
import { localSyncKeys, useLocalSyncStatus } from "@/features/local-sync";
import { ACTIVE_POLL_MS } from "./constants";
import { sourceKeys } from "./api";
import type { Source } from "./model/types";

// Interactive authentication can wait up to one hour for the local browser.
const SIGN_IN_TIMEOUT_MS = 60 * 60_000;
const SIGN_IN_TIMEOUT_MESSAGE = "Still waiting for Jira sign-in. Check the browser on your computer, then try again.";

/** Renews a Jira browser session through the user's existing local daemon. */
export function useJiraSignIn() {
  const api = useApi();
  const queryClient = useQueryClient();
  const daemon = useLocalSyncStatus();
  const admitted = useRef<{ jobId: string; sourceId: string } | null>(null);
  return useMutation({
    mutationFn: async (source: Source) => {
      try {
        if (admitted.current && admitted.current.sourceId !== source.id) {
          throw new Error("Check the previous source's sign-in before starting another connection.");
        }
        if (!admitted.current) {
          const baseUrl = typeof source.config.base_url === "string" ? source.config.base_url.trim() : "";
          if (!baseUrl) throw new Error("The Jira URL is missing. Configure this source before signing in.");
          if (daemon.isPending) throw new Error("Checking local sync. Try signing in again in a moment.");
          if (daemon.data?.status !== "online") throw new Error("Start local sync on your computer, then try signing in again.");
          const created = await unwrap(api.POST("/api/cloud/local-agent/jobs", {
            body: {
              source_id: source.id,
              source_type: "jira",
              operation: "jira_auth",
              payload: { base_url: baseUrl, auth_mode: "browser_cookie" },
            },
          }));
          admitted.current = { jobId: created.job_id, sourceId: source.id };
        }
        const jobId = admitted.current.jobId;
        const deadline = Date.now() + SIGN_IN_TIMEOUT_MS;
        while (Date.now() < deadline) {
          const job = await unwrap(api.GET("/api/cloud/local-agent/jobs/{job_id}", {
            params: { path: { job_id: jobId } },
            signal: AbortSignal.timeout(Math.max(1, deadline - Date.now())),
          }));
          if (job.status === "succeeded" || job.status === "failed") admitted.current = null;
          if (job.status === "succeeded") return job;
          if (job.status === "failed") {
            const detail = typeof job.result.error === "string" ? job.result.error.trim() : job.last_error?.trim();
            throw new Error(detail === "principal_changed"
              ? "Signed in with a different Jira account. Use the account this source was set up with, then try again."
              : detail || "The local sign-in did not complete. Try signing in again.");
          }
          await new Promise((resolve) => window.setTimeout(resolve, Math.min(ACTIVE_POLL_MS, Math.max(0, deadline - Date.now()))));
        }
        throw new Error(SIGN_IN_TIMEOUT_MESSAGE);
      } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") throw new Error(SIGN_IN_TIMEOUT_MESSAGE);
        if (isApiErrorStatus(error, HTTP_STATUS.forbidden)) {
          throw new Error("You do not have permission to renew this connection. Ask a workspace admin for help.");
        }
        throw error;
      }
    },
    onSettled: () => Promise.all([
      queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
      queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
    ]),
  });
}
