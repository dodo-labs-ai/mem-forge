import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef } from "react";
import { HTTP_STATUS, isApiErrorStatus, unwrap, useApi, type SetupJobOperation, type SetupJobResults } from "@/api";
import { localSyncStatusQuery } from "@/features/local-sync";
import { ACTIVE_POLL_MS } from "./constants";

/** The Source type each setup operation belongs to, as the server's job route checks it. */
const SETUP_JOB_SOURCE_TYPES: Readonly<Record<SetupJobOperation, string>> = {
  github_repo_preview_tree: "github_repo",
  jira_auth: "jira",
  local_markdown_pick_root: "local_markdown",
  local_markdown_preview_tree: "local_markdown",
  teams_auth: "teams",
  teams_auth_check: "teams",
  teams_browse: "teams",
};

/** The job states a daemon does not change again. */
const FINISHED: ReadonlySet<string> = new Set(["succeeded", "failed"]);

const MINUTE_MS = 60_000;

/**
 * How long the page waits for each operation. A Jira sign-in or the folder
 * picker waits for the person at their computer for up to an hour; a Teams
 * sign-in gives up on its own after a few minutes, and listing conversations
 * may run one first; the rest only read.
 */
const SETUP_JOB_TIMEOUT_MS: Readonly<Record<SetupJobOperation, number>> = {
  github_repo_preview_tree: 3 * MINUTE_MS,
  jira_auth: 60 * MINUTE_MS,
  local_markdown_pick_root: 60 * MINUTE_MS,
  local_markdown_preview_tree: 3 * MINUTE_MS,
  teams_auth: 10 * MINUTE_MS,
  teams_auth_check: 3 * MINUTE_MS,
  teams_browse: 10 * MINUTE_MS,
};

/** What a setup job is asked to do: the edited Source, when there is one, and the operation's payload. */
export interface LocalSetupJobRequest {
  sourceId?: string;
  payload: Record<string, unknown>;
}

/** The sentences a caller can word for its own step; each has a general default. */
export interface LocalSetupJobMessages {
  /** Local sync is not running on the person's computer. */
  offline: string;
  /** The page waited longer than the operation's time limit. */
  timeout: string;
  /** The person may not use local sync in this workspace. */
  forbidden: string;
  /** The job of another request of this page is still queued or leased, so a new one would enqueue a second job. */
  busy: string;
  /** The job failed without saying why. */
  failed: string;
}

const NO_OPTIONS: LocalSetupJobOptions = {};

const DEFAULT_MESSAGES: LocalSetupJobMessages = {
  offline: "Start local sync on your computer, then try again.",
  timeout: "Local sync did not finish in time. Check local sync on your computer, then try again.",
  forbidden: "You do not have permission to use local sync in this workspace. Ask a workspace admin for help.",
  busy: "Local sync is still working on the previous request. Try again once it finishes.",
  failed: "Local sync could not finish this step. Try again.",
};

/** A caller's wording, declared once at module level so the returned function keeps its identity. */
export interface LocalSetupJobOptions {
  messages?: Partial<LocalSetupJobMessages>;
  /** Words a failure the daemon reported, such as Jira's `principal_changed`; the daemon's own words otherwise. */
  describeFailure?: (detail: string) => string | undefined;
}

/**
 * A setup job that completed as failed. `message` says why, and `result`
 * holds what the daemon reported with the failure, such as a Teams session
 * check's `authenticated: false`.
 */
export class LocalSetupJobError<Op extends SetupJobOperation = SetupJobOperation> extends Error {
  readonly result: Partial<SetupJobResults[Op]>;

  constructor(message: string, result: Partial<SetupJobResults[Op]>) {
    super(message);
    this.name = "LocalSetupJobError";
    this.result = result;
  }
}

/**
 * Runs one setup operation on the person's local sync daemon, such as
 * listing Teams conversations or opening the folder picker, and returns its
 * result typed by the operation.
 *
 * - The job is created through the selected workspace's local-agent route
 *   and read through the host job route by its id.
 * - The admitted job id is kept while the page is open, so when reading the
 *   job fails, running the same request again reads the same job instead of
 *   enqueuing another. A different request reads the admitted job once: it is
 *   refused while that job is still queued or leased, and admitted once that
 *   job has finished or is gone.
 * - Success means the job completed durably. A job that completes as failed
 *   throws a `LocalSetupJobError`; it is never treated as success.
 *
 * Example: `const browse = useLocalSetupJob("teams_browse")`, then
 * `await browse({ payload: { region: "emea" } })` returns the conversations.
 */
export function useLocalSetupJob<Op extends SetupJobOperation>(operation: Op, options: LocalSetupJobOptions = NO_OPTIONS) {
  const api = useApi();
  const queryClient = useQueryClient();
  const admitted = useRef<{ jobId: string; request: string } | null>(null);

  return useCallback(
    async ({ sourceId, payload }: LocalSetupJobRequest): Promise<SetupJobResults[Op]> => {
      const messages = { ...DEFAULT_MESSAGES, ...options.messages };
      const request = JSON.stringify([sourceId ?? "", payload]);
      try {
        if (admitted.current && admitted.current.request !== request) {
          const previous = await unwrap(
            api.GET("/api/cloud/local-agent/jobs/{job_id}", { params: { path: { job_id: admitted.current.jobId } } }),
          ).catch((error: unknown) => {
            if (isApiErrorStatus(error, HTTP_STATUS.notFound)) return null;
            throw error;
          });
          if (previous && !FINISHED.has(previous.status)) throw new Error(messages.busy);
          admitted.current = null;
        }
        if (!admitted.current) {
          const daemon = await queryClient.fetchQuery(localSyncStatusQuery(api)).catch(() => null);
          if (daemon?.status !== "online") throw new Error(messages.offline);
          const created = await unwrap(
            api.POST("/api/cloud/local-agent/jobs", {
              body: { source_id: sourceId ?? "", source_type: SETUP_JOB_SOURCE_TYPES[operation], operation, payload },
            }),
          );
          admitted.current = { jobId: created.job_id, request };
        }
        const { jobId } = admitted.current;
        const deadline = Date.now() + SETUP_JOB_TIMEOUT_MS[operation];
        while (Date.now() < deadline) {
          const job = await unwrap(
            api.GET("/api/cloud/local-agent/jobs/{job_id}", {
              params: { path: { job_id: jobId } },
              signal: AbortSignal.timeout(Math.max(1, deadline - Date.now())),
            }),
          );
          if (FINISHED.has(job.status)) admitted.current = null;
          if (job.status === "succeeded") return job.result as SetupJobResults[Op];
          if (job.status === "failed") {
            const result = job.result as Partial<SetupJobResults[Op]>;
            const detail = (typeof job.result.error === "string" ? job.result.error.trim() : "") || job.last_error?.trim() || "";
            const message = (detail && options.describeFailure?.(detail)) || detail || messages.failed;
            throw new LocalSetupJobError<Op>(message, result);
          }
          await new Promise((resolve) => window.setTimeout(resolve, Math.min(ACTIVE_POLL_MS, Math.max(0, deadline - Date.now()))));
        }
        throw new Error(messages.timeout);
      } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") throw new Error(messages.timeout);
        if (isApiErrorStatus(error, HTTP_STATUS.forbidden)) throw new Error(messages.forbidden);
        throw error;
      }
    },
    [api, queryClient, operation, options],
  );
}
