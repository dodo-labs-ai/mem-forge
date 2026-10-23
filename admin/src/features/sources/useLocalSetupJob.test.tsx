import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { ApiProvider, createApiClient, createWorkspaceController, type SetupJobOperation } from "@/api";
import { errorResponse, fakeFetch, type Route } from "@/test/fakeApi";
import { LocalSetupJobError, useLocalSetupJob } from "./useLocalSetupJob";

afterEach(() => {
  vi.unstubAllGlobals();
});

const CREATE = "POST /api/cloud/local-agent/jobs";
const READ = "GET /api/cloud/local-agent/jobs/laj-1";

function renderJob<Op extends SetupJobOperation>(operation: Op, routes: Record<string, Route>) {
  const fetch = fakeFetch({
    "GET /api/cloud/local-agent/status": () => ({ status: "online" }),
    [CREATE]: () => ({ job_id: "laj-1", status: "queued" }),
    ...routes,
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const workspace = createWorkspaceController(() => queryClient.clear());
  function wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ApiProvider api={createApiClient(workspace, "http://admin.test")} workspace={workspace}>
          {children}
        </ApiProvider>
      </QueryClientProvider>
    );
  }
  const hook = renderHook(() => useLocalSetupJob(operation), { wrapper });
  return { run: hook.result.current, ...fetch };
}

function requested(calls: Request[], key: string): number {
  return calls.filter((request) => `${request.method} ${new URL(request.url).pathname}` === key).length;
}

const BROWSED = {
  operation: "teams_browse",
  region: "emea",
  favorites: [],
  teams: [],
  group_chats: [{ id: "19:planning@thread.v2", topic: "Planning", lastActivity: null }],
  individual_chats: [],
  error: null,
  error_type: null,
  heartbeat_errors: null,
};

test("creates the job for its Source type and returns the result once it succeeds", async () => {
  const { run, calls } = renderJob("teams_browse", { [READ]: () => ({ job_id: "laj-1", status: "succeeded", result: BROWSED }) });

  await expect(run({ payload: { region: "emea" } })).resolves.toEqual(BROWSED);
  const created = calls.find((request) => request.method === "POST")!;
  expect(await created.clone().json()).toEqual({ source_id: "", source_type: "teams", operation: "teams_browse", payload: { region: "emea" } });
});

test("resumes the admitted job after a failed read", async () => {
  let reads = 0;
  const { run, calls } = renderJob("teams_browse", {
    [READ]: () => (++reads === 1 ? errorResponse(503, "Service unavailable") : { job_id: "laj-1", status: "succeeded", result: BROWSED }),
  });

  await expect(run({ payload: { region: "emea" } })).rejects.toThrow();
  await expect(run({ payload: { region: "emea" } })).resolves.toEqual(BROWSED);
  expect(requested(calls, CREATE)).toBe(1);
  expect(reads).toBe(2);
});

test("refuses a different request while the admitted job is still queued or leased", async () => {
  let reads = 0;
  const { run, calls } = renderJob("teams_browse", {
    [READ]: () => (++reads === 1 ? errorResponse(503, "Service unavailable") : { job_id: "laj-1", status: "leased", result: {} }),
  });

  await expect(run({ payload: { region: "emea" } })).rejects.toThrow();
  await expect(run({ payload: { region: "amer" } })).rejects.toThrow("Local sync is still working on the previous request.");
  expect(requested(calls, CREATE)).toBe(1);
});

test("admits a different request once the admitted job has finished", async () => {
  let created = 0;
  let reads = 0;
  const amer = { ...BROWSED, region: "amer" };
  const { run, calls } = renderJob("teams_browse", {
    [CREATE]: () => ({ job_id: `laj-${++created}`, status: "queued" }),
    [READ]: () => (++reads === 1 ? errorResponse(503, "Service unavailable") : { job_id: "laj-1", status: "succeeded", result: BROWSED }),
    "GET /api/cloud/local-agent/jobs/laj-2": () => ({ job_id: "laj-2", status: "succeeded", result: amer }),
  });

  await expect(run({ payload: { region: "emea" } })).rejects.toThrow();
  await expect(run({ payload: { region: "amer" } })).resolves.toEqual(amer);
  expect(requested(calls, CREATE)).toBe(2);
  expect(reads).toBe(2);
});

test("a failed job result is reported, not treated as success", async () => {
  const { run } = renderJob("teams_auth_check", {
    [READ]: () => ({
      job_id: "laj-1",
      status: "failed",
      last_error: null,
      result: { operation: "teams_auth_check", region: "emea", authenticated: false, expires_in_minutes: null, error: "No Teams session found" },
    }),
  });

  const failure = await run({ payload: { region: "emea" } }).catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(LocalSetupJobError);
  expect((failure as LocalSetupJobError<"teams_auth_check">).message).toBe("No Teams session found");
  expect((failure as LocalSetupJobError<"teams_auth_check">).result.authenticated).toBe(false);
});

test("a failed job without a result error reports the job's last error", async () => {
  const { run } = renderJob("local_markdown_pick_root", {
    [READ]: () => ({ job_id: "laj-1", status: "failed", last_error: "No folder picker on this computer", result: { retryable: false } }),
  });

  await expect(run({ payload: {} })).rejects.toThrow("No folder picker on this computer");
});

test("asks for local sync to be started instead of enqueuing a job it would never run", async () => {
  const { run, calls } = renderJob("teams_browse", { "GET /api/cloud/local-agent/status": () => ({ status: "offline" }) });

  await expect(run({ payload: { region: "emea" } })).rejects.toThrow("Start local sync on your computer, then try again.");
  expect(requested(calls, CREATE)).toBe(0);
});
