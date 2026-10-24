import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { ApiError, ApiProvider, createApiClient, createWorkspaceController } from "@/api";
import { fakeFetch, type Route } from "@/test/fakeApi";
import { useRunningSyncSave } from "./useRunningSyncSave";

afterEach(() => {
  vi.unstubAllGlobals();
});

const STOP = "POST /api/v1/sources/src-1/sync/stop";
const RUN = "GET /api/v1/sources/src-1/sync-runs/ssr-1";

function renderSave(routes: Record<string, Route>, save: (payload: string) => Promise<unknown>, activeRunId: string | null = "ssr-1") {
  const fetch = fakeFetch({
    [STOP]: () => ({ source_id: "src-1", outcome: "stopping", run_id: "ssr-1", local_agent_job_ids: [] }),
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
  const hook = renderHook(() => useRunningSyncSave<string>({ sourceId: "src-1", activeRunId, save }), { wrapper });
  return { ...hook, ...fetch };
}

function requested(calls: Request[], key: string): number {
  return calls.filter((request) => `${request.method} ${new URL(request.url).pathname}` === key).length;
}

test("asks before saving during a sync, and saves once the stopped sync has ended", async () => {
  const save = vi.fn().mockResolvedValue(undefined);
  const { result, calls } = renderSave({ [RUN]: () => ({ run_id: "ssr-1", status: "failed" }) }, save);

  expect(result.current.running).toBe(true);
  act(() => result.current.save("edits"));
  expect(result.current.confirming).toBe(true);
  expect(save).not.toHaveBeenCalled();

  act(() => result.current.stopAndSave());

  await waitFor(() => expect(save).toHaveBeenCalledWith("edits"));
  expect(requested(calls, STOP)).toBe(1);
  expect(result.current.confirming).toBe(false);
});

test("asks when the server refuses a save because a sync started meanwhile", async () => {
  const refusal = new ApiError(409, "A sync of this Source is running.", "source_sync_running", { run_id: "ssr-1" });
  const save = vi.fn().mockRejectedValue(refusal);
  const { result, calls } = renderSave({ "GET /api/v1/sources": () => ({ data: [] }) }, save, null);

  act(() => result.current.save("edits"));

  await waitFor(() => expect(result.current.confirming).toBe(true));
  act(() => result.current.wait());
  expect(result.current.confirming).toBe(false);
  expect(requested(calls, STOP)).toBe(0);
});

test.each([
  ["the page closes", "unmount"],
  ["the person waits for the sync instead", "wait"],
])("saves nothing when %s while the sync is stopping", async (_, leave) => {
  let answer: (status: string) => void = () => {};
  const save = vi.fn().mockResolvedValue(undefined);
  const { result, unmount, calls } = renderSave(
    {
      [RUN]: () =>
        new Promise((resolve) => {
          answer = (status) => resolve({ run_id: "ssr-1", status });
        }),
    },
    save,
  );
  act(() => result.current.save("edits"));
  act(() => result.current.stopAndSave());
  await waitFor(() => expect(requested(calls, RUN)).toBe(1));
  const runRead = calls.find((request) => request.method === "GET")!;

  if (leave === "unmount") {
    unmount();
  } else {
    act(() => result.current.wait());
    expect(result.current.confirming).toBe(false);
    expect(result.current.stopping).toBe(false);
  }
  expect(runRead.signal.aborted).toBe(true);
  answer("failed");
  await new Promise((resolve) => setTimeout(resolve, 20));

  expect(save).not.toHaveBeenCalled();
  expect(requested(calls, RUN)).toBe(1);
  expect(requested(calls, STOP)).toBe(1);
});
