import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import type { Source } from "@/api/types";
import { activeSyncRunOf, runningSyncOfRefusedSave, stopSyncAndWaitForRunEnd, useRunningSyncSave } from "./runningSyncSave";

vi.mock("@/api/client", () => ({ resourceClient: { get: vi.fn(), post: vi.fn() } }));

afterEach(() => vi.resetAllMocks());

const refusal = {
  response: {
    status: 409,
    data: { detail: { code: "source_sync_running", run_id: "ssr-1", message: "A sync of this Source is running." } },
  },
};

it("reads the running sync from a refused save, and nothing from other errors", () => {
  expect(runningSyncOfRefusedSave(refusal)).toBe("ssr-1");
  expect(runningSyncOfRefusedSave({ response: { status: 409, data: { detail: "source_provider_namespace_immutable" } } }))
    .toBeNull();
  expect(runningSyncOfRefusedSave(new Error("network"))).toBeNull();
});

it("stops the sync and waits until its run has ended", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopping" } });
  vi.mocked(resourceClient.get)
    .mockResolvedValueOnce({ data: { run_id: "ssr-1", status: "running" } })
    .mockResolvedValueOnce({ data: { run_id: "ssr-1", status: "failed" } });

  await stopSyncAndWaitForRunEnd("src-1", "ssr-1", { pollMs: 1 });

  expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-1/sync/stop");
  expect(resourceClient.get).toHaveBeenCalledTimes(2);
  expect(resourceClient.get).toHaveBeenLastCalledWith("/sources/src-1/sync-runs/ssr-1", { signal: undefined });
});

it("waits for the run the stop reached when another sync started meanwhile", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopping", run_id: "ssr-new" } });
  vi.mocked(resourceClient.get).mockResolvedValue({ data: { run_id: "ssr-new", status: "failed" } });

  await stopSyncAndWaitForRunEnd("src-1", "ssr-1", { pollMs: 1 });

  expect(resourceClient.get).toHaveBeenCalledWith("/sources/src-1/sync-runs/ssr-new", { signal: undefined });
  expect(resourceClient.get).not.toHaveBeenCalledWith("/sources/src-1/sync-runs/ssr-1", expect.anything());
});

it("gives up waiting with a message to save again later", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopping" } });
  vi.mocked(resourceClient.get).mockResolvedValue({ data: { run_id: "ssr-1", status: "running" } });

  await expect(stopSyncAndWaitForRunEnd("src-1", "ssr-1", { pollMs: 1, timeoutMs: 5 }))
    .rejects.toThrow("The sync has not stopped yet. Wait a moment, then save again.");
});

it("reads the sync in progress from a Source, and nothing once it has ended", () => {
  const sync = { run_id: "ssr-1", started_at: null, finished_at: null, error_message: null };
  expect(activeSyncRunOf({ sync: { ...sync, status: "running" } } as Source)).toBe("ssr-1");
  expect(activeSyncRunOf({ sync: { ...sync, status: "pending" } } as Source)).toBe("ssr-1");
  expect(activeSyncRunOf({ sync: { ...sync, status: "failed" } } as Source)).toBeNull();
  expect(activeSyncRunOf(null)).toBeNull();
});

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

it("asks before saving during a sync, and saves once the stopped sync has ended", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopping" } });
  vi.mocked(resourceClient.get).mockResolvedValue({ data: { run_id: "ssr-1", status: "failed" } });
  const save = vi.fn().mockResolvedValue(undefined);

  const { result } = renderHook(
    () => useRunningSyncSave<string>({ sourceId: "src-1", activeRunId: "ssr-1", saveError: null, save }),
    { wrapper },
  );
  expect(result.current.running).toBe(true);
  act(() => result.current.save("edits"));
  expect(result.current.confirming).toBe(true);
  expect(save).not.toHaveBeenCalled();

  act(() => result.current.stopAndSave());

  await waitFor(() => expect(save).toHaveBeenCalledWith("edits"));
  expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-1/sync/stop");
  expect(result.current.confirming).toBe(false);
});

it("asks when the server refuses a save because a sync started meanwhile", async () => {
  const save = vi.fn().mockRejectedValue(refusal);

  const { result } = renderHook(
    () => useRunningSyncSave<string>({ sourceId: "src-1", activeRunId: null, saveError: refusal, save }),
    { wrapper },
  );
  act(() => result.current.save("edits"));

  await waitFor(() => expect(result.current.confirming).toBe(true));
  expect(result.current.refusedForRunningSync).toBe(true);
  act(() => result.current.wait());
  expect(result.current.confirming).toBe(false);
  expect(resourceClient.post).not.toHaveBeenCalled();
});

/** A run read that answers only when the test says so, while the hook waits. */
function holdRunRead() {
  let answer: (status: string) => void = () => {};
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopping", run_id: "ssr-1" } });
  vi.mocked(resourceClient.get).mockImplementation(() => new Promise((resolve) => {
    answer = (status) => resolve({ data: { run_id: "ssr-1", status } });
  }));
  return {
    answer: (status: string) => answer(status),
    signal: () => (vi.mocked(resourceClient.get).mock.calls[0][1] as { signal: AbortSignal }).signal,
  };
}

it.each([
  ["the dialog closes", "unmount"],
  ["the person waits for the sync instead", "wait"],
])("saves nothing when %s while the sync is stopping", async (_, leave) => {
  const runRead = holdRunRead();
  const save = vi.fn().mockResolvedValue(undefined);
  const { result, unmount } = renderHook(
    () => useRunningSyncSave<string>({ sourceId: "src-1", activeRunId: "ssr-1", saveError: null, save }),
    { wrapper },
  );
  act(() => result.current.save("edits"));
  act(() => result.current.stopAndSave());
  await waitFor(() => expect(resourceClient.get).toHaveBeenCalledOnce());

  if (leave === "unmount") {
    unmount();
  } else {
    act(() => result.current.wait());
    expect(result.current.confirming).toBe(false);
    expect(result.current.stopping).toBe(false);
  }
  expect(runRead.signal().aborted).toBe(true);
  runRead.answer("failed");
  await new Promise((resolve) => setTimeout(resolve, 20));

  expect(save).not.toHaveBeenCalled();
  expect(resourceClient.get).toHaveBeenCalledOnce();
  expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-1/sync/stop");
});
