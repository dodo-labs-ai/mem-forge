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
  expect(resourceClient.get).toHaveBeenLastCalledWith("/sources/src-1/sync-runs/ssr-1");
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
