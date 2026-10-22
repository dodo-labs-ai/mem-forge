import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import { runningSyncOfRefusedSave, stopSyncAndWaitForRunEnd, useStopRunningSyncAndSave } from "./runningSyncSave";

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

it("saves again once the stopped sync has ended", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { outcome: "stopped" } });
  vi.mocked(resourceClient.get).mockResolvedValue({ data: { run_id: "ssr-1", status: "failed" } });
  const saveAgain = vi.fn();
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );

  const { result } = renderHook(() => useStopRunningSyncAndSave("src-1", refusal, saveAgain), { wrapper });
  expect(result.current.runId).toBe("ssr-1");
  act(() => result.current.run());

  await waitFor(() => expect(saveAgain).toHaveBeenCalledOnce());
  expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-1/sync/stop");
});
