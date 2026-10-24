import { expect, test, vi } from "vitest";
import { ApiError } from "@/api";
import { makeSource } from "@/test/sourceFixtures";
import {
  activeSyncRunOf,
  RUN_END_TIMEOUT_MESSAGE,
  runningSyncOfRefusedSave,
  stopSyncAndWaitForRunEnd,
  type SyncStopCalls,
} from "./runningSyncSave";

function calls(stopRunId: string | null, statuses: string[]) {
  const reads = [...statuses];
  return {
    stop: vi.fn<SyncStopCalls["stop"]>(async () => ({ run_id: stopRunId })),
    readRun: vi.fn<SyncStopCalls["readRun"]>(async () => ({ status: reads.length > 1 ? reads.shift()! : reads[0]! })),
  };
}

test("reads the running sync from a refused save, and nothing from other errors", () => {
  const refusal = new ApiError(409, "A sync of this Source is running.", "source_sync_running", { run_id: "ssr-1" });
  expect(runningSyncOfRefusedSave(refusal)).toBe("ssr-1");
  expect(runningSyncOfRefusedSave(new ApiError(409, "source_provider_namespace_immutable"))).toBeNull();
  expect(runningSyncOfRefusedSave(new ApiError(400, "Invalid", "config_invalid", { fields: [] }))).toBeNull();
  expect(runningSyncOfRefusedSave(new Error("network"))).toBeNull();
});

test("stops the sync and waits until its run has ended", async () => {
  const server = calls(null, ["running", "failed"]);

  await stopSyncAndWaitForRunEnd(server, "ssr-1", { pollMs: 1 });

  expect(server.stop).toHaveBeenCalledOnce();
  expect(server.readRun).toHaveBeenCalledTimes(2);
  expect(server.readRun).toHaveBeenLastCalledWith("ssr-1", undefined);
});

test("waits for the run the stop reached when another sync started meanwhile", async () => {
  const server = calls("ssr-new", ["failed"]);

  await stopSyncAndWaitForRunEnd(server, "ssr-1", { pollMs: 1 });

  expect(server.readRun).toHaveBeenCalledWith("ssr-new", undefined);
  expect(server.readRun).not.toHaveBeenCalledWith("ssr-1", expect.anything());
});

test("gives up waiting with a message to save again later", async () => {
  const server = calls(null, ["running"]);

  await expect(stopSyncAndWaitForRunEnd(server, "ssr-1", { pollMs: 1, timeoutMs: 5 })).rejects.toThrow(RUN_END_TIMEOUT_MESSAGE);
});

test("an aborted wait stops reading the run, not the sync", async () => {
  const server = calls(null, ["running"]);
  const controller = new AbortController();
  const waiting = stopSyncAndWaitForRunEnd(server, "ssr-1", { pollMs: 1_000, signal: controller.signal });
  await vi.waitFor(() => expect(server.readRun).toHaveBeenCalledOnce());

  controller.abort(new Error("left"));

  await expect(waiting).rejects.toThrow("left");
  expect(server.readRun).toHaveBeenCalledOnce();
});

test("reads the sync in progress from a Source, and nothing once it has ended", () => {
  const sync = { run_id: "ssr-1", started_at: null, finished_at: null, error_message: null, progress: null };
  expect(activeSyncRunOf(makeSource({ sync: { ...sync, status: "running" } }))).toBe("ssr-1");
  expect(activeSyncRunOf(makeSource({ sync: { ...sync, status: "pending" } }))).toBe("ssr-1");
  expect(activeSyncRunOf(makeSource({ sync: { ...sync, status: "failed" } }))).toBeNull();
  expect(activeSyncRunOf(undefined)).toBeNull();
});
