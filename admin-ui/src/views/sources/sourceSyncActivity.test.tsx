import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SourceSyncStatusCard } from "@/components/admin/SourceSyncStatusCard";
import { sourceSyncActivityFromLocalJob, presentSourceSyncActivity, sourceSyncActivityPolicy } from "./sourceSyncActivity";

afterEach(() => { cleanup(); vi.useRealTimers(); });

it("shows a static retry date and a usable exact-job retry while queued", () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-09-03T00:00:00Z"));
  const activity = sourceSyncActivityFromLocalJob({
    job_id: "laj-1", status: "queued", result: null, last_error: "VPN unavailable",
    next_attempt_at: "2026-09-03T12:00:00Z",
  });
  render(<SourceSyncStatusCard activity={activity} sourceName="GitHub" itemLabel="files" onRetry={() => {}} />);
  expect(screen.getByText("Waiting to retry")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Retry now" })).toBeTruthy();
  expect(screen.queryByRole("progressbar")).toBeNull();
  expect(document.querySelector(".animate-spin")).toBeNull();
  expect(sourceSyncActivityPolicy(activity).activeRowLabel).toBe("Waiting to retry");
  expect(activity.retryTarget).toEqual({ execution_kind: "local_agent_job", execution_id: "laj-1" });
});

it("shows eligible but unclaimed work as waiting for device, not syncing", () => {
  const activity = sourceSyncActivityFromLocalJob({
    job_id: "laj-1", status: "queued", result: null, last_error: null,
    next_attempt_at: "2020-01-01T00:00:00Z",
  });
  expect(presentSourceSyncActivity(activity, "GitHub", "files").message).toBe("Waiting for your device");
  render(<SourceSyncStatusCard activity={activity} sourceName="GitHub" itemLabel="files" />);
  expect(screen.queryByRole("progressbar")).toBeNull();
});
