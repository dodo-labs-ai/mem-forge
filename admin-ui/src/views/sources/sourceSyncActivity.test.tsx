import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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

it.each(["discovery", "Internal network / VPN sync"])("shows bounded file-limit details and configuration action for %s", (mode) => {
  const onConfigureScope = vi.fn();
  const onRetry = vi.fn();
  render(<SourceSyncStatusCard activity={{ state: "failed", error: {
    message: `GitHub Repository ${mode} matched 571 files, exceeding max_files=500`,
  } }} sourceName="Cookbook" itemLabel="files" onConfigureScope={onConfigureScope} onRetry={onRetry} />);
  expect(screen.getByText("Sync scope exceeds file limit")).toBeTruthy();
  expect(screen.getByText(/Last sync matched 571 files, exceeding its configured limit of 500/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Configure file scope" }));
  expect(onConfigureScope).toHaveBeenCalledOnce();
  expect(onRetry).not.toHaveBeenCalled();
});

it("does not expose arbitrary raw errors or offer unavailable configuration", () => {
  render(<SourceSyncStatusCard activity={{ state: "failed", error: {
    message: "private credentials: secret",
  } }} sourceName="Cookbook" itemLabel="files" />);
  expect(screen.getByText("Sync failed. Retry when ready.")).toBeTruthy();
  expect(screen.queryByText(/private credentials/)).toBeNull();
  expect(screen.queryByRole("button", { name: "Configure file scope" })).toBeNull();
});
