import { describe, expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { OVERDUE_GRACE_MS, sourceAttention } from "./sourceAttention";
import { sourceSyncActivityFromStatus } from "./sourceSyncActivity";

const NOW = new Date("2026-09-29T12:00:00Z");

describe("sourceAttention", () => {
  test("a healthy source needs nothing", () => {
    const source = makeSource();
    expect(sourceAttention({ source, readiness: null, activity: sourceSyncActivityFromStatus(source.sync!), now: NOW })).toBeNull();
  });

  test("a paused source needs nothing, even when it failed", () => {
    const source = makeSource({ status: "paused" });
    expect(sourceAttention({ source, readiness: "sign_in_required", activity: { state: "failed" }, now: NOW })).toBeNull();
  });

  test("readiness problems come before sync failures", () => {
    const attention = sourceAttention({ source: makeSource(), readiness: "local_sync_unavailable", activity: { state: "failed" }, now: NOW });
    expect(attention?.reason).toBe("local_sync_unavailable");
    expect(attention?.action).toBe("start_local_sync");
  });

  test("an exceeded file limit asks to narrow the scope", () => {
    const attention = sourceAttention({
      source: makeSource({ type: "github_repo" }),
      readiness: null,
      activity: {
        state: "failed",
        error: { message: "GitHub Repository discovery matched 12000 files, exceeding max_files=5000" },
      },
      now: NOW,
    });
    expect(attention).toMatchObject({ reason: "file_limit_exceeded", action: "configure_scope" });
  });

  test("other failures offer a retry", () => {
    const attention = sourceAttention({ source: makeSource(), readiness: null, activity: { state: "failed", error: { message: "boom" } }, now: NOW });
    expect(attention).toMatchObject({ reason: "sync_failed", action: "retry" });
  });

  test("a scheduled run later than the grace period is overdue", () => {
    const late = new Date(NOW.getTime() - OVERDUE_GRACE_MS - 1).toISOString();
    const onTime = new Date(NOW.getTime() - OVERDUE_GRACE_MS + 1).toISOString();
    const schedule = (next_run_at: string) => ({ enabled: true, interval_minutes: 60, next_run_at, updated_at: null });
    expect(sourceAttention({ source: makeSource({ sync_schedule: schedule(late) }), readiness: null, activity: undefined, now: NOW })?.reason).toBe("overdue");
    expect(sourceAttention({ source: makeSource({ sync_schedule: schedule(onTime) }), readiness: null, activity: undefined, now: NOW })).toBeNull();
    expect(
      sourceAttention({ source: makeSource({ sync_schedule: schedule(late) }), readiness: null, activity: { state: "active" }, now: NOW }),
    ).toBeNull();
  });

  test("a source without a project asks for one, except plugin sources", () => {
    expect(sourceAttention({ source: makeSource({ project_binding: null }), readiness: null, activity: undefined, now: NOW })?.reason).toBe("unmapped");
    expect(
      sourceAttention({ source: makeSource({ type: "agent_session", project_binding: null }), readiness: null, activity: undefined, now: NOW }),
    ).toBeNull();
  });
});
