import { expect, test } from "vitest";
import { readEvaluationParams, writeEvaluationParams } from "./evaluationParams";

test("reads the URL and falls back to the defaults for unknown values", () => {
  expect(readEvaluationParams(new URLSearchParams("days=7&source_id=src-1&criterion=evidence_localization&label=fail"))).toEqual({
    days: 7,
    sourceId: "src-1",
    sourceType: null,
    criterion: "evidence_localization",
    label: "fail",
  });
  expect(readEvaluationParams(new URLSearchParams("days=5&label=maybe"))).toMatchObject({ days: 1, label: null });
});

test("changing the source scope clears the check filters and keeps other parameters", () => {
  const current = new URLSearchParams("workspace=mount_tai&source_id=src-1&criterion=evidence_localization&label=fail");
  const next = writeEvaluationParams(current, { sourceId: null, sourceType: "jira" });
  expect(Object.fromEntries(next)).toEqual({ workspace: "mount_tai", source_type: "jira" });
});

test("changing a check filter keeps the scope, and the default window is left out", () => {
  const current = new URLSearchParams("source_id=src-1&days=7");
  expect(Object.fromEntries(writeEvaluationParams(current, { label: "needs_review" }))).toEqual({
    source_id: "src-1",
    days: "7",
    label: "needs_review",
  });
  expect(Object.fromEntries(writeEvaluationParams(current, { days: 1 }))).toEqual({ source_id: "src-1" });
});
