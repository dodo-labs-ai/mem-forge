import { expect, test } from "vitest";
import { makeCoverage, makeIssueGroup, makeOverview, makeSourceHealth } from "@/test/evaluationFixtures";
import { buildEvaluationView } from "./evaluationView";
import {
  checkName,
  evaluationMetrics,
  issueGroupSummary,
  reasonDescription,
  shortId,
  sourceHealthSummary,
  windowLabel,
} from "./presentation";

const NOW = new Date("2026-09-29T12:00:00Z");
const TYPE_LABELS = { confluence: "Confluence", github_pages: "GitHub Pages" };

test("check names are in sentence case and unknown reasons read as words", () => {
  expect(checkName("evidence_reference_validity")).toBe("Evidence reference validity");
  expect(reasonDescription("schema_validation_failed")).toBe("The model output did not conform to the structured contract.");
  expect(reasonDescription("new_reason_code")).toBe("New reason code");
});

test("long IDs keep their head and tail", () => {
  expect(shortId("su_8f21a0c3d4e5f6a7b89b12e4")).toBe("su_8f21a…9b12e4");
  expect(shortId("rev-91d0")).toBe("rev-91d0");
});

test("window labels", () => {
  expect(windowLabel(1)).toBe("the last 24 hours");
  expect(windowLabel(7)).toBe("the last 7 days");
});

test("metrics name the problems and stay calm when there are none", () => {
  const overview = makeOverview();
  const metrics = evaluationMetrics(overview, buildEvaluationView(overview, { criterion: null, label: null }));
  expect(metrics).toEqual([
    { label: "Affected sources", value: "2 / 3", detail: "Sources with failures, patterns to check, or coverage gaps", tone: "danger" },
    { label: "Live quality", value: "2 issue groups", detail: "17 failed occurrences", tone: "danger" },
    { label: "Patterns to check", value: "9 occurrences", detail: "1 degraded pattern" },
    { label: "Evaluation coverage", value: "97%", detail: "1,168 / 1,204 eligible occurrences", tone: "warn" },
  ]);

  const healthy = makeOverview({
    issue_groups: [],
    coverage: makeCoverage({ assessed_occurrences: 1204, pending_occurrences: 0, coverage_rate: 1 }),
    summary: { ...overview.summary, affected_source_count: 0 },
  });
  const calm = evaluationMetrics(healthy, buildEvaluationView(healthy, { criterion: null, label: null }));
  expect(calm.map((metric) => [metric.value, metric.tone])).toEqual([
    ["0 / 3", undefined],
    ["Healthy", undefined],
    ["Clear", undefined],
    ["100%", undefined],
  ]);
});

test("an issue group reads as full sentences", () => {
  expect(issueGroupSummary(makeIssueGroup(), TYPE_LABELS, NOW)).toBe(
    "14 of 402 evidence reference validity checks (3.5%). Affects 2 sources (Confluence and GitHub Pages). Last seen 2 h ago.",
  );
});

test("a source's evaluation reads as full sentences", () => {
  expect(sourceHealthSummary(makeSourceHealth(), NOW)).toBe("2 issue groups, 0 patterns to check. Last event 2 h ago.");
  expect(
    sourceHealthSummary(
      makeSourceHealth({
        evaluation_status: "coverage_gap",
        action_issue_group_count: 0,
        coverage: makeCoverage({ pending_occurrences: 32, evaluator_failure_occurrences: 0 }),
      }),
      NOW,
    ),
  ).toBe("0 issue groups, 0 patterns to check. 32 occurrences waiting for evaluation. Last event 2 h ago.");
  expect(sourceHealthSummary(makeSourceHealth({ evaluation_status: "healthy" }), NOW)).toBe("No issues. Last event 2 h ago.");
  expect(sourceHealthSummary(makeSourceHealth({ evaluation_status: "no_data", last_event_at: null }), NOW)).toBe(
    "No events in this window.",
  );
});
