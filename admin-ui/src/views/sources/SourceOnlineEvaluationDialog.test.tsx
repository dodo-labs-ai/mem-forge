import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import type { Source, SourceAgentEvaluationResponse } from "@/api/types";
import { SourceOnlineEvaluationDialog } from "./SourceOnlineEvaluationDialog";

vi.mock("@/api/client", () => ({
  resourceClient: {
    get: vi.fn(),
  },
}));

const getMock = vi.mocked(resourceClient.get);

const source = {
  id: "src-teams",
  name: "Flexible Payroll Teams Chat",
  type: "teams",
  status: "active",
} as Source;

function responseFixture(): SourceAgentEvaluationResponse {
  const sharedCase = {
    assessment_id: "assessment-fail",
    event_id: "event-fail",
    label: "fail" as const,
    criterion: "evidence_reference_validity",
    reason_code: "legacy_quote_unresolved",
    occurred_at: "2026-08-19T04:00:00+00:00",
    occurrence_count: 1,
    source_id: "src-teams",
    source_type: "teams",
    doc_id: "doc-window-1",
    source_unit_id: "source-unit-1",
    target_unit_revision_id: "source-unit-revision-1",
    observation_id: null,
    observation_revision_id: null,
    projection_run_id: "projection-run-1",
    operation_id: "operation-1",
    execution_id: "execution-1",
    derivation_id: "derivation-1",
    batch_id: "batch-1",
    trace_id: "trace-1",
    provider: "openai",
    model: "gpt-5",
    contract_version: null,
    extraction_contract_version: "projection-extraction-v8",
    deployment_revision: "revision-cloud",
  };
  return {
    source_id: "src-teams",
    window: {
      from: "2026-08-18T05:00:00+00:00",
      to: "2026-08-19T05:00:00+00:00",
      days: 1,
    },
    summary: {
      total_assessments: 253,
      runtime_event_count: 281,
      eligible_assessment_count: 253,
      missing_assessment_count: 0,
      action_issue_group_count: 1,
      review_issue_group_count: 1,
      label_counts: { pass: 232, fail: 4, needs_review: 17 },
      criterion_counts: {},
      status_counts: { completed: 253 },
      truncated: false,
    },
    coverage: {
      policy: "semantic_evaluator_v1",
      eligible_occurrences: 253,
      assessed_occurrences: 253,
      pending_occurrences: 0,
      coverage_rate: 1,
      oldest_pending_at: null,
      evaluator_failure_occurrences: 0,
    },
    issue_groups: [
      {
        group_id: "group-fail",
        label: "fail",
        criterion: "evidence_reference_validity",
        reason_code: "legacy_quote_unresolved",
        evaluator_name: "memforge.deterministic.runtime_contract",
        evaluator_version: "1",
        occurrence_count: 4,
        distinct_event_count: 4,
        criterion_occurrence_count: 20,
        criterion_rate: 0.2,
        first_seen_at: "2026-08-19T03:00:00+00:00",
        last_seen_at: "2026-08-19T04:00:00+00:00",
        representative_cases: [sharedCase],
      },
      {
        group_id: "group-review",
        label: "needs_review",
        criterion: "evidence_localization",
        reason_code: "whole_block_fallback",
        evaluator_name: "memforge.deterministic.runtime_contract",
        evaluator_version: "1",
        occurrence_count: 17,
        distinct_event_count: 17,
        criterion_occurrence_count: 100,
        criterion_rate: 0.17,
        first_seen_at: "2026-08-19T02:00:00+00:00",
        last_seen_at: "2026-08-19T04:00:00+00:00",
        representative_cases: [
          {
            ...sharedCase,
            assessment_id: "assessment-review",
            event_id: "event-review",
            label: "needs_review",
            criterion: "evidence_localization",
            reason_code: "whole_block_fallback",
            observation_id: "observation-1",
            observation_revision_id: "observation-revision-1",
          },
        ],
      },
    ],
    runtime_events: [],
    assessments: [],
  };
}

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <SourceOnlineEvaluationDialog source={source} onOpenChange={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("SourceOnlineEvaluationDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getMock.mockResolvedValue({ data: responseFixture() });
  });

  afterEach(cleanup);

  it("defaults to actionable groups and hands one bounded case to an agent", async () => {
    const user = userEvent.setup();
    const clipboardWrite = vi.spyOn(navigator.clipboard, "writeText");
    renderDialog();

    expect(await screen.findByText("1 issue groups")).toBeTruthy();
    expect(screen.queryByText(/158 eligible runtime events/i)).toBeNull();
    expect(getMock).toHaveBeenCalledWith("/sources/src-teams/agent-evaluation", {
      params: { days: 1 },
    });

    await user.click(screen.getByRole("button", { name: /Evidence Reference Validity/ }));
    await user.click(screen.getByRole("button", { name: "Investigate with agent" }));

    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledOnce());
    const prompt = String(clipboardWrite.mock.calls[0][0]);
    expect(prompt).toContain("Runtime event: event-fail");
    expect(prompt).toContain("Source Unit: source-unit-1");
    expect(prompt).toContain("Do not rerun source ingestion");
    expect(await screen.findByRole("button", { name: "Prompt copied" })).toBeTruthy();
  });

  it("separates review and coverage from confirmed failures and supports a narrower window", async () => {
    const user = userEvent.setup();
    renderDialog();
    await screen.findByText("1 issue groups");

    await user.click(screen.getByRole("tab", { name: "Review queue (1)" }));
    expect(screen.getByRole("button", { name: /Evidence Localization/ })).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Coverage" }));
    expect(screen.getByText("Evaluation is keeping up")).toBeTruthy();
    expect(screen.getByText("253 of 253 eligible occurrences have a semantically matching durable assessment.")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "7d" }));
    await waitFor(() =>
      expect(getMock).toHaveBeenCalledWith("/sources/src-teams/agent-evaluation", {
        params: { days: 7 },
      }),
    );
  });
});
