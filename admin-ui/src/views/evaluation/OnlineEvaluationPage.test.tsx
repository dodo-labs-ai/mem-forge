import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { resourceClient } from "@/api/client";
import type { WorkspaceAgentEvaluationResponse } from "@/api/types";
import { OnlineEvaluationPage } from "./OnlineEvaluationPage";

vi.mock("@/api/client", () => ({
  resourceClient: { get: vi.fn() },
}));

const getMock = vi.mocked(resourceClient.get);

function responseFixture(): WorkspaceAgentEvaluationResponse {
  const coverage = {
    policy: "semantic_evaluator_v1" as const,
    eligible_occurrences: 3,
    assessed_occurrences: 3,
    pending_occurrences: 0,
    coverage_rate: 1,
    oldest_pending_at: null,
    evaluator_failure_occurrences: 0,
  };
  return {
    scope: { kind: "workspace", source_id: null, source_type: null },
    window: {
      from: "2026-08-19T00:00:00+00:00",
      to: "2026-08-20T00:00:00+00:00",
      days: 1,
    },
    summary: {
      total_assessments: 3,
      runtime_event_count: 3,
      eligible_assessment_count: 3,
      missing_assessment_count: 0,
      action_issue_group_count: 1,
      review_issue_group_count: 1,
      source_count: 3,
      affected_source_count: 2,
      label_counts: { pass: 1, fail: 1, needs_review: 1 },
      criterion_counts: {},
      status_counts: { completed: 3 },
      truncated: false,
    },
    coverage,
    issue_groups: [
      {
        group_id: "group-fail",
        label: "fail",
        criterion: "evidence_reference_validity",
        reason_code: "legacy_quote_unresolved",
        evaluator_name: "memforge.deterministic.runtime_contract",
        evaluator_version: "1",
        occurrence_count: 1,
        distinct_event_count: 1,
        criterion_occurrence_count: 1,
        criterion_rate: 1,
        affected_source_ids: ["src-teams"],
        affected_source_count: 1,
        source_types: ["teams"],
        first_seen_at: "2026-08-19T03:00:00+00:00",
        last_seen_at: "2026-08-19T04:00:00+00:00",
        representative_cases: [],
      },
      {
        group_id: "group-review",
        label: "needs_review",
        criterion: "evidence_localization",
        reason_code: "whole_block_fallback",
        evaluator_name: "memforge.deterministic.runtime_contract",
        evaluator_version: "1",
        occurrence_count: 1,
        distinct_event_count: 1,
        criterion_occurrence_count: 1,
        criterion_rate: 1,
        affected_source_ids: ["src-github"],
        affected_source_count: 1,
        source_types: ["github_repo"],
        first_seen_at: "2026-08-19T03:00:00+00:00",
        last_seen_at: "2026-08-19T04:00:00+00:00",
        representative_cases: [],
      },
    ],
    available_source_types: ["github_repo", "jira", "teams"],
    sources: [
      {
        source_id: "src-teams",
        name: "Workspace Teams",
        type: "teams",
        source_status: "active",
        evaluation_status: "attention",
        action_issue_group_count: 1,
        review_issue_group_count: 0,
        fail_occurrences: 1,
        review_occurrences: 0,
        coverage,
        last_event_at: "2026-08-19T04:00:00+00:00",
      },
      {
        source_id: "src-github",
        name: "My Repository",
        type: "github_repo",
        source_status: "active",
        evaluation_status: "review",
        action_issue_group_count: 0,
        review_issue_group_count: 1,
        fail_occurrences: 0,
        review_occurrences: 1,
        coverage,
        last_event_at: "2026-08-19T04:00:00+00:00",
      },
      {
        source_id: "src-jira",
        name: "Healthy Jira",
        type: "jira",
        source_status: "active",
        evaluation_status: "healthy",
        action_issue_group_count: 0,
        review_issue_group_count: 0,
        fail_occurrences: 0,
        review_occurrences: 0,
        coverage,
        last_event_at: "2026-08-19T04:00:00+00:00",
      },
    ],
    runtime_events: [],
    assessments: [],
  };
}

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}{location.search}</div>;
}

function renderPage(initialEntry = "/evaluation") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            path="/evaluation"
            element={<><OnlineEvaluationPage /><LocationProbe /></>}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("OnlineEvaluationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getMock.mockResolvedValue({ data: responseFixture() });
  });

  afterEach(cleanup);

  it("loads one workspace overview and drills a Source through the same route", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByText("2 / 3")).toBeTruthy();
    expect(getMock).toHaveBeenCalledWith("/agent-evaluations/online-overview", {
      params: { days: 1, source_id: undefined, source_type: undefined },
    });
    await user.click(screen.getByRole("tab", { name: "Sources (3)" }));
    expect(screen.getByText("Healthy Jira")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Open evaluation for Workspace Teams" }));

    await waitFor(() => expect(screen.getByTestId("location").textContent).toContain(
      "/evaluation?source_id=src-teams",
    ));
    await waitFor(() => expect(getMock).toHaveBeenCalledWith(
      "/agent-evaluations/online-overview",
      { params: { days: 1, source_id: "src-teams", source_type: undefined } },
    ));
  });

  it("preserves an explicit workspace hint and Source deep link", async () => {
    renderPage("/evaluation?workspace=mount_tai&source_id=src-teams");

    expect(await screen.findByText("2 / 3")).toBeTruthy();
    expect(screen.getByTestId("location").textContent).toContain("workspace=mount_tai");
    expect(getMock).toHaveBeenCalledWith("/agent-evaluations/online-overview", {
      params: { days: 1, source_id: "src-teams", source_type: undefined },
    });
  });

  it("keeps the workspace Source Type choices stable while applying a filter", async () => {
    const user = userEvent.setup();
    renderPage("/evaluation?workspace=mount_tai&source_id=src-teams");

    expect(await screen.findByText("2 / 3")).toBeTruthy();
    await user.click(screen.getByRole("combobox", { name: "Filter by Source Type" }));
    await user.click(await screen.findByRole("option", { name: "Jira" }));

    await waitFor(() => {
      const location = screen.getByTestId("location").textContent ?? "";
      expect(location).toContain("workspace=mount_tai");
      expect(location).toContain("source_type=jira");
      expect(location).not.toContain("source_id=");
    });
    await waitFor(() => expect(getMock).toHaveBeenCalledWith(
      "/agent-evaluations/online-overview",
      { params: { days: 1, source_id: undefined, source_type: "jira" } },
    ));
    await user.click(screen.getByRole("combobox", { name: "Filter by Source Type" }));
    expect(await screen.findByRole("option", { name: "Teams" })).toBeTruthy();
  });
});
