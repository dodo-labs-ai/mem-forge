import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { resourceClient } from "@/api/client";
import type { MemoryReviewDetail, MemoryReviewListResponse } from "@/api/types";
import { ReviewDetailPage } from "@/views/review/ReviewDetailPage";
import { ReviewQueuePage } from "@/views/review/ReviewQueuePage";

vi.mock("@/api/client", () => ({
  resourceClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
}));

const getMock = vi.mocked(resourceClient.get);
const postMock = vi.mocked(resourceClient.post);

function detailFixture(): MemoryReviewDetail {
  return {
    id: "review-update",
    kind: "supersede",
    status: "pending",
    review_origin: "memory",
    source_id: null,
    source_name: "Payroll Agent",
    incumbent_memory_id: "memory-a",
    challenger_memory_id: "memory-b",
    reason: "newer payroll close day",
    review_note: null,
    reviewer: null,
    expected_incumbent_updated_at: "2026-08-08T09:00:00+00:00",
    expected_challenger_updated_at: "2026-08-08T09:30:00+00:00",
    created_at: "2026-08-08T10:00:00+00:00",
    resolved_at: null,
    is_stale: false,
    decision_fingerprint: "review-decision-v1:exact",
    presentation: {
      decision_label: "Updated",
      summary: "Use the proposed memory or keep the current one?",
      why_human: "The update would change active memory state.",
      current_label: "Current memory",
      proposed_label: "Proposed memory",
      proposed_empty_text: "The proposed memory snapshot is unavailable.",
      actions: [
        {
          key: "use_latest_state",
          decision: "approve",
          label: "Use latest state",
          consequence: "Use the proposed state going forward.",
          requires_note: false,
        },
        {
          key: "keep_current_state",
          decision: "reject",
          label: "Keep current state",
          consequence: "Keep the current memory active and discard this proposal.",
          requires_note: true,
        },
      ],
      technical_reason: "newer payroll close day",
    },
    incumbent: {
      id: "memory-a",
      memory_type: "fact",
      content: "Payroll area A closes Friday.",
      confidence: 0.9,
      corroboration_count: 1,
      status: "active",
      entity_refs: [],
      evidence: [],
      created_at: "2026-08-08T09:00:00+00:00",
      updated_at: "2026-08-08T09:00:00+00:00",
    },
    challenger: {
      id: "memory-b",
      memory_type: "fact",
      content: "Payroll area A closes Thursday.",
      confidence: 0.9,
      corroboration_count: 1,
      status: "active",
      entity_refs: [],
      evidence: [],
      created_at: "2026-08-08T09:30:00+00:00",
      updated_at: "2026-08-08T09:30:00+00:00",
    },
    related_challengers: [],
  };
}

function lifecycleDetailFixture(): MemoryReviewDetail {
  const detail = detailFixture();
  return {
    ...detail,
    id: "review-lifecycle",
    kind: "lifecycle",
    review_origin: "lifecycle",
    source_id: "source-payroll",
    challenger_memory_id: "memory-b",
    presentation: {
      ...detail.presentation,
      decision_label: "Updated",
      summary: "Use the newly projected payroll state?",
      actions: [
        {
          key: "use_latest_state",
          decision: "approve",
          label: "Use latest state",
          consequence: "Apply the existing Lifecycle Plan.",
          requires_note: false,
        },
        {
          key: "keep_current_state",
          decision: "reject",
          label: "Keep current state",
          consequence: "Reject the existing Lifecycle Plan.",
          requires_note: true,
        },
      ],
    },
  };
}

function renderAt(pathname: string, element: ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[pathname]}>
        <Routes>
          <Route path={pathname.startsWith("/review/") ? "/review/:id" : "/review"} element={element} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Review interaction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(cleanup);

  it("requires a note to keep the current state and sends the fingerprint", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockResolvedValue({ data: { ...detail, status: "rejected" } });
    const user = userEvent.setup();

    renderAt("/review/review-update", <ReviewDetailPage />);

    expect(await screen.findByRole("heading", { name: detail.presentation.summary })).toBeTruthy();
    expect(screen.getByText(/discard this proposal/i)).toBeTruthy();
    const dismiss = screen.getByRole("button", { name: "Keep current state" });
    expect((dismiss as HTMLButtonElement).disabled).toBe(true);
    await user.type(screen.getByLabelText("Decision note"), "Different payroll environments");
    expect((dismiss as HTMLButtonElement).disabled).toBe(false);
    dismiss.focus();
    await user.keyboard("{Enter}");

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-update/reject", {
        expected_fingerprint: "review-decision-v1:exact",
        note: "Different payroll environments",
      }),
    );
  });

  it("uses the latest state without requiring a note", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockResolvedValue({ data: { ...detail, status: "approved" } });
    const user = userEvent.setup();

    renderAt("/review/review-update", <ReviewDetailPage />);
    await user.click(await screen.findByRole("button", { name: "Use latest state" }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-update/approve", {
        expected_fingerprint: "review-decision-v1:exact",
        note: null,
      }),
    );
  });

  it("uses the existing lifecycle action and reject endpoint", async () => {
    const detail = lifecycleDetailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockResolvedValue({ data: { ...detail, status: "rejected" } });
    const user = userEvent.setup();

    renderAt("/review/review-lifecycle", <ReviewDetailPage />);
    await user.type(await screen.findByLabelText("Decision note"), "Keep authoritative state");
    await user.click(screen.getByRole("button", { name: "Keep current state" }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-lifecycle/reject", {
        expected_fingerprint: "review-decision-v1:exact",
        note: "Keep authoritative state",
      }),
    );
  });

  it("refreshes a stale lifecycle proposal and opens the new pending Review", async () => {
    const stale = {
      ...lifecycleDetailFixture(),
      status: "stale" as const,
      is_stale: true,
      decision_fingerprint: "review-decision-v1:stale-lifecycle",
    };
    const refreshed = {
      ...lifecycleDetailFixture(),
      id: "review-lifecycle-refreshed",
      decision_fingerprint: "review-decision-v1:refreshed",
    };
    getMock.mockResolvedValue({ data: stale });
    postMock.mockResolvedValue({ data: refreshed });
    const user = userEvent.setup();

    renderAt("/review/review-lifecycle", <ReviewDetailPage />);
    await user.click(await screen.findByRole("button", { name: "Recheck current state" }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-lifecycle/refresh", {
        expected_fingerprint: "review-decision-v1:stale-lifecycle",
      }),
    );
    expect(screen.getByText(/This record remains in audit history/)).toBeTruthy();
  });

  it("surfaces a stale decision response and keeps the Review actionable after refetch", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockRejectedValue({
      response: { data: { detail: { error: "stale", message: "Review participants changed" } } },
    });
    const user = userEvent.setup();

    renderAt("/review/review-update", <ReviewDetailPage />);
    await screen.findByRole("button", { name: "Use latest state" });
    await user.click(screen.getByRole("button", { name: "Use latest state" }));

    expect(await screen.findByText("Review participants changed")).toBeTruthy();
    expect(getMock.mock.calls.length).toBeGreaterThan(1);
  });

  it("surfaces management permission failures without hiding the decision context", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockRejectedValue({ response: { status: 403, data: { detail: "Source management required" } } });
    const user = userEvent.setup();

    renderAt("/review/review-update", <ReviewDetailPage />);
    await user.click(await screen.findByRole("button", { name: "Use latest state" }));

    expect(await screen.findByText("Source management required")).toBeTruthy();
    expect(screen.getByText("Payroll area A closes Friday.")).toBeTruthy();
  });

  it("filters the exact queue cohort and resets pagination", async () => {
    const response: MemoryReviewListResponse = { data: [], total: 0, limit: 20, offset: 0 };
    getMock.mockResolvedValue({ data: response });

    renderAt("/review?page=3", <ReviewQueuePage />);
    const filter = await screen.findByLabelText("Filter reviews");
    fireEvent.change(filter, { target: { value: "memory" } });

    await waitFor(() => {
      const [, options] = getMock.mock.calls.at(-1) ?? [];
      expect(options?.params).toMatchObject({
        status: "open",
        origin: "memory",
        offset: 0,
      });
    });
    expect(screen.getByText(/Resolve lifecycle proposals and Memory updates/)).toBeTruthy();
  });

  it("shows the exact total and requests the next actionable page", async () => {
    const detail = detailFixture();
    const response: MemoryReviewListResponse = {
      data: [detail],
      total: 26,
      limit: 25,
      offset: 0,
    };
    getMock.mockResolvedValue({ data: response });
    const user = userEvent.setup();

    renderAt("/review", <ReviewQueuePage />);
    expect(await screen.findByText("26 reviews need your input")).toBeTruthy();
    expect(screen.getByText("Showing 1-25 of 26 reviews")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await waitFor(() => {
      const [, options] = getMock.mock.calls.at(-1) ?? [];
      expect(options?.params).toMatchObject({ limit: 25, offset: 25 });
    });
  });
});
