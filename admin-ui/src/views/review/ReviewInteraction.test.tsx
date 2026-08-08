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
    id: "review-conflict",
    kind: "cross_source_conflict",
    status: "pending",
    review_origin: "memory",
    source_id: null,
    source_name: "Payroll Agent",
    incumbent_memory_id: "memory-a",
    challenger_memory_id: "memory-b",
    reason: "same-scope contradiction",
    review_note: null,
    reviewer: null,
    expected_incumbent_updated_at: "2026-08-08T09:00:00+00:00",
    expected_challenger_updated_at: "2026-08-08T09:30:00+00:00",
    created_at: "2026-08-08T10:00:00+00:00",
    resolved_at: null,
    is_stale: false,
    decision_fingerprint: "review-decision-v1:exact",
    presentation: {
      decision_label: "Conflict",
      summary: "Do these source-backed memories really conflict?",
      why_human: "Source authority has not been decided.",
      current_label: "Source-backed memory A",
      proposed_label: "Source-backed memory B",
      proposed_empty_text: "Memory B unavailable",
      actions: [
        {
          key: "confirm_conflict",
          decision: "approve",
          label: "Confirm conflict",
          consequence: "Close the finding and keep both memories active.",
          requires_note: false,
        },
        {
          key: "not_a_conflict",
          decision: "reject",
          label: "Not a conflict",
          consequence: "Dismiss the finding and keep both memories active.",
          requires_note: true,
        },
      ],
      technical_reason: "same-scope contradiction",
    },
    incumbent: {
      id: "memory-a",
      memory_type: "fact",
      content: "Payroll area A closes Friday.",
      confidence: 0.9,
      corroboration_count: 1,
      status: "active",
      entity_refs: [],
      sources: [],
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
      sources: [],
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

  it("uses truthful cross-source dismissal, requires a note, and sends the fingerprint", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockResolvedValue({ data: { ...detail, status: "rejected" } });
    const user = userEvent.setup();

    renderAt("/review/review-conflict", <ReviewDetailPage />);

    expect(await screen.findByRole("heading", { name: detail.presentation.summary })).toBeTruthy();
    expect(screen.getAllByText(/keep both memories active/i)).toHaveLength(2);
    const dismiss = screen.getByRole("button", { name: "Not a conflict" });
    expect((dismiss as HTMLButtonElement).disabled).toBe(true);
    await user.type(screen.getByLabelText("Decision note"), "Different payroll environments");
    expect((dismiss as HTMLButtonElement).disabled).toBe(false);
    dismiss.focus();
    await user.keyboard("{Enter}");

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-conflict/reject", {
        expected_fingerprint: "review-decision-v1:exact",
        note: "Different payroll environments",
      }),
    );
  });

  it("confirms a cross-source finding without requiring a note", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockResolvedValue({ data: { ...detail, status: "approved" } });
    const user = userEvent.setup();

    renderAt("/review/review-conflict", <ReviewDetailPage />);
    await user.click(await screen.findByRole("button", { name: "Confirm conflict" }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memory-reviews/review-conflict/approve", {
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

  it("surfaces a stale decision response and keeps the Review actionable after refetch", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockRejectedValue({
      response: { data: { detail: { error: "stale", message: "Review participants changed" } } },
    });
    const user = userEvent.setup();

    renderAt("/review/review-conflict", <ReviewDetailPage />);
    await screen.findByRole("button", { name: "Confirm conflict" });
    await user.click(screen.getByRole("button", { name: "Confirm conflict" }));

    expect(await screen.findByText("Review participants changed")).toBeTruthy();
    expect(getMock.mock.calls.length).toBeGreaterThan(1);
  });

  it("surfaces management permission failures without hiding the decision context", async () => {
    const detail = detailFixture();
    getMock.mockResolvedValue({ data: detail });
    postMock.mockRejectedValue({ response: { status: 403, data: { detail: "Source management required" } } });
    const user = userEvent.setup();

    renderAt("/review/review-conflict", <ReviewDetailPage />);
    await user.click(await screen.findByRole("button", { name: "Confirm conflict" }));

    expect(await screen.findByText("Source management required")).toBeTruthy();
    expect(screen.getByText("Payroll area A closes Friday.")).toBeTruthy();
  });

  it("filters the exact queue cohort and resets pagination", async () => {
    const response: MemoryReviewListResponse = { data: [], total: 0, limit: 20, offset: 0 };
    getMock.mockResolvedValue({ data: response });

    renderAt("/review?page=3", <ReviewQueuePage />);
    const filter = await screen.findByLabelText("Filter reviews");
    fireEvent.change(filter, { target: { value: "cross_source_conflict" } });

    await waitFor(() => {
      const [, options] = getMock.mock.calls.at(-1) ?? [];
      expect(options?.params).toMatchObject({
        status: "open",
        kind: "cross_source_conflict",
        offset: 0,
      });
    });
    expect(screen.getByText(/Resolve lifecycle proposals and conflict findings/)).toBeTruthy();
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
