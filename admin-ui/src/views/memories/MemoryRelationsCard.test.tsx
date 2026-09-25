import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import type { Memory, RelatedMemory } from "@/api/types";
import { MemoryRelationsCard } from "@/views/memories/MemoryRelationsCard";

vi.mock("@/api/client", () => ({
  resourceClient: {
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

const postMock = vi.mocked(resourceClient.post);
const deleteMock = vi.mocked(resourceClient.delete);

const counterpart: RelatedMemory = {
  memory_id: "memory-b",
  summary: "Payroll closes on Thursday.",
  content_hash: "hash-b",
  sources: [{ source_id: "src-jira", source_type: "jira", name: "Payroll backlog" }],
  revision_at: "2026-08-08T09:30:00+00:00",
};

function memoryFixture(overrides: Partial<Memory> = {}): Memory {
  return {
    id: "memory-a",
    memory_type: "fact",
    content: "Payroll closes on Friday.",
    content_hash: "hash-a",
    visibility: "workspace",
    owner_user_id: null,
    project_key: null,
    confidence: 0.9,
    corroboration_count: 1,
    status: "active",
    retirement_reason: null,
    retired_at: null,
    superseded_at: null,
    superseded_by: null,
    replacement_reason: null,
    valid_from: null,
    valid_until: null,
    created_at: "2026-08-08T09:00:00+00:00",
    updated_at: "2026-08-08T09:00:00+00:00",
    extraction_context: null,
    entity_refs: [],
    evidence: [],
    origin_source_type: null,
    relations: [],
    dismissed_relations: [],
    ...overrides,
  };
}

function renderCard(memory: Memory) {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <MemoryRelationsCard memory={memory} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Memory relations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(cleanup);

  it("renders nothing for a Memory without relations", () => {
    const { container } = renderCard(memoryFixture());
    expect(container.textContent).toBe("");
  });

  it("dismisses the shown label for both current contents", async () => {
    postMock.mockResolvedValue({ data: {} });
    const user = userEvent.setup();
    renderCard(
      memoryFixture({
        relations: [
          {
            label: "updates",
            role: "older",
            counterpart,
            reason: "The close day moved.",
            decided_by: "classifier",
          },
        ],
      }),
    );

    expect(screen.getByText("Updated by the newer")).toBeTruthy();
    expect(screen.getByText(/Payroll backlog/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Not an update" }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/memories/memory-a/relations/memory-b/dismissal", {
        label: "updates",
        expected_content_hash: "hash-a",
        counterpart_expected_content_hash: "hash-b",
      }),
    );
  });

  it("undoes a dismissal in force", async () => {
    deleteMock.mockResolvedValue({ data: { restored_dismissal_ids: ["rdm-1"] } });
    const user = userEvent.setup();
    renderCard(
      memoryFixture({
        dismissed_relations: [
          {
            labels: ["contradicts", "updates"],
            counterpart,
            dismissed_by: "reviewer@example.test",
            dismissed_at: "2026-08-09T09:00:00+00:00",
            note: "Different payroll areas.",
          },
        ],
      }),
    );

    expect(
      screen.getByText(/Not a conflict, Not an update: dismissed by reviewer@example.test/),
    ).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() =>
      expect(deleteMock).toHaveBeenCalledWith("/memories/memory-a/relations/memory-b/dismissal"),
    );
  });
});
