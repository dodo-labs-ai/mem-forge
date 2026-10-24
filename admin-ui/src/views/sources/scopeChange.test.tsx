import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import type { SourceScopeChangePreview, SourceScopeTransition } from "@/api/types";
import {
  lastSyncAbsenceNotes,
  refusedSaveMessage,
  scopeChangeWarning,
  scopeTransitionSummary,
  useScopeChangePreview,
} from "./scopeChange";
import { SourceSetupShell } from "./SourceSetupShell";

vi.mock("@/api/client", () => ({ resourceClient: { get: vi.fn(), post: vi.fn() } }));

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

const CANNOT_LIST =
  "A scope change needs a configuration that lists its whole scope, and the mode 'subtree read by link crawl' cannot, "
  + "so the change could never complete. Use a personal access token. Or create a new Source for this scope.";

function preview(overrides: Partial<SourceScopeChangePreview>): SourceScopeChangePreview {
  return { outcome: "unchanged", open_transition: false, removes: "at_completion", refusal: null, ...overrides };
}

function transition(overrides: Partial<SourceScopeTransition> = {}): SourceScopeTransition {
  return {
    id: "scope-transition-1",
    status: "failed",
    previous_scope: { spaces: ["ENG"] },
    target_scope: { spaces: ["ENG", "OPS"] },
    created_at: "2026-09-29T09:00:00Z",
    updated_at: "2026-09-29T11:00:00Z",
    members_committed: 117,
    listed_count: 120,
    excluded_scopes_due: false,
    error: null,
    failing_items: { count: 3, retrying_after_upgrade: 0, items: [] },
    restore_config: null,
    ...overrides,
  };
}

it("warns before saving a change into a mode that cannot list, with the mode and what to choose instead", () => {
  expect(
    scopeChangeWarning(
      preview({
        outcome: "refused",
        refusal: { code: "scope_change_requires_listing", mode: "subtree read by link crawl", message: CANNOT_LIST },
      }),
    ),
  ).toEqual({ tone: "warning", title: "This mode cannot change what the source syncs", message: CANNOT_LIST });
  expect(
    scopeChangeWarning(
      preview({ outcome: "refused", open_transition: true, refusal: { code: "transition_requires_listing", message: "Keep it." } }),
    ),
  ).toEqual({ tone: "warning", title: "This would keep the scope change in progress from finishing", message: "Keep it." });
});

it("says that saving replaces the scope change in progress", () => {
  expect(scopeChangeWarning(preview({ outcome: "supersedes_transition", open_transition: true }))).toMatchObject({
    tone: "info",
    title: "This replaces the scope change in progress",
  });
  expect(scopeChangeWarning(preview({ outcome: "restores_previous_scope", open_transition: true }))).toEqual({
    tone: "info",
    title: "This replaces the scope change in progress",
    message: "The source returns to its previous scope, and the change in progress closes. Nothing is removed.",
  });
  expect(scopeChangeWarning(preview({ outcome: "returns_to_previous_scope", open_transition: true }))).toEqual({
    tone: "info",
    title: "This replaces the scope change in progress",
    message:
      "The source returns to its previous scope. Nothing the previous scope synced is removed. Once a sync lists the previous scope in full, items only the replaced change added are removed.",
  });
});

it("says that a conversation source's next sync removes the deselected conversations, never that nothing is removed", () => {
  const removal =
    "The next sync removes the conversations the new selection leaves out; the change completes once every selected conversation is synced.";

  expect(
    scopeChangeWarning(preview({ outcome: "supersedes_transition", open_transition: true, removes: "next_sync" })),
  ).toEqual({
    tone: "info",
    title: "This replaces the scope change in progress",
    message: `The source changes from the scope it had before that change to this one. ${removal}`,
  });
  expect(
    scopeChangeWarning(preview({ outcome: "returns_to_previous_scope", open_transition: true, removes: "next_sync" })),
  ).toEqual({
    tone: "info",
    title: "This replaces the scope change in progress",
    message: `The source returns to its previous scope. ${removal}`,
  });
});

it("keeps the document wording for a document source that replaces its scope change", () => {
  expect(scopeChangeWarning(preview({ outcome: "supersedes_transition", open_transition: true }))?.message).toBe(
    "The source changes from the scope it had before that change to this one. Nothing is removed until every item of the new scope is synced.",
  );
});

it("says nothing when the change keeps or simply opens a scope change", () => {
  expect(scopeChangeWarning(undefined)).toBeNull();
  expect(scopeChangeWarning(preview({ outcome: "unchanged", open_transition: true }))).toBeNull();
  expect(scopeChangeWarning(preview({ outcome: "opens_transition" }))).toBeNull();
});

it("explains a conversation source that waits for its scope change", () => {
  expect(
    scopeChangeWarning(preview({ outcome: "refused", open_transition: true, refusal: { code: "projection_scope_transition_active" } })),
  ).toEqual({
    tone: "warning",
    title: "This change cannot be saved",
    message: "This source finishes its scope change in progress before it accepts another one. Save again once it completes.",
  });
});

it("maps the refusals of a save to clear messages, and leaves the rest to the caller", () => {
  const refused = (detail: unknown) => ({ response: { status: 409, data: { detail } } });

  expect(refusedSaveMessage(refused("projection_scope_transition_stale"))).toBe(
    "The scope change in progress changed while you were editing. Close this dialog, open the source again, and save.",
  );
  expect(refusedSaveMessage(refused("source_provider_namespace_immutable"))).toBe(
    "This change points the source at another site or project. Create a new source for it instead.",
  );
  expect(refusedSaveMessage(refused({ code: "scope_change_requires_listing", message: CANNOT_LIST }))).toBeNull();
  expect(refusedSaveMessage(new Error("network"))).toBeNull();
});

it("asks the server what saving would do, a moment after the last edit", async () => {
  vi.mocked(resourceClient.post).mockImplementation(async (_url, body) => ({
    data: (body as { config: { spaces: string[] } }).config.spaces.length > 1
      ? preview({ outcome: "supersedes_transition", open_transition: true })
      : preview({ open_transition: true }),
  }));
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

  const { result, rerender } = renderHook(({ config }) => useScopeChangePreview("src-1", config), {
    wrapper,
    initialProps: { config: { spaces: ["ENG"] } as Record<string, unknown> | null },
  });
  await waitFor(() => expect(result.current?.outcome).toBe("unchanged"));
  rerender({ config: { spaces: ["ENG", "O"] } });
  rerender({ config: { spaces: ["ENG", "OPS"] } });

  await waitFor(() => expect(result.current?.outcome).toBe("supersedes_transition"));
  // The edits in between are not asked about.
  expect(vi.mocked(resourceClient.post).mock.calls).toEqual([
    ["/sources/src-1/scope-change-preview", { config: { spaces: ["ENG"] } }],
    ["/sources/src-1/scope-change-preview", { config: { spaces: ["ENG", "OPS"] } }],
  ]);
});

it("keeps a typed secret out of the cache key, but sends it", async () => {
  vi.mocked(resourceClient.post).mockResolvedValue({ data: preview({ outcome: "opens_transition" }) });
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

  const { result } = renderHook(
    () => useScopeChangePreview("src-1", { spaces: ["OPS"], pat: "typed-secret" }, ["pat"]),
    { wrapper },
  );

  await waitFor(() => expect(result.current?.outcome).toBe("opens_transition"));
  expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-1/scope-change-preview", {
    config: { spaces: ["OPS"], pat: "typed-secret" },
  });
  expect(JSON.stringify(client.getQueryCache().getAll().map((query) => query.queryKey))).not.toContain("typed-secret");
});

it("keeps the last answer while the next one is on its way", async () => {
  let answer: (value: unknown) => void = () => {};
  vi.mocked(resourceClient.post)
    .mockResolvedValueOnce({ data: preview({ outcome: "supersedes_transition", open_transition: true }) })
    .mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

  const { result, rerender } = renderHook(({ config }) => useScopeChangePreview("src-1", config), {
    wrapper,
    initialProps: { config: { spaces: ["OPS"] } as Record<string, unknown> | null },
  });
  await waitFor(() => expect(result.current?.outcome).toBe("supersedes_transition"));
  rerender({ config: { spaces: ["OPS", "PAY"] } });

  await waitFor(() => expect(resourceClient.post).toHaveBeenCalledTimes(2));
  expect(result.current?.outcome).toBe("supersedes_transition");
  answer({ data: preview({ outcome: "unchanged", open_transition: true }) });
  await waitFor(() => expect(result.current?.outcome).toBe("unchanged"));
});

it("asks nothing for a new source or an incomplete form", async () => {
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

  const { result } = renderHook(() => useScopeChangePreview(undefined, { spaces: ["ENG"] }), { wrapper });
  renderHook(() => useScopeChangePreview("src-1", null), { wrapper });

  await new Promise((resolve) => setTimeout(resolve, 500));
  expect(result.current).toBeUndefined();
  expect(resourceClient.post).not.toHaveBeenCalled();
});

it("the configuration form shows the warning above its save button", () => {
  render(
    <Dialog open>
      <DialogContent>
        <SourceSetupShell
          sourceType="confluence"
          sourceLabel="Confluence"
          sourceName="Payroll wiki"
          connection={{ mode: "direct", label: "Server" }}
          isEdit
          sections={[]}
          openSection="basics"
          onOpenSectionChange={() => {}}
          scopeChange={{ tone: "info", title: "This replaces the scope change in progress", message: "Nothing is removed." }}
          saving={false}
          onCancel={() => {}}
          onSave={() => {}}
        />
      </DialogContent>
    </Dialog>,
  );

  const note = screen.getByText("This replaces the scope change in progress").closest("[role=status]");
  expect(note?.textContent).toBe("This replaces the scope change in progressNothing is removed.");
});

it("a finished sync's details say why it removed nothing and what it kept", () => {
  expect(lastSyncAbsenceNotes({ absence_check_skipped_reason: "absence_confirmation_failed", unlisted_present_count: 2 })).toEqual([
    "Deletion check skipped: the source could not confirm which missing items were deleted. Nothing was removed; the next sync checks again.",
    "2 items are outside the sync scope but still in the source, so they are kept and checked again each sync.",
  ]);
  expect(lastSyncAbsenceNotes({ absence_check_skipped_reason: null, unlisted_present_count: 1 })).toEqual([
    "1 item is outside the sync scope but still in the source, so it is kept and checked again each sync.",
  ]);
  expect(lastSyncAbsenceNotes({})).toEqual([]);
});

it("summarizes the scope change in progress in one line", () => {
  expect(scopeTransitionSummary(transition())).toBe(
    "Scope change in progress: 117 of 120 listed items synced, 3 failed. Nothing is removed until it completes.",
  );
  expect(scopeTransitionSummary(transition({ listed_count: null }))).toBe(
    "Scope change in progress: the next sync lists what the new scope holds.",
  );
  expect(scopeTransitionSummary(transition({ excluded_scopes_due: true }))).toBe(
    "Scope change in progress: every listed item is synced, and the next sync completes it.",
  );
  expect(scopeTransitionSummary(transition({ target_scope: { spaces: ["ENG"] }, listed_count: null }))).toBe(
    "Returning to the previous scope: the next sync lists what the previous scope holds.",
  );
  expect(scopeTransitionSummary(transition({ previous_scope: null, target_scope: null }))).toBe(
    "Scope change in progress: 117 of 120 listed items synced, 3 failed. Nothing is removed until it completes.",
  );
});
