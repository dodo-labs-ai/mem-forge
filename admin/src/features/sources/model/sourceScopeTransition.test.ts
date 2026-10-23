import { describe, expect, test } from "vitest";
import { ApiError } from "@/api";
import { errorCodeOf } from "@/api/errors";
import { makeFailedItem, makeScopeTransition, makeSource } from "@/test/sourceFixtures";
import {
  presentSourceScopeTransition,
  scopeChangeErrorMessage,
  scopeFieldChanges,
  scopeValueLabel,
  transitionFailedItemsTitle,
} from "./sourceScopeTransition";

describe("presentSourceScopeTransition", () => {
  test("a source without a scope change in progress shows none", () => {
    expect(presentSourceScopeTransition(makeSource(), undefined)).toBeNull();
  });

  test("shows what the change moves, how far it is, why it is open, and the way back", () => {
    const transition = makeScopeTransition({
      failing_items: { count: 1, retrying_after_upgrade: 0, items: [makeFailedItem({ document_id: "page-3", title: "OPS runbook" })] },
    });

    const view = presentSourceScopeTransition(makeSource({ scope_transition: transition }), undefined, { spaces: "Spaces" });

    expect(view).toEqual({
      status: "The last sync left it open",
      reason: "3 listed Documents have not committed under the transition",
      changes: [{ field: "spaces", label: "Spaces", from: "ENG", to: "ENG, OPS" }],
      progressLabel: "117 of 120 listed items synced under the new scope",
      failedItems: {
        count: 1,
        retryingAfterUpgrade: 0,
        items: [
          { documentId: "page-3", title: "OPS runbook", stage: "Reading", reason: "403 Forbidden", retryingAfterUpgrade: false },
        ],
      },
      restoreConfig: transition.restore_config,
    });
  });

  test("before a sync lists the new scope, it says what comes next", () => {
    const view = presentSourceScopeTransition(
      makeSource({ scope_transition: makeScopeTransition({ status: "pending", listed_count: null, members_committed: 0, error: null }) }),
      undefined,
    );

    expect(view?.status).toBe("Waiting for the next sync");
    expect(view?.progressLabel).toBe("The next sync lists what the new scope holds.");
    expect(view?.reason).toBeNull();
    expect(view?.failedItems).toBeNull();
  });

  test("once every listed item is synced, it says the next sync checks the previous scope", () => {
    const view = presentSourceScopeTransition(
      makeSource({ scope_transition: makeScopeTransition({ members_committed: 120, excluded_scopes_due: true }) }),
      undefined,
    );

    expect(view?.progressLabel).toBe(
      "Every listed item is synced. The next sync checks what the previous scope held, then completes the change.",
    );
    expect(view?.reason).toBeNull();
  });

  test("items synced earlier that the latest listing no longer holds do not count past it", () => {
    const view = presentSourceScopeTransition(
      makeSource({ scope_transition: makeScopeTransition({ members_committed: 125, listed_count: 1 }) }),
      undefined,
    );

    expect(view?.progressLabel).toBe("1 of 1 listed item synced under the new scope");
  });

  test("a viewer who cannot change the configuration gets no way back", () => {
    const view = presentSourceScopeTransition(
      makeSource({ scope_transition: makeScopeTransition({ restore_config: null }) }),
      undefined,
    );

    expect(view?.restoreConfig).toBeNull();
  });
});

describe("scope settings", () => {
  test("lists only the settings that differ, by label, with readable values", () => {
    expect(
      scopeFieldChanges({
        previous_scope: { query_mode: "simple", projects: ["OPS", "PAY"], include_comments: true },
        target_scope: { query_mode: "advanced", jql: "project = PAY", include_comments: true },
      }),
    ).toEqual([
      { field: "jql", label: "Jql", from: "Not set", to: "project = PAY" },
      { field: "projects", label: "Projects", from: "OPS, PAY", to: "Not set" },
      { field: "query_mode", label: "Query mode", from: "simple", to: "advanced" },
    ]);
  });

  test.each([
    [null, "Not set"],
    [[], "Not set"],
    [false, "No"],
    [3, "3"],
    [{ depth: 2 }, '{"depth":2}'],
  ])("reads %o as %s", (value, label) => {
    expect(scopeValueLabel(value)).toBe(label);
  });

  test("names the failed items that keep the change open", () => {
    expect(transitionFailedItemsTitle(1)).toBe("1 failed item keeps the change open");
    expect(transitionFailedItemsTitle(3)).toBe("3 failed items keep the change open");
  });
});

describe("scopeChangeErrorMessage", () => {
  test("explains each refusal of a scope change in words the user can act on", () => {
    expect(scopeChangeErrorMessage(new ApiError(409, "A sync of this Source is running.", "source_sync_running"))).toBe(
      "A sync of this source is running. Stop the sync, then try again.",
    );
    expect(scopeChangeErrorMessage(new ApiError(409, "projection_scope_transition_stale"))).toBe(
      "The scope change in progress changed while you were looking at it. Reload the source, then try again.",
    );
    expect(scopeChangeErrorMessage(new ApiError(409, "projection_scope_transition_active"))).toBe(
      "This source finishes its scope change in progress before it accepts another one. Try again once it completes.",
    );
  });

  test("a mode that cannot list keeps the server's words, which name it and what to choose instead", () => {
    const message = "A scope change needs a configuration that lists its whole scope, and the mode 'link crawl' cannot.";

    expect(scopeChangeErrorMessage(new ApiError(409, message, "scope_change_requires_listing"))).toBe(message);
  });

  test("a coded refusal carries its code as `code` or `error`", () => {
    expect(errorCodeOf({ detail: { code: "transition_requires_listing", message: "…" } })).toBe("transition_requires_listing");
    expect(errorCodeOf({ detail: { error: "reindex_in_progress", message: "…" } })).toBe("reindex_in_progress");
    expect(errorCodeOf({ detail: "projection_scope_transition_stale" })).toBeNull();
  });
});
