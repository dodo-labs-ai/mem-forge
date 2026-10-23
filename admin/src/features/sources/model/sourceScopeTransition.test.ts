import { describe, expect, test } from "vitest";
import { ApiError } from "@/api";
import { errorCodeOf } from "@/api/errors";
import { makeFailedItem, makeScopeTransition, makeSource } from "@/test/sourceFixtures";
import {
  presentRestore,
  presentSourceScopeTransition,
  scopeChangeErrorMessage,
  scopeFieldChanges,
  scopeValueLabel,
  transitionFailedItemsTitle,
  type ScopeChangePreview,
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
      returning: false,
      status: "The last sync left it open",
      summary: "Nothing is removed until every item the new scope lists is synced. Each sync retries the items that failed.",
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

  test("a viewer who cannot read the configuration sees the progress, but neither scope nor a way back", () => {
    const view = presentSourceScopeTransition(
      makeSource({
        scope_transition: makeScopeTransition({ previous_scope: null, target_scope: null, restore_config: null }),
      }),
      undefined,
    );

    expect(view?.changes).toEqual([]);
    expect(view?.restoreConfig).toBeNull();
    expect(view?.progressLabel).toBe("117 of 120 listed items synced under the new scope");
  });

  test("a change back to the previous scope says it returns there", () => {
    const view = presentSourceScopeTransition(
      makeSource({
        scope_transition: makeScopeTransition({
          previous_scope: { spaces: ["ENG"], sync_mode: "spaces" },
          target_scope: { sync_mode: "spaces", spaces: ["ENG"] },
          restore_config: null,
          members_committed: 3,
          listed_count: 40,
        }),
      }),
      undefined,
    );

    expect(view?.returning).toBe(true);
    expect(view?.changes).toEqual([]);
    expect(view?.progressLabel).toBe("3 of 40 listed items synced under the previous scope");
    expect(view?.summary).toBe(
      "Nothing the previous scope synced is removed. Once a sync lists it in full, items only the replaced change added are removed.",
    );
  });

  test.each([
    ["a change", { conversation_ids: ["19:a@thread.v2", "19:c@thread.v2"] }, false],
    ["a change back to the previous selection", { conversation_ids: ["19:a@thread.v2"] }, true],
  ])("%s of a conversation source says its next sync removes the deselected conversations", (_name, target, returning) => {
    const view = presentSourceScopeTransition(
      makeSource({
        scope_transition: makeScopeTransition({
          status: "pending",
          previous_scope: { conversation_ids: ["19:a@thread.v2"] },
          target_scope: target,
          members_committed: 0,
          listed_count: null,
          error: null,
          removes: "next_sync",
        }),
      }),
      undefined,
    );

    expect(view?.returning).toBe(returning);
    expect(view?.summary).toBe("The next sync removes the conversations the new selection leaves out.");
    expect(view?.progressLabel).toBe("The change completes once every selected conversation is synced.");
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

  test("names the failed items not yet synced under the change", () => {
    expect(transitionFailedItemsTitle(1)).toBe("1 failed item not yet synced under the change");
    expect(transitionFailedItemsTitle(3)).toBe("3 failed items not yet synced under the change");
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

describe("presentRestore", () => {
  const preview = (
    outcome: ScopeChangePreview["outcome"],
    removes: ScopeChangePreview["removes"] = "at_completion",
  ): ScopeChangePreview => ({
    outcome,
    open_transition: true,
    removes,
    refusal: null,
  });

  test("a configuration that cannot list closes the change and removes nothing", () => {
    expect(presentRestore("Wiki", preview("restores_previous_scope"), false)).toEqual({
      description: "This closes the scope change in progress and saves Wiki's previous scope again, keeping its other settings. Nothing is removed.",
      ready: true,
    });
  });

  test("a configuration that lists returns through a change that removes what only the replaced change added", () => {
    const restoring = presentRestore("Wiki", preview("returns_to_previous_scope"), false);

    expect(restoring.ready).toBe(true);
    expect(restoring.description).toContain("Nothing the previous scope synced is removed.");
    expect(restoring.description).toContain("items only the replaced change added are removed");
  });

  test("a conversation source returns through a change whose next sync removes what only the replaced change added", () => {
    expect(presentRestore("Team chats", preview("returns_to_previous_scope", "next_sync"), false)).toEqual({
      description:
        "This replaces the scope change in progress with a change back to Team chats's previous scope, keeping its other settings. "
        + "The next sync removes the conversations the new selection leaves out. The change completes once every selected conversation is synced.",
      ready: true,
    });
  });

  test("waits for the preview, and offers nothing when the change moved on or would be refused", () => {
    expect(presentRestore("Wiki", undefined, false).ready).toBe(false);
    expect(presentRestore("Wiki", preview("unchanged"), false)).toEqual({
      description: "The scope change in progress changed while you were looking at it. Reload the source, then try again.",
      ready: false,
    });
    expect(
      presentRestore("Wiki", { ...preview("refused"), refusal: { code: "source_sync_running" } }, false),
    ).toEqual({
      description: "A sync of this source is running. Stop the sync, then try again.",
      ready: false,
    });
  });

  test("a preview that fails still lets the restore go ahead, as the save decides again", () => {
    expect(presentRestore("Wiki", undefined, true).ready).toBe(true);
  });
});
