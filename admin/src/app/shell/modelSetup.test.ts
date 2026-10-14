import { describe, expect, test } from "vitest";
import {
  NO_MODELS,
  kindSettings,
  makeBinding,
  makeModelSettings,
  makeValidation,
  makeVectorIndex,
} from "@/test/modelSettingsFixtures";
import type { AdminExtension } from "../extension/contract";
import { modelSetupMessage, modelSetupMessages, searchIndexMessage, settingsLinkTarget, showsModelSetup } from "./modelSetup";

describe("modelSetupMessage", () => {
  test("says nothing while the generation model is ready, whatever the embedding model is", () => {
    expect(modelSetupMessage(makeModelSettings())).toBeNull();
    expect(modelSetupMessage(makeModelSettings({ embedding: kindSettings(null) }))).toBeNull();
  });

  test("a fresh installation is told to choose a generation model", () => {
    expect(modelSetupMessage(makeModelSettings(NO_MODELS))).toEqual({
      title: "No generation model is set up",
      body: "MemForge cannot write memories from your sources until you choose one.",
    });
  });

  test("a saved edit that has not passed validation does not count as a model", () => {
    const edit = makeBinding({ state: "not_validated", validation: null });

    expect(modelSetupMessage(makeModelSettings({ generation: kindSettings(null, edit) }))?.title).toBe(
      "No generation model is set up",
    );
  });

  test("an installation the deployment manages is told who has to act", () => {
    expect(modelSetupMessage(makeModelSettings({ ...NO_MODELS, writable: false }))?.body).toBe(
      "MemForge cannot write memories from your sources until the deployment environment names one.",
    );
  });

  test("a model in use that is not ready is named by what happened to it", () => {
    const message = (overrides: Parameters<typeof makeBinding>[0]) =>
      modelSetupMessage(makeModelSettings({ generation: kindSettings(makeBinding(overrides)) }))?.title;

    expect(message({ state: "not_validated", validation: null })).toBe("The generation model has not been validated yet");
    expect(
      message({ state: "needs_revalidation", validation: makeValidation({ passed: false, error_code: "timeout" }) }),
    ).toBe("The generation model failed its latest validation");
    expect(message({ state: "needs_revalidation", validation: makeValidation({ current: false }) })).toBe(
      "The generation model needs revalidation",
    );
  });
});

describe("searchIndexMessage", () => {
  test("says that vector search is off while the search index has to be rebuilt", () => {
    const settings = makeModelSettings({ vector_index: makeVectorIndex({ status: "reindex_required", reindex_target: "active" }) });

    expect(searchIndexMessage(settings)).toEqual({
      title: "Vector search is off",
      body: "The search index does not fit the embedding model in use. Search ranks by matching words until the index is rebuilt in Settings.",
    });
  });

  test("says so through a rebuild that started while vector search was off", () => {
    const settings = makeModelSettings({ vector_index: makeVectorIndex({ status: "reindexing", serves_search: false }) });

    expect(searchIndexMessage(settings)).toEqual({
      title: "Vector search is off",
      body: "The search index is being rebuilt for the embedding model in use. Search ranks by matching words until the rebuild has finished.",
    });
  });

  test("says nothing while the index fits, is being replaced while search reads it, or is not there to rebuild", () => {
    for (const status of ["ready", "reindexing", "awaiting_validation", "not_configured", "not_managed"] as const) {
      expect(searchIndexMessage(makeModelSettings({ vector_index: makeVectorIndex({ status }) }))).toBeNull();
    }
  });
});

test("an installation with neither a generation model nor a usable search index hears both, the generation model first", () => {
  const settings = makeModelSettings({
    generation: kindSettings(null),
    vector_index: makeVectorIndex({ status: "reindex_required", reindex_target: "active" }),
  });

  expect(modelSetupMessages(settings).map((message) => message.title)).toEqual([
    "No generation model is set up",
    "Vector search is off",
  ]);
  expect(modelSetupMessages(makeModelSettings())).toEqual([]);
});

test("the notice belongs on every product page but Settings", () => {
  expect(showsModelSetup("/sources")).toBe(true);
  expect(showsModelSetup("/memories/mem-1")).toBe(true);
  expect(showsModelSetup("/settings")).toBe(false);
  expect(showsModelSetup("/cloud/workspaces")).toBe(false);
});

describe("settingsLinkTarget", () => {
  function redirecting(visible?: boolean): AdminExtension {
    return {
      id: "cloud",
      reservedRouteRedirects: [
        { from: "settings", to: "/cloud/settings", ...(visible === undefined ? {} : { visibleWhen: () => visible }) },
      ],
    };
  }

  test("is the Settings page of the standalone app", () => {
    expect(settingsLinkTarget(undefined)).toBe("/settings");
    expect(settingsLinkTarget({ id: "cloud" })).toBe("/settings");
  });

  test("is the extension's page when it redirects Settings", () => {
    expect(settingsLinkTarget(redirecting())).toBe("/cloud/settings");
    expect(settingsLinkTarget(redirecting(true))).toBe("/cloud/settings");
  });

  test("is nothing for a user who cannot open the extension's page", () => {
    expect(settingsLinkTarget(redirecting(false))).toBeNull();
  });
});
