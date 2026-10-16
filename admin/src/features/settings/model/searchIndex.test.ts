import { describe, expect, test } from "vitest";
import { ApiError } from "@/api";
import {
  PROVIDERS,
  kindSettings,
  makeEmbeddingBinding,
  makeIndexRecord,
  makeProfile,
  makeReindex,
  makeVectorIndex,
} from "@/test/modelSettingsFixtures";
import {
  builtWith,
  failedRebuild,
  rebuildFailure,
  rebuildFailureReason,
  rebuildLabel,
  rebuildProgress,
  rebuildTarget,
  searchIndexMarks,
  searchIndexSummary,
  startRefusalReason,
} from "./searchIndex";

const NOW = new Date("2026-03-02T09:05:00Z");
const EMBEDDING = kindSettings(makeEmbeddingBinding());
/** The vector space of the hosted model the similarity thresholds were tuned on. */
const OPENAI_PROFILE = makeProfile({
  provider: "openai",
  route: "openai/text-embedding-3-small",
  endpoint: "https://api.openai.com/v1",
  model_id: "text-embedding-3-small",
  dimension: 1536,
  fingerprint: "b2".repeat(32),
  thresholds_calibrated: true,
});

describe("searchIndexSummary", () => {
  test("an index that fits the model in use says that vector search is on", () => {
    expect(searchIndexSummary(makeVectorIndex(), EMBEDDING, PROVIDERS, NOW)).toEqual({
      tone: "ok",
      headline: "Vector search is on.",
      detail: null,
    });
  });

  test("an empty index keeps vector search on and says how memories get into it", () => {
    const empty = makeVectorIndex({ index: makeIndexRecord({ vector_count: 0 }) });

    expect(searchIndexSummary(empty, EMBEDDING, PROVIDERS, NOW)).toEqual({
      tone: "ok",
      headline: "Vector search is on.",
      detail: "No memories yet: add a source and sync it, and they become searchable as they are written.",
    });
  });

  test("an index that a rebuild put in place says when", () => {
    const rebuilt = makeVectorIndex({
      reindex: makeReindex({ state: "succeeded", memories_indexed: 1204, finished_at: "2026-03-02T09:03:00Z" }),
    });

    expect(searchIndexSummary(rebuilt, EMBEDDING, PROVIDERS, NOW)?.detail).toBe("The index was rebuilt 2 min ago.");
  });

  test("a rebuild of an index that is no longer the one in use is not mentioned", () => {
    const replaced = makeVectorIndex({
      reindex: makeReindex({ state: "succeeded", profile: OPENAI_PROFILE, finished_at: "2026-03-02T09:03:00Z" }),
    });

    expect(searchIndexSummary(replaced, EMBEDDING, PROVIDERS, NOW)?.detail).toBeNull();
  });

  test("an index of another model says which two models, and what search does meanwhile", () => {
    const other = makeVectorIndex({
      status: "reindex_required",
      reindex_target: "active",
      index: makeIndexRecord({ profile: OPENAI_PROFILE }),
    });

    expect(searchIndexSummary(other, EMBEDDING, PROVIDERS, NOW)).toEqual({
      tone: "danger",
      headline: "Vector search is off.",
      detail:
        "This index was built with OpenAI text-embedding-3-small, and MemForge now uses Ollama nomic-embed-text. Search ranks by matching words, and new memories cannot be created, until the index is rebuilt.",
    });
  });

  describe("an index of the model in use under other settings says what differs", () => {
    const BUILT = makeProfile();
    const required = makeVectorIndex({ status: "reindex_required", reindex_target: "active" });
    const detailFor = (inUse: Parameters<typeof makeEmbeddingBinding>[0]) =>
      searchIndexSummary(required, kindSettings(makeEmbeddingBinding(inUse)), PROVIDERS, NOW)?.detail;

    test("the vector size the model now answers with", () => {
      const shorter = makeProfile({ dimension: 512, fingerprint: "c3".repeat(32) });

      expect(detailFor({ profile: shorter })).toBe(
        "This index holds vectors of 768 numbers from Ollama nomic-embed-text, and that model now returns vectors of 512 numbers. Search ranks by matching words, and new memories cannot be created, until the index is rebuilt.",
      );
    });

    test("the vector size it is now asked for", () => {
      const asked = makeProfile({ requested_dimensions: 256, dimension: 256, fingerprint: "d4".repeat(32) });

      expect(detailFor({ profile: asked })).toContain(
        "This index was built with Ollama nomic-embed-text asked for vectors of the model's own size, and MemForge now asks it for vectors of 256 numbers.",
      );
      expect(detailFor({ profile: null, parameters: { dimensions: 256 } })).toContain("MemForge now asks it for vectors of 256 numbers.");
    });

    test("the address it is now reached at, also before the model has been validated there", () => {
      const moved = "http://embeddings.internal:11434";

      for (const inUse of [{ profile: makeProfile({ endpoint: moved, fingerprint: "e5".repeat(32) }) }, { profile: null, endpoint: moved }]) {
        expect(detailFor(inUse)).toContain(
          `This index was built with Ollama nomic-embed-text at ${BUILT.endpoint}, and MemForge now reaches that model at ${moved}.`,
        );
      }
    });

    test("nothing it can name, when the settings it sees are the ones the index was built with", () => {
      expect(detailFor({ profile: null, endpoint: `${BUILT.endpoint}/` })).toBe(
        "This index was built with other settings of Ollama nomic-embed-text than the ones MemForge now uses. Search ranks by matching words, and new memories cannot be created, until the index is rebuilt.",
      );
    });
  });

  test("stored vectors that do not fit the model in use leave no index to name", () => {
    const none = makeVectorIndex({ status: "reindex_required", reindex_target: "active", index: null });

    expect(searchIndexSummary(none, EMBEDDING, PROVIDERS, NOW)?.detail).toBe(
      "The vectors stored so far do not fit Ollama nomic-embed-text. Search ranks by matching words, and new memories cannot be created, until the index is rebuilt.",
    );
  });

  test("a running rebuild names the model it embeds with", () => {
    const running = makeVectorIndex({ status: "reindexing", reindex: makeReindex({ profile: OPENAI_PROFILE }) });

    expect(searchIndexSummary(running, EMBEDDING, PROVIDERS, NOW)).toEqual({
      tone: "live",
      headline: "Rebuilding the search index.",
      detail: "Every memory is embedded again with OpenAI text-embedding-3-small.",
    });
  });

  test("a model that has not passed validation has no index yet", () => {
    const waiting = makeVectorIndex({ status: "awaiting_validation", index: null });

    expect(searchIndexSummary(waiting, EMBEDDING, PROVIDERS, NOW)).toEqual({
      tone: "warn",
      headline: "No search index yet.",
      detail: "MemForge builds it once the embedding model has passed validation.",
    });
  });

  test("there is nothing to say without an embedding model, or about an index the deployment keeps", () => {
    expect(searchIndexSummary(makeVectorIndex({ status: "not_configured", index: null }), kindSettings(null), PROVIDERS)).toBeNull();
    expect(searchIndexSummary(makeVectorIndex({ status: "not_managed", index: null }), EMBEDDING, PROVIDERS)).toBeNull();
  });
});

test("the index names the model that built it, when the server recorded one", () => {
  expect(builtWith(makeVectorIndex(), PROVIDERS)).toBe("Ollama nomic-embed-text");
  expect(builtWith(makeVectorIndex({ index: makeIndexRecord({ profile: null }) }), PROVIDERS)).toBeNull();
  expect(builtWith(makeVectorIndex({ index: null }), PROVIDERS)).toBeNull();
});

describe("searchIndexMarks", () => {
  test("an index that was carried over, and thresholds tuned on another model, are both marked", () => {
    const carried = makeVectorIndex({ reindex_target: "active", index: makeIndexRecord({ adopted: true }) });

    expect(searchIndexMarks(carried).map((mark) => mark.label)).toEqual(["Carried over", "Thresholds not calibrated"]);
    expect(searchIndexMarks(carried)[0]!.text).toContain("Rebuild the index if another model made them.");
  });

  test("an index of another model than the one in use is not read, so nothing about its vectors is marked", () => {
    const other = makeVectorIndex({ status: "reindex_required", reindex_target: "active", index: makeIndexRecord({ adopted: true }) });

    expect(searchIndexMarks(other)).toEqual([]);
  });

  test("an index that is being replaced because it does not fit the model in use is not read either", () => {
    const replaced = makeVectorIndex({
      status: "reindexing",
      serves_search: false,
      index: makeIndexRecord({ adopted: true }),
      reindex: makeReindex({ target: "active", profile: OPENAI_PROFILE }),
    });
    const kept = makeVectorIndex({ status: "reindexing", index: makeIndexRecord({ adopted: true }), reindex: makeReindex() });

    expect(searchIndexMarks(replaced)).toEqual([]);
    expect(searchIndexMarks(kept).map((mark) => mark.label)).toEqual(["Carried over", "Thresholds not calibrated"]);
  });

  test("an index built by the model the thresholds were tuned on has nothing to mark", () => {
    expect(searchIndexMarks(makeVectorIndex({ index: makeIndexRecord({ profile: OPENAI_PROFILE }) }))).toEqual([]);
    expect(searchIndexMarks(makeVectorIndex({ index: null }))).toEqual([]);
  });
});

describe("rebuildTarget", () => {
  test("is the model the server would rebuild for", () => {
    expect(rebuildTarget(makeVectorIndex({ reindex_target: "candidate" }))).toBe("candidate");
    expect(rebuildTarget(makeVectorIndex({ reindex_target: "active" }))).toBe("active");
    expect(rebuildTarget(makeVectorIndex())).toBeNull();
  });

  test("is the model in use for an index that does not fit it, which the server validates first when it has to", () => {
    expect(rebuildTarget(makeVectorIndex({ status: "reindex_required" }))).toBe("active");
  });

  test("is not the model in use while it has no index and has not been validated: validation comes first", () => {
    const stored = makeVectorIndex({ status: "awaiting_validation", index: null, reindex_target: "active" });
    const edited = makeVectorIndex({ status: "awaiting_validation", index: null, reindex_target: "candidate" });

    expect(rebuildTarget(stored)).toBeNull();
    expect(rebuildTarget(edited)).toBe("candidate");
  });

  test("is nothing while a rebuild runs or the deployment keeps the index", () => {
    expect(rebuildTarget(makeVectorIndex({ status: "reindexing", reindex: makeReindex() }))).toBeNull();
    expect(rebuildTarget(makeVectorIndex({ status: "not_managed", index: null }))).toBeNull();
  });
});

describe("rebuildProgress", () => {
  test("counts the memories embedded so far, and says that search keeps working", () => {
    const running = makeVectorIndex({ status: "reindexing", reindex: makeReindex() });

    expect(rebuildProgress(running)).toEqual({
      percent: 9,
      text: "120 of 1,204 memories embedded",
      meanwhile:
        "Search keeps working on the current index until the new one is ready. Source syncs and changes to memories wait until the rebuild has finished.",
    });
  });

  test("has no share to show until the memories are counted", () => {
    const starting = makeVectorIndex({ status: "reindexing", reindex: makeReindex({ memories_total: null, memories_indexed: 0 }) });

    expect(rebuildProgress(starting)).toMatchObject({ percent: null, text: "Counting the memories to embed…" });
  });

  test("an installation without memories is done at once", () => {
    const empty = makeVectorIndex({ status: "reindexing", reindex: makeReindex({ memories_total: 0, memories_indexed: 0 }) });

    expect(rebuildProgress(empty)).toMatchObject({ percent: 100, text: "0 of 0 memories embedded" });
  });

  test("without an index in use, search waits for the first one", () => {
    const first = makeVectorIndex({ status: "reindexing", index: null, reindex: makeReindex() });

    expect(rebuildProgress(first)?.meanwhile).toContain("Search ranks by matching words until the index is ready.");
  });

  test("a rebuild that started while vector search was off does not say that search keeps working", () => {
    const off = makeVectorIndex({ status: "reindexing", serves_search: false, reindex: makeReindex({ target: "active" }) });

    expect(rebuildProgress(off)?.meanwhile).toBe(
      "Search ranks by matching words until the index is ready. Source syncs and changes to memories wait until the rebuild has finished.",
    );
  });

  test("is nothing while no rebuild runs", () => {
    expect(rebuildProgress(makeVectorIndex())).toBeNull();
  });
});

describe("a failed rebuild", () => {
  const failed = makeReindex({ state: "failed", error_code: "rate_limited", finished_at: "2026-03-02T09:02:00Z" });

  test("gives the reason, says nothing in use changed, and can be run again", () => {
    const waiting = makeVectorIndex({ reindex_target: "candidate", reindex: failed });

    expect(rebuildFailure(waiting, NOW)).toEqual({
      title: "The rebuild failed 3 min ago",
      body: "The provider is limiting requests right now. Try again in a moment. The model and the index in use are unchanged.",
    });
    expect(rebuildLabel(waiting)).toBe("Run the rebuild again");
  });

  test("is not brought up once there is nothing left to rebuild for", () => {
    const settled = makeVectorIndex({ reindex: failed });

    expect(failedRebuild(settled)).toBeNull();
    expect(rebuildFailure(settled, NOW)).toBeNull();
    expect(rebuildLabel(settled)).toBe("Rebuild search index");
  });

  test("has a reason for every cause, also one the page has no words for", () => {
    expect(rebuildFailureReason("interrupted")).toBe("The server stopped before the rebuild had finished.");
    expect(rebuildFailureReason("memories_changed")).toBe("Memories changed while the index was being rebuilt.");
    expect(rebuildFailureReason("authentication_failed")).toBe("The provider rejected the credentials. Check the API key.");
    expect(rebuildFailureReason("something_new")).toBe("The rebuild did not finish.");
    expect(rebuildFailureReason(null)).toBe("The rebuild did not finish.");
  });
});

test("a rebuild that cannot start says why in the words of the page, or in the server's", () => {
  expect(startRefusalReason(new ApiError(409, "A Source is syncing.", "sync_in_progress"))).toBe(
    "A source is syncing. Start the rebuild once no source is syncing.",
  );
  expect(startRefusalReason(new ApiError(422, "The key cannot be resolved.", "credential_unavailable"))).toBe(
    "The saved API key is not available to the server. Enter it again.",
  );
  expect(startRefusalReason(new ApiError(500, "The database is locked."))).toBe("The database is locked.");
});
