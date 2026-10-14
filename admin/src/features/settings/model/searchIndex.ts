import { ApiError } from "@/api";
import { errorMessage } from "@/lib/errors";
import { formatCount, formatRelative, pluralize } from "@/lib/format";
import { knownFailureReason, providerLabel } from "./modelStatus";
import type { EmbeddingProfile, ModelBinding, ModelKindSettings, ModelProvider, Reindex, StatusSummary, VectorIndex } from "./types";

/**
 * The search index holds one vector for every memory, and all of them come
 * from one embedding model: vectors of two models cannot be compared. So the
 * index either fits the model in use or has to be rebuilt, which embeds every
 * memory again. This module says, in the words of the Settings page, which of
 * the two it is and what a rebuild is doing.
 */

type Providers = ReadonlyArray<Pick<ModelProvider, "id" | "label">>;

/** "Ollama nomic-embed-text": an embedding model as a sentence names it. */
function modelName(model: Pick<EmbeddingProfile, "provider" | "model_id">, providers: Providers): string {
  return `${providerLabel(model.provider, providers)} ${model.model_id}`;
}

/** The model the index in use was built with, or `null` when the server did not record it. */
export function builtWith(vectorIndex: VectorIndex, providers: Providers): string | null {
  const profile = vectorIndex.index?.profile ?? null;
  return profile === null ? null : modelName(profile, providers);
}

/**
 * Which model a rebuild started now would be for: the saved edit that waits
 * for the index, or the model in use. `null` when there is nothing to rebuild
 * for, a rebuild is running, or the deployment keeps the index by its own
 * means. An index of another model than the one in use can always be rebuilt:
 * the server validates the model first when it has to. While there is no
 * index yet, the model in use is validated before anything is rebuilt for it:
 * its validation shows whether the vectors stored so far fit as they are.
 */
export function rebuildTarget(vectorIndex: VectorIndex): "active" | "candidate" | null {
  const { status, reindex_target: target } = vectorIndex;
  if (status === "reindexing" || status === "not_managed") return null;
  if (status === "awaiting_validation") return target === "candidate" ? target : null;
  if (target != null) return target;
  return status === "reindex_required" ? "active" : null;
}

/** Whether the latest rebuild failed and one can be started again. */
export function failedRebuild(vectorIndex: VectorIndex): Reindex | null {
  const reindex = vectorIndex.reindex;
  return reindex?.state === "failed" && rebuildTarget(vectorIndex) !== null ? reindex : null;
}

/** "Run the rebuild again" after a failure, "Rebuild search index" otherwise. */
export function rebuildLabel(vectorIndex: VectorIndex): string {
  return failedRebuild(vectorIndex) === null ? "Rebuild search index" : "Run the rebuild again";
}

const WITHOUT_VECTORS = "Search ranks by matching words, and new memories cannot be created, until the index is rebuilt.";

/** "vectors of 768 numbers", or the size a model answers with when none is asked for. */
function vectorSize(dimensions: number | null | undefined): string {
  return dimensions == null ? "vectors of the model's own size" : `vectors of ${formatCount(dimensions)} numbers`;
}

/** One base URL as every spelling of it reads: without a trailing slash, in lower case. */
function sameAddress(left: string | null, right: string | null): boolean {
  const written = (address: string | null) => (address ?? "").trim().replace(/\/+$/, "").toLowerCase();
  return written(left) === written(right);
}

/** The output size a model's requests ask for, when its settings name one. */
function requestedDimensions(model: ModelBinding): number | null {
  if (model.profile != null) return model.profile.requested_dimensions;
  const dimensions = model.parameters["dimensions"];
  return typeof dimensions === "number" ? dimensions : null;
}

/**
 * What sets the model in use apart from the one that built the index when
 * both carry one name: the address it is reached at, the vector size it is
 * asked for, or the size it answers with. An index is bound to all three.
 */
function sameNameDifference(built: EmbeddingProfile, inUse: ModelBinding, name: string): string {
  const address = inUse.profile?.endpoint ?? inUse.endpoint;
  if (!sameAddress(built.endpoint, address) && built.endpoint !== null && address !== null) {
    return `This index was built with ${name} at ${built.endpoint}, and MemForge now reaches that model at ${address}.`;
  }
  const requested = requestedDimensions(inUse);
  if (built.requested_dimensions !== requested) {
    return `This index was built with ${name} asked for ${vectorSize(built.requested_dimensions)}, and MemForge now asks it for ${vectorSize(requested)}.`;
  }
  const returned = inUse.profile?.dimension ?? null;
  if (returned !== null && returned !== built.dimension) {
    return `This index holds ${vectorSize(built.dimension)} from ${name}, and that model now returns ${vectorSize(returned)}.`;
  }
  return `This index was built with other settings of ${name} than the ones MemForge now uses.`;
}

function offDetail(vectorIndex: VectorIndex, inUse: ModelBinding | null, providers: Providers): string {
  const built = builtWith(vectorIndex, providers);
  const used = inUse === null ? "the embedding model in use" : modelName(inUse, providers);
  if (vectorIndex.index === null) return `The vectors stored so far do not fit ${used}. ${WITHOUT_VECTORS}`;
  const profile = vectorIndex.index.profile ?? null;
  if (profile !== null && inUse !== null && built === used) {
    return `${sameNameDifference(profile, inUse, used)} ${WITHOUT_VECTORS}`;
  }
  return `This index was built with ${built ?? "another embedding model"}, and MemForge now uses ${used}. ${WITHOUT_VECTORS}`;
}

/**
 * The state of the search index in plain words, or `null` when there is
 * nothing to say about it: no embedding model is set up, or the deployment
 * keeps the index by its own means.
 */
export function searchIndexSummary(
  vectorIndex: VectorIndex,
  embedding: ModelKindSettings,
  providers: Providers,
  now: Date = new Date(),
): StatusSummary | null {
  const { status, index, reindex } = vectorIndex;
  if (status === "not_managed" || status === "not_configured") return null;
  if (status === "awaiting_validation") {
    return {
      tone: "warn",
      headline: "No search index yet.",
      detail: "MemForge builds it once the embedding model has passed validation.",
    };
  }
  if (status === "reindexing") {
    const target = reindex?.profile ?? null;
    return {
      tone: "live",
      headline: "Rebuilding the search index.",
      detail: target === null ? "Every memory is embedded again." : `Every memory is embedded again with ${modelName(target, providers)}.`,
    };
  }
  if (status === "reindex_required") {
    return { tone: "danger", headline: "Vector search is off.", detail: offDetail(vectorIndex, embedding.active, providers) };
  }
  const rebuilt =
    index !== null && reindex?.state === "succeeded" && reindex.fingerprint === index.fingerprint && reindex.finished_at
      ? reindex.finished_at
      : null;
  return {
    tone: "ok",
    headline: "Vector search is on.",
    detail: rebuilt === null ? null : `The index was rebuilt ${formatRelative(rebuilt, now)}.`,
  };
}

export interface SearchIndexMark {
  /** A few words that name what is special about the index. */
  label: string;
  /** What it means for the user. */
  text: string;
}

/** The index holds vectors that were in place before MemForge recorded which model builds an index. */
const CARRIED_OVER: SearchIndexMark = {
  label: "Carried over",
  text: "These vectors were stored before MemForge recorded which model builds an index, so it takes them to come from the model in use. Rebuild the index if another model made them.",
};

/** The similarity thresholds were tuned on another embedding model than this one. */
export const THRESHOLDS_NOT_CALIBRATED: SearchIndexMark = {
  label: "Thresholds not calibrated",
  text: "MemForge's thresholds for recognizing duplicates and near matches were tuned on another model. They apply unchanged, so expect duplicate detection to be somewhat more or less strict with this one.",
};

/**
 * What the user should know about the index beyond its state, while search
 * reads it. An index of another model than the one in use is not read, also
 * not while it is being replaced, so there is nothing to say about its
 * vectors.
 */
export function searchIndexMarks(vectorIndex: VectorIndex): SearchIndexMark[] {
  const index = vectorIndex.index;
  if (index === null || !vectorIndex.serves_search) return [];
  const marks: SearchIndexMark[] = [];
  if (index.adopted) marks.push(CARRIED_OVER);
  if (index.profile?.thresholds_calibrated === false) marks.push(THRESHOLDS_NOT_CALIBRATED);
  return marks;
}

export interface RebuildProgress {
  /** How far the rebuild is, from 0 to 100; `null` until the memories are counted. */
  percent: number | null;
  /** "120 of 1,204 memories embedded". */
  text: string;
  /** What keeps working and what waits while the rebuild runs. */
  meanwhile: string;
}

const WAITING_MEANWHILE = "Source syncs and changes to memories wait until the rebuild has finished.";

/** The progress of the running rebuild, or `null` when none is running. */
export function rebuildProgress(vectorIndex: VectorIndex): RebuildProgress | null {
  if (vectorIndex.status !== "reindexing") return null;
  const reindex = vectorIndex.reindex;
  const total = reindex?.memories_total ?? null;
  const indexed = reindex?.memories_indexed ?? 0;
  // A rebuild that started while vector search was off leaves it off until it has switched.
  const searching = vectorIndex.serves_search
    ? "Search keeps working on the current index until the new one is ready."
    : "Search ranks by matching words until the index is ready.";
  return {
    percent: total === null ? null : total === 0 ? 100 : Math.min(100, Math.floor((indexed / total) * 100)),
    text: total === null ? "Counting the memories to embed…" : `${formatCount(indexed)} of ${pluralize(total, "memory", "memories")} embedded`,
    meanwhile: `${searching} ${WAITING_MEANWHILE}`,
  };
}

/** Why a rebuild failed, by the code the server records for causes of its own. */
const REBUILD_FAILURES: Record<string, string> = {
  embedding_dimension_mismatch: "The model returned vectors of another size than when it was validated. Validate the model again.",
  vector_count_mismatch: "The new index did not hold one vector for every memory.",
  vector_invalid: "The model returned a vector MemForge cannot use.",
  memories_changed: "Memories changed while the index was being rebuilt.",
  candidate_changed: "The embedding model settings were saved again while the index was being rebuilt.",
  reindex_lease_lost: "The rebuild stopped reporting and was given up.",
  interrupted: "The server stopped before the rebuild had finished.",
  index_error: "The vector store reported an error.",
};

/** Why the rebuild failed. A failed call to the model has the reason a failed validation has. */
export function rebuildFailureReason(errorCode: string | null | undefined): string {
  return (errorCode && REBUILD_FAILURES[errorCode]) || knownFailureReason(errorCode) || "The rebuild did not finish.";
}

export interface RebuildFailure {
  title: string;
  /** The reason, and that the model and index in use are as they were. */
  body: string;
}

/** The failed rebuild that can be run again, in plain words; `null` when there is none. */
export function rebuildFailure(vectorIndex: VectorIndex, now: Date = new Date()): RebuildFailure | null {
  const failed = failedRebuild(vectorIndex);
  if (failed === null) return null;
  const when = failed.finished_at ?? failed.started_at;
  return {
    title: when ? `The rebuild failed ${formatRelative(when, now)}` : "The rebuild failed",
    body: `${rebuildFailureReason(failed.error_code)} The model and the index in use are unchanged.`,
  };
}

/** Why a rebuild could not start, by the code the server refuses it with. */
const START_REFUSALS: Record<string, string> = {
  sync_in_progress: "A source is syncing. Start the rebuild once no source is syncing.",
  reindex_not_required: "The search index already fits the embedding model in use.",
  embedding_not_validated: "The embedding model has to pass validation before the search index is rebuilt for it.",
  reindex_not_available: "The search index of this deployment is not rebuilt from here.",
};

/** The message for a rebuild that could not start. */
export function startRefusalReason(error: unknown): string {
  const code = error instanceof ApiError ? error.code : null;
  return (code && (START_REFUSALS[code] ?? knownFailureReason(code))) || errorMessage(error);
}
