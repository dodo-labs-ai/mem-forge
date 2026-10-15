import type { components } from "@/api";
import type { Tone } from "@/patterns";

export type { ModelBinding, ModelKind, ModelKindSettings, ModelSettings, VectorIndex } from "@/api";

type Schemas = components["schemas"];

/** One way of reaching a model, with the fields it needs. */
export type ModelProvider = Schemas["ModelProviderResponse"];
/** How requests to a model are authenticated. */
export type AuthStrategy = Schemas["ModelBindingResponse"]["auth_strategy"];
export type ModelState = Schemas["ModelBindingResponse"]["state"];
/** The latest validation of a model. */
export type ModelValidation = Schemas["ModelValidationResult"];
/** The outcome of validating a model, with the settings it left behind. */
export type ModelValidationOutcome = Schemas["ModelValidationResponse"];
export type ModelCandidateRequest = Schemas["ModelCandidateRequest"];
export type ModelDiscoveryRequest = Schemas["ModelDiscoveryRequest"];
export type ModelDiscoveryResult = Schemas["ModelDiscoveryResponse"];
/** One model an endpoint listed, with the token limits the listing gives for it. */
export type ModelOption = Schemas["ModelOption"];
/** The vector space of an embedding model: its settings and the size of its vectors. */
export type EmbeddingProfile = Schemas["EmbeddingProfileResponse"];
/** The latest rebuild of the search index, running or ended. */
export type Reindex = Schemas["VectorReindexResponse"];

/** One state in plain words. */
export interface StatusSummary {
  tone: Tone;
  /** What was found, and when. */
  headline: string;
  /** What the user should know or do about it. */
  detail: string | null;
}
