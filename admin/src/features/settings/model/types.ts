import type { components } from "@/api";

export type { ModelBinding, ModelKind, ModelKindSettings, ModelSettings } from "@/api";

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
