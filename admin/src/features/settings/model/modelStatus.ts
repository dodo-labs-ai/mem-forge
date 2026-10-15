import type { ModelValidationFacts } from "@/api";
import { formatRelative } from "@/lib/format";
import type { Tone } from "@/patterns";
import { KEY_VARIABLES } from "./kinds";
import type { ModelBinding, ModelKind, ModelValidation, StatusSummary } from "./types";

export interface StateBadge {
  label: string;
  tone: Tone;
}

const STATE_BADGES: Record<ModelBinding["state"], StateBadge> = {
  ready: { label: "Ready", tone: "ok" },
  not_validated: { label: "Not validated", tone: "warn" },
  needs_revalidation: { label: "Needs revalidation", tone: "warn" },
};

/** The state of the model in use, or that none is. */
export function stateBadge(active: ModelBinding | null): StateBadge {
  return active === null ? { label: "Not set up", tone: "idle" } : STATE_BADGES[active.state];
}

/** Why a validation failed, by the code the server reports. */
const FAILURE_REASONS: Record<string, string> = {
  authentication_failed: "The provider rejected the credentials. Check the API key.",
  model_not_found: "The provider does not know this model. Check the model name.",
  rate_limited: "The provider is limiting requests right now. Try again in a moment.",
  request_rejected: "The provider rejected the request. The model may not support what MemForge asks of it.",
  timeout: "The provider did not answer in time.",
  provider_unreachable: "MemForge could not reach the endpoint. Check the base URL, and that the MemForge server can reach it.",
  provider_error: "The provider answered with an error. Try again later.",
  structured_output_invalid:
    "The model did not return the structured answer MemForge asked for. Choose a model that returns JSON reliably.",
  embedding_response_invalid: "The endpoint did not return usable vectors. Check that this is an embedding model.",
  embedding_dimension_mismatch: "The model returned vectors of another size than the output dimensions set here.",
  embedding_dimension_unstable: "The model returned vectors of different sizes from one request to the next.",
  parameter_not_supported_by_route: "This provider does not take the output dimensions setting. Clear it and validate again.",
  token_limits_required:
    "The model answered, but MemForge has no token limits for it. Enter its context window and maximum output, then validate again.",
  credential_unavailable: "The saved API key is not available to the server. Enter it again.",
  api_key_origin_changed: "The saved API key belongs to another endpoint. Enter the API key for this one.",
  not_configured: "The model settings are incomplete.",
};

/** The reason for a failure the page has words for, or `undefined` for a code it does not know. */
export function knownFailureReason(errorCode: string | null | undefined): string | undefined {
  return errorCode ? FAILURE_REASONS[errorCode] : undefined;
}

export function failureReason(errorCode: string | null | undefined): string {
  return knownFailureReason(errorCode) ?? "The validation request failed.";
}

/**
 * Why the latest validation of `binding` failed. What the server reads from
 * its environment is named by its variables: the API key when that is unset,
 * and the token limits of a model the deployment manages, which has no form.
 */
function failureReasonOf(binding: ModelBinding): string {
  const errorCode = binding.validation?.error_code;
  if (errorCode === "credential_unavailable" && binding.api_key.source === "environment") {
    return `The server reads the API key of this model from ${KEY_VARIABLES[binding.kind]}, which is not set.`;
  }
  if (errorCode === "token_limits_required" && binding.config_source === "deployment") {
    return "The model answered, but MemForge has no token limits for it. Set MEMFORGE_LLM_MAX_INPUT_TOKENS, MEMFORGE_LLM_CONTEXT_WINDOW_TOKENS and MEMFORGE_LLM_MAX_OUTPUT_TOKENS on the server.";
  }
  return failureReason(errorCode);
}

/**
 * Whether the latest validation of `binding` failed only because the server
 * has no token limits for its model. The model answered, so entering the
 * limits is the one step left, not a fault to fix.
 */
export function awaitsTokenLimits(binding: ModelBinding): boolean {
  const validation = binding.validation;
  return validation !== null && !validation.passed && validation.error_code === "token_limits_required";
}

/**
 * Whether the latest validation of `binding` found that the server has no
 * token limits for its model, so that the model needs them stated.
 */
export function needsTokenLimits(binding: ModelBinding): boolean {
  const validation = binding.validation;
  if (validation === null) return false;
  return validation.error_code === "token_limits_required" || validationFacts(validation).token_limits_known === false;
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** The observations of a validation the page shows, read from the free-form object the server sends. */
export function validationFacts(validation: ModelValidation): ModelValidationFacts {
  const raw = validation.facts ?? {};
  const facts: ModelValidationFacts = {};
  if (isNumber(raw.latency_ms)) facts.latency_ms = raw.latency_ms;
  if (isNumber(raw.dimension)) facts.dimension = raw.dimension;
  if (typeof raw.schema_fallback === "boolean") facts.schema_fallback = raw.schema_fallback;
  if (typeof raw.token_limits_known === "boolean") facts.token_limits_known = raw.token_limits_known;
  if (typeof raw.provider_model === "string" && raw.provider_model !== "") facts.provider_model = raw.provider_model;
  return facts;
}

const NEVER_VALIDATED: Record<"active" | "candidate", StatusSummary> = {
  active: {
    tone: "warn",
    headline: "Not validated yet.",
    detail: "MemForge uses this model as it is, and validates it in the background after the server starts.",
  },
  candidate: { tone: "warn", headline: "Saved, but not validated yet.", detail: null },
};

/**
 * The latest validation of a model in plain words. A model in use stays in
 * use whatever the result; an edit is put to use only once it passes.
 */
export function validationSummary(binding: ModelBinding, role: "active" | "candidate", now: Date = new Date()): StatusSummary {
  const validation = binding.validation;
  if (validation === null) return NEVER_VALIDATED[role];
  const when = formatRelative(validation.checked_at, now);
  if (awaitsTokenLimits(binding)) {
    const reason = failureReasonOf(binding);
    return {
      tone: "warn",
      headline: `One more step: ${binding.config_source === "deployment" ? "set" : "enter"} the token limits.`,
      detail: role === "active" ? `${reason} MemForge keeps using this model.` : reason,
    };
  }
  if (!validation.passed) {
    const reason = failureReasonOf(binding);
    return {
      tone: "danger",
      headline: `Validation failed ${when}.`,
      detail: role === "active" ? `${reason} MemForge keeps using this model.` : reason,
    };
  }
  if (!validation.current || binding.state !== "ready") {
    return {
      tone: "warn",
      headline: `Passed validation ${when}, before the latest change.`,
      detail: "The settings or the installed model library changed since then. Validate again.",
    };
  }
  return { tone: "ok", headline: `Passed validation ${when}.`, detail: null };
}

/** Consequences of a passed validation that the user should know. */
export function validationNotes(kind: ModelKind, validation: ModelValidation | null): string[] {
  if (validation === null || !validation.passed || kind !== "generation") return [];
  return validationFacts(validation).schema_fallback === true
    ? ["The model does not accept a JSON schema, so MemForge asks it for JSON as text and checks the answer itself."]
    : [];
}

/** "OpenAI", or the raw id for a provider this server no longer lists. */
export function providerLabel(providerId: string, providers: ReadonlyArray<{ id: string; label: string }>): string {
  return providers.find((provider) => provider.id === providerId)?.label ?? providerId;
}
