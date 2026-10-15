import { z } from "zod";
import { needsTokenLimits } from "./modelStatus";
import { parameterNames, providersFor, tokenLimitNames } from "./providerFields";
import { endpointOrigin, originName, routeOrigin, type SavedKey } from "./savedKey";
import type {
  AuthStrategy,
  ModelBinding,
  ModelCandidateRequest,
  ModelDiscoveryRequest,
  ModelKind,
  ModelOption,
  ModelProvider,
} from "./types";

export const INVALID_URL_MESSAGE = "Enter a URL starting with http:// or https://.";
export const URL_CREDENTIALS_MESSAGE = "Enter the base URL without a user name or password.";
export const MODEL_REQUIRED_MESSAGE = "Enter a model.";
export const MODEL_ROUTE_MESSAGE = "Enter the model as provider/model.";
export const API_KEY_REQUIRED_MESSAGE = "Enter an API key.";
export const WHOLE_NUMBER_MESSAGE = "Enter a whole number greater than zero.";

/**
 * One model as the user edits it. `apiKey` holds only a key the user typed;
 * a saved key never leaves the server. Extra settings and token limits are
 * kept as the text in their fields, by the name the server gives them.
 */
export interface ModelFormValues {
  provider: string;
  endpoint: string;
  authStrategy: AuthStrategy;
  apiKey: string;
  model: string;
  options: Record<string, string>;
  parameters: Record<string, string>;
  tokenLimits: Record<string, string>;
}

function fieldText(values: Record<string, unknown>): Record<string, string> {
  const text: Record<string, string> = {};
  for (const [name, value] of Object.entries(values)) {
    if (typeof value === "string" || typeof value === "number") text[name] = String(value);
  }
  return text;
}

/** The form for a provider nothing is saved for: its own address and its first way of authenticating. */
export function blankValues(provider: ModelProvider | undefined): ModelFormValues {
  return {
    provider: provider?.id ?? "",
    endpoint: provider?.default_endpoint ?? "",
    authStrategy: provider?.auth_strategies[0] ?? "api_key",
    apiKey: "",
    model: "",
    options: {},
    parameters: {},
    tokenLimits: {},
  };
}

/** The form as the server last stored it, or the first provider of the kind when nothing is stored. */
export function savedValues(
  binding: ModelBinding | null,
  kind: ModelKind,
  providers: readonly ModelProvider[],
): ModelFormValues {
  if (binding === null) return blankValues(providersFor(kind, providers)[0]);
  return {
    provider: binding.provider,
    endpoint: binding.endpoint ?? "",
    authStrategy: binding.auth_strategy,
    apiKey: "",
    model: binding.model_id,
    options: fieldText(binding.options),
    parameters: fieldText(binding.parameters),
    tokenLimits: fieldText(binding.token_limits),
  };
}

/**
 * Whether the form shows the token limit fields for the stored model: it
 * states limits already, or its latest validation found that the server has
 * none for it. LiteLLM knows the limits of most hosted models, so the fields
 * stay out of the way until a model needs them.
 */
export function offersTokenLimits(stored: ModelBinding | null): boolean {
  if (stored === null) return false;
  return Object.keys(stored.token_limits).length > 0 || needsTokenLimits(stored);
}

/** Whether any of the token limit fields holds a value. */
export function hasTokenLimitValues(tokenLimits: Record<string, string>): boolean {
  return Object.values(tokenLimits).some((value) => value.trim() !== "");
}

/** The token limits a listed model comes with, among the fields in `names`, as the text of those fields. */
export function listedTokenLimits(model: ModelOption, names: readonly string[]): Record<string, string> {
  const listed: Record<string, unknown> = { ...model };
  const limits: Record<string, string> = {};
  for (const name of names) {
    const tokens = listed[name];
    if (typeof tokens === "number" && Number.isInteger(tokens) && tokens > 0) limits[name] = String(tokens);
  }
  return limits;
}

function sameFields(a: Record<string, string>, b: Record<string, string>, names: readonly string[]): boolean {
  return names.every((name) => (a[name] ?? "").trim() === (b[name] ?? "").trim());
}

/**
 * The token limit fields after a listed model is chosen, or `null` to leave
 * them as they are. Fields that are empty, or that still hold the limits an
 * earlier choice filled in, take the limits the listing gives for the model;
 * limits the user entered are kept.
 */
export function tokenLimitsForChoice(
  current: Record<string, string>,
  names: readonly string[],
  filledBefore: Record<string, string> | null,
  listed: Record<string, string>,
): Record<string, string> | null {
  const replaceable = !hasTokenLimitValues(trimmedFields(current, names)) || (filledBefore !== null && sameFields(current, filledBefore, names));
  if (!replaceable) return null;
  const next: Record<string, string> = { ...current };
  for (const name of names) next[name] = listed[name] ?? "";
  return sameFields(next, current, names) ? null : next;
}

/** The token limits the form filled in from the model list, and the model they belong to. */
export interface ListedLimits {
  model: string;
  limits: Record<string, string>;
}

/**
 * The token limit fields after the user types `typed` in the Model field, or
 * `null` to leave them as they are. Limits the listing gave for another model
 * are cleared, since they do not describe the model now named; limits the
 * user entered are kept.
 */
export function tokenLimitsForTypedModel(
  current: Record<string, string>,
  names: readonly string[],
  filled: ListedLimits | null,
  typed: string,
): Record<string, string> | null {
  if (filled === null || typed.trim() === filled.model || !sameFields(current, filled.limits, names)) return null;
  const next: Record<string, string> = { ...current };
  for (const name of names) next[name] = "";
  return sameFields(next, current, names) ? null : next;
}

/** The form after choosing a provider: the saved values when it is the saved provider, a blank form otherwise. */
export function valuesForProvider(provider: ModelProvider, saved: ModelFormValues): ModelFormValues {
  return provider.id === saved.provider ? saved : blankValues(provider);
}

/** The base URL to send: none for a blank field, which leaves the provider's own address in place. */
function requestedEndpoint(values: ModelFormValues): string | null {
  const typed = values.endpoint.trim().replace(/\/+$/, "");
  return typed === "" ? null : typed;
}

/**
 * The base URL in the form that cannot be saved with credentials from the
 * server environment, or `null` when there is none. The server saves those
 * credentials only for the provider's own address, so that a base URL entered
 * here never sends them to another server. A model the server read from its
 * environment variables can pair the two; the user then decides which one to
 * give up before anything is saved.
 */
export function environmentEndpointConflict(values: ModelFormValues, provider: ModelProvider): string | null {
  if (values.authStrategy !== "environment") return null;
  const endpoint = requestedEndpoint(values);
  return endpoint === null || endpoint === provider.default_endpoint ? null : endpoint;
}

/** Why a base URL cannot be saved with credentials from the server environment, and the two ways out. */
export function environmentEndpointMessage(provider: ModelProvider): string {
  return provider.default_endpoint === null
    ? "Credentials from the server environment cannot be saved with a base URL. Remove the base URL, or choose another way of authenticating."
    : `Credentials from the server environment go to ${provider.default_endpoint} only. Use that address, or choose another way of authenticating.`;
}

/** The base URL requests would go to, with the provider's own address filled in. */
function effectiveEndpoint(values: ModelFormValues, provider: ModelProvider): string | null {
  return requestedEndpoint(values) ?? provider.default_endpoint;
}

/**
 * The origin a key saved with these values would belong to: that of the base
 * URL, or of the route when the provider has no address of its own.
 */
export function formOrigin(values: ModelFormValues, provider: ModelProvider): string | null {
  const endpoint = effectiveEndpoint(values, provider);
  if (endpoint !== null) return endpointOrigin(endpoint);
  return provider.model_is_route ? routeOrigin(values.model) : null;
}

function typedKey(values: ModelFormValues): string {
  return values.authStrategy === "api_key" ? values.apiKey.trim() : "";
}

function trimmedFields(values: Record<string, string>, names: readonly string[]): Record<string, string> {
  const fields: Record<string, string> = {};
  for (const name of names) {
    const value = (values[name] ?? "").trim();
    if (value !== "") fields[name] = value;
  }
  return fields;
}

/** The filled-in fields among `names` as the numbers the server stores. */
function numberFields(values: Record<string, string>, names: readonly string[]): Record<string, number> {
  const fields: Record<string, number> = {};
  for (const [name, value] of Object.entries(trimmedFields(values, names))) fields[name] = Number(value);
  return fields;
}

/** Everything saving would store, in one comparable shape. A typed key always counts as a change. */
function storedShape(values: ModelFormValues, provider: ModelProvider, kind: ModelKind): string {
  return JSON.stringify([
    values.provider,
    effectiveEndpoint(values, provider),
    values.authStrategy,
    values.model.trim(),
    trimmedFields(values.options, provider.options),
    trimmedFields(values.parameters, parameterNames(provider, kind)),
    trimmedFields(values.tokenLimits, tokenLimitNames(provider, kind)),
  ]);
}

/** Whether the form differs from what the server stored, ignoring surrounding whitespace. */
export function isChanged(
  saved: ModelFormValues,
  current: ModelFormValues,
  provider: ModelProvider | undefined,
  kind: ModelKind,
): boolean {
  if (saved.provider !== current.provider) return true;
  if (provider === undefined) return false;
  return typedKey(current) !== "" || storedShape(saved, provider, kind) !== storedShape(current, provider, kind);
}

/**
 * The model to save, with the base URL exactly as the form shows it. A blank
 * base URL leaves the provider's own address in place, and a blank key keeps
 * the key saved for this origin.
 */
export function candidateRequest(values: ModelFormValues, provider: ModelProvider, kind: ModelKind): ModelCandidateRequest {
  const request: ModelCandidateRequest = {
    provider: provider.id,
    model_id: values.model.trim(),
    endpoint: requestedEndpoint(values),
    auth_strategy: values.authStrategy,
    options: trimmedFields(values.options, provider.options),
    parameters: numberFields(values.parameters, parameterNames(provider, kind)),
    token_limits: numberFields(values.tokenLimits, tokenLimitNames(provider, kind)),
  };
  const key = typedKey(values);
  if (key !== "") request.api_key = key;
  return request;
}

/** The request that lists an endpoint's models, with the typed key or, without one, the saved key of this origin. */
export function discoveryRequest(values: ModelFormValues): ModelDiscoveryRequest {
  return {
    provider: values.provider,
    endpoint: requestedEndpoint(values),
    auth_strategy: values.authStrategy,
    api_key: typedKey(values) || null,
  };
}

/** A field that cannot be saved as it is, by its path in the form. */
export interface FieldProblem {
  path: string[];
  message: string;
}

function endpointProblem(values: ModelFormValues, provider: ModelProvider): string | null {
  if (environmentEndpointConflict(values, provider) !== null) return environmentEndpointMessage(provider);
  const endpoint = requestedEndpoint(values);
  if (endpoint === null) return provider.endpoint_required ? "Enter the base URL of the endpoint." : null;
  if (endpointOrigin(endpoint) === null) return INVALID_URL_MESSAGE;
  const url = new URL(endpoint);
  return url.username !== "" || url.password !== "" ? URL_CREDENTIALS_MESSAGE : null;
}

function modelProblem(values: ModelFormValues, provider: ModelProvider): string | null {
  const model = values.model.trim();
  if (model === "") return MODEL_REQUIRED_MESSAGE;
  if (!provider.model_is_route) return null;
  const [routeProvider, ...rest] = model.split("/");
  return routeProvider === "" || rest.join("/") === "" || model.includes("://") ? MODEL_ROUTE_MESSAGE : null;
}

function apiKeyProblem(values: ModelFormValues, savedKey: SavedKey, origin: string | null): string | null {
  if (values.authStrategy !== "api_key" || typedKey(values) !== "" || savedKey.status === "kept") return null;
  if (savedKey.status !== "other_origin") return API_KEY_REQUIRED_MESSAGE;
  const here = origin === null ? "this endpoint" : originName(origin);
  return `Enter the API key for ${here}. The saved key belongs to ${originName(savedKey.origin)}.`;
}

function isWholeNumber(text: string): boolean {
  return /^\d+$/.test(text) && Number(text) >= 1;
}

export interface FormContext {
  provider: ModelProvider | undefined;
  kind: ModelKind;
  /** The saved key a blank key field would keep for the form's origin. */
  savedKey: SavedKey;
}

/** What has to be corrected before the model can be saved. */
export function fieldProblems(values: ModelFormValues, { provider, kind, savedKey }: FormContext): FieldProblem[] {
  if (provider === undefined) return [{ path: ["provider"], message: "Choose a provider." }];
  const problems: FieldProblem[] = [];
  const add = (path: string[], message: string | null) => {
    if (message !== null) problems.push({ path, message });
  };
  add(["endpoint"], endpointProblem(values, provider));
  add(["apiKey"], apiKeyProblem(values, savedKey, formOrigin(values, provider)));
  add(["model"], modelProblem(values, provider));
  for (const [name, value] of Object.entries(trimmedFields(values.parameters, parameterNames(provider, kind)))) {
    add(["parameters", name], isWholeNumber(value) ? null : WHOLE_NUMBER_MESSAGE);
  }
  for (const [name, value] of Object.entries(trimmedFields(values.tokenLimits, tokenLimitNames(provider, kind)))) {
    add(["tokenLimits", name], isWholeNumber(value) ? null : WHOLE_NUMBER_MESSAGE);
  }
  return problems;
}

const valuesSchema = z.object({
  provider: z.string(),
  endpoint: z.string(),
  authStrategy: z.enum(["api_key", "none", "environment"]),
  apiKey: z.string(),
  model: z.string(),
  options: z.record(z.string(), z.string()),
  parameters: z.record(z.string(), z.string()),
  tokenLimits: z.record(z.string(), z.string()),
});

/** The form schema: every problem `problemsOf` finds is reported on its field. */
export function modelFormSchema(problemsOf: (values: ModelFormValues) => FieldProblem[]) {
  return valuesSchema.superRefine((values, context) => {
    for (const problem of problemsOf(values)) {
      context.addIssue({ code: "custom", path: problem.path, message: problem.message });
    }
  });
}
