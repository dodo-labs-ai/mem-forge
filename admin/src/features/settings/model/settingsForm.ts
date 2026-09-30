import { z } from "zod";
import type { EndpointKind, LlmConfig, LlmConfigUpdate, LlmProbeRequest } from "./types";

export const ENDPOINT_KINDS: readonly EndpointKind[] = ["enrichment", "embedding"];

export const ENDPOINT_LABELS: Record<EndpointKind, string> = {
  enrichment: "Enrichment",
  embedding: "Embedding",
};

export const INVALID_URL_MESSAGE = "Enter a URL starting with http:// or https://.";

/**
 * One endpoint card as the user edits it. `apiKey` holds only a key the user
 * typed; the saved key never leaves the server in full.
 */
export interface EndpointValues {
  baseUrl: string;
  model: string;
  apiKey: string;
  removeSavedKey: boolean;
}

export type SettingsValues = Record<EndpointKind, EndpointValues>;

/** What saving does to an endpoint's stored API key. */
export type KeyIntent = "keep" | "replace" | "remove";

export function isHttpUrl(value: string): boolean {
  try {
    const { protocol } = new URL(value);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

const endpointSchema = z.object({
  baseUrl: z.string().refine((value) => value.trim() === "" || isHttpUrl(value.trim()), INVALID_URL_MESSAGE),
  model: z.string(),
  apiKey: z.string(),
  removeSavedKey: z.boolean(),
});

export const settingsSchema = z.object({ enrichment: endpointSchema, embedding: endpointSchema });

function endpointValues(config: LlmConfig, kind: EndpointKind): EndpointValues {
  return {
    baseUrl: config[`${kind}_base_url`] ?? "",
    model: config[`${kind}_model`] ?? "",
    apiKey: "",
    removeSavedKey: false,
  };
}

/** The form as the server last stored it. */
export function settingsValues(config: LlmConfig): SettingsValues {
  return { enrichment: endpointValues(config, "enrichment"), embedding: endpointValues(config, "embedding") };
}

/** A typed key replaces the saved one; otherwise the saved key stays unless the user removed it. */
export function keyIntent(endpoint: EndpointValues): KeyIntent {
  if (endpoint.apiKey.trim() !== "") return "replace";
  return endpoint.removeSavedKey ? "remove" : "keep";
}

function endpointChanged(saved: EndpointValues, current: EndpointValues): boolean {
  return (
    saved.baseUrl.trim() !== current.baseUrl.trim() ||
    saved.model.trim() !== current.model.trim() ||
    keyIntent(current) !== "keep"
  );
}

/** The cards whose values differ from what the server stored, ignoring surrounding whitespace. */
export function changedEndpoints(saved: SettingsValues, current: SettingsValues): EndpointKind[] {
  return ENDPOINT_KINDS.filter((kind) => endpointChanged(saved[kind], current[kind]));
}

/** "Enrichment", "Enrichment and Embedding". */
export function endpointList(kinds: readonly EndpointKind[]): string {
  return kinds.map((kind) => ENDPOINT_LABELS[kind]).join(" and ");
}

/** The key the server should test with: the typed one, none, or (`null`) the saved one. */
function requestKey(endpoint: EndpointValues): string | null {
  const intent = keyIntent(endpoint);
  if (intent === "replace") return endpoint.apiKey.trim();
  if (intent === "remove") return "";
  return null;
}

/**
 * The update for the changed cards only. Fields left out keep their stored
 * value, and an API key is sent only when the user replaced or removed it.
 */
export function updateRequest(saved: SettingsValues, current: SettingsValues): LlmConfigUpdate {
  const request: LlmConfigUpdate = {};
  for (const kind of changedEndpoints(saved, current)) {
    const endpoint = current[kind];
    request[`${kind}_base_url`] = endpoint.baseUrl.trim();
    request[`${kind}_model`] = endpoint.model.trim();
    const key = requestKey(endpoint);
    if (key !== null) request[`${kind}_api_key`] = key;
  }
  return request;
}

export function probeRequest(kind: EndpointKind, endpoint: EndpointValues): LlmProbeRequest {
  return { kind, base_url: endpoint.baseUrl.trim(), api_key: requestKey(endpoint) };
}
