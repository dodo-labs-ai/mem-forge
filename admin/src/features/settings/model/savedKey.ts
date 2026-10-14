import { KEY_VARIABLES } from "./kinds";
import type { ModelBinding, ModelKind, ModelKindSettings } from "./types";

const DEFAULT_PORTS: Record<string, string> = { "http:": "80", "https:": "443" };

/** The origin of a model that has no base URL: its requests go wherever its route's provider decides. */
const ROUTE_ORIGIN_PREFIX = "provider:";

/**
 * The scheme, host and port a base URL addresses, written the way the server
 * writes the origin of a saved key: `https://api.example.com`, with the port
 * only when it is not the scheme's own. `null` for anything that is not an
 * HTTP(S) URL.
 */
export function endpointOrigin(url: string): string | null {
  let parsed: URL;
  try {
    parsed = new URL(url.trim());
  } catch {
    return null;
  }
  const defaultPort = DEFAULT_PORTS[parsed.protocol];
  const host = parsed.hostname.replace(/\.$/, "");
  if (defaultPort === undefined || host === "") return null;
  const port = parsed.port === "" || parsed.port === defaultPort ? "" : `:${parsed.port}`;
  return `${parsed.protocol}//${host}${port}`;
}

/** The origin of a model whose route, not a base URL, decides where requests go. */
export function routeOrigin(route: string): string | null {
  const provider = route.trim().split("/")[0] ?? "";
  return provider === "" ? null : `${ROUTE_ORIGIN_PREFIX}${provider}`;
}

/** An origin as a sentence names it. */
export function originName(origin: string): string {
  return origin.startsWith(ROUTE_ORIGIN_PREFIX)
    ? `${origin.slice(ROUTE_ORIGIN_PREFIX.length)} models without a base URL`
    : origin;
}

/**
 * What saving does about the API key when the field is left blank. A saved
 * key belongs to the origin it was saved for and is never sent to another.
 */
export type SavedKey =
  /** A key saved for this origin stays in use. */
  | { status: "kept"; source: "saved" | "environment"; last4: string | null }
  /** The model at this origin names a key the server does not have, so one has to be entered. */
  | { status: "unavailable"; source: "saved" | "environment"; kind: ModelKind }
  /** The only saved keys belong to other origins, so this one needs its own. */
  | { status: "other_origin"; origin: string }
  | { status: "none" };

function holdsKey(binding: ModelBinding | null): binding is ModelBinding {
  return binding !== null && binding.api_key.source != null;
}

/**
 * The saved key a model at `origin` would use. The edit that waits to be
 * validated is asked first, then the model in use, as the server does.
 */
export function savedKeyFor(settings: ModelKindSettings, origin: string | null): SavedKey {
  const holders = [settings.candidate, settings.active].filter(holdsKey);
  const holder = holders.find((binding) => binding.origin === origin);
  if (holder) {
    const source = holder.api_key.source ?? "saved";
    return holder.api_key.is_set
      ? { status: "kept", source, last4: holder.api_key.last4 ?? null }
      : { status: "unavailable", source, kind: holder.kind };
  }
  const [first] = holders;
  return first ? { status: "other_origin", origin: first.origin } : { status: "none" };
}

/** "****a91f": enough to recognise a key without revealing it. */
export function maskedKey(last4: string): string {
  return `****${last4}`;
}

export interface ApiKeyStatus {
  message: string;
  /** True when the model cannot be saved until a key is entered. */
  warning: boolean;
}

function keyName(savedKey: Extract<SavedKey, { status: "kept" }>): string {
  const name = savedKey.source === "environment" ? "key from the server environment" : "saved key";
  return savedKey.last4 ? `${name} (${maskedKey(savedKey.last4)})` : name;
}

/** What saving will do about the API key, as the line under the field says it. */
export function apiKeyStatus(savedKey: SavedKey, hasTypedKey: boolean): ApiKeyStatus {
  if (savedKey.status === "kept") {
    const name = keyName(savedKey);
    return {
      message: hasTypedKey ? `Takes the place of the ${name} when you save.` : `The ${name} is used. Leave blank to keep it.`,
      warning: false,
    };
  }
  if (hasTypedKey || savedKey.status === "none") {
    return { message: "Stored encrypted on the server and never shown again.", warning: false };
  }
  if (savedKey.status === "unavailable") {
    return {
      message:
        savedKey.source === "environment"
          ? `${KEY_VARIABLES[savedKey.kind]} is not set on the server. Enter a key here, or set the variable and restart the server.`
          : "The saved key is not available to the server. Enter it again.",
      warning: true,
    };
  }
  return {
    message: `The saved key belongs to ${originName(savedKey.origin)} and is not sent anywhere else. Enter the key for this endpoint.`,
    warning: true,
  };
}
