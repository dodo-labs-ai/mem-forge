import type { AuthStrategy, ModelKind, ModelProvider } from "./types";

export const AUTH_LABELS: Record<AuthStrategy, string> = {
  api_key: "API key",
  none: "No authentication",
  environment: "Credentials from the server environment",
};

/** What a way of authenticating means, for the ones that have no field of their own to explain it. */
export const AUTH_HINTS: Partial<Record<AuthStrategy, string>> = {
  none: "No key is sent. Use this for a server that does not ask for one.",
  environment:
    "MemForge sends no key of its own. The provider's credentials are read from the environment variables of the MemForge server.",
};

interface ExtraField {
  label: string;
  hint: string;
}

/** The provider settings beyond endpoint, key and model, by the name the server gives them. */
const OPTION_FIELDS: Record<string, ExtraField> = {
  organization: { label: "Organization ID", hint: "Optional. Needed only when the key belongs to more than one organization." },
};

/** The request settings a kind of model accepts, by the name the server gives them. */
const PARAMETER_FIELDS: Record<string, ExtraField> = {
  dimensions: { label: "Output dimensions", hint: "Optional. Leave blank for the model's own vector size." },
};

function fallbackField(name: string): ExtraField {
  const words = name.replaceAll("_", " ");
  return { label: words.charAt(0).toUpperCase() + words.slice(1), hint: "Optional." };
}

export function optionField(name: string): ExtraField {
  return OPTION_FIELDS[name] ?? fallbackField(name);
}

export function parameterField(name: string): ExtraField {
  return PARAMETER_FIELDS[name] ?? fallbackField(name);
}

/** The providers that serve models of `kind`, in the order the server lists them. */
export function providersFor(kind: ModelKind, providers: readonly ModelProvider[]): ModelProvider[] {
  return providers.filter((provider) => provider.kinds.includes(kind));
}

/** The providers of `kind` that can reach a server without sending it a key. */
export function keylessProviders(kind: ModelKind, providers: readonly ModelProvider[]): ModelProvider[] {
  return providersFor(kind, providers).filter((provider) => provider.auth_strategies.includes("none"));
}

/** The request settings a model of `kind` may set with this provider. */
export function parameterNames(provider: ModelProvider, kind: ModelKind): string[] {
  return provider.parameters[kind] ?? [];
}

export interface TextFieldCopy {
  label: string;
  placeholder: string;
  hint: string | null;
}

export interface EndpointFieldCopy extends TextFieldCopy {
  /** True when the provider's own address is the only one allowed, so the field cannot be edited. */
  fixed: boolean;
}

/**
 * How the base URL field reads for a provider. Credentials from the server
 * environment go only to the provider's own address, so the field is fixed
 * while that way of authenticating is chosen.
 */
export function endpointField(provider: ModelProvider, authStrategy: AuthStrategy): EndpointFieldCopy {
  const ownAddress = provider.default_endpoint;
  if (authStrategy === "environment") {
    return {
      label: "Base URL",
      placeholder: "Decided by the route",
      hint:
        ownAddress === null
          ? "With credentials from the server environment, the route decides where requests go."
          : `With credentials from the server environment, requests go to ${ownAddress} only. Choose API key to use another base URL.`,
      fixed: true,
    };
  }
  if (provider.endpoint_required) {
    return { label: "Base URL", placeholder: "https://api.example.com/v1", hint: null, fixed: false };
  }
  if (ownAddress === null) {
    return {
      label: "Base URL (optional)",
      placeholder: "Decided by the route",
      hint: "Leave blank unless the route needs the address of a specific server.",
      fixed: false,
    };
  }
  return { label: "Base URL", placeholder: ownAddress, hint: null, fixed: false };
}

/** How the model field reads for a provider. */
export function modelField(provider: ModelProvider): TextFieldCopy {
  if (provider.model_is_route) {
    return {
      label: "Model route",
      placeholder: "provider/model",
      hint: "The provider and the model, separated by a slash, as LiteLLM names them.",
    };
  }
  return { label: "Model", placeholder: "Model id", hint: null };
}
