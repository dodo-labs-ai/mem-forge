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

/** Marks the label of a field that can be left blank; a field without the mark has to be filled in. */
export function optionalLabel(label: string): string {
  return `${label} (optional)`;
}

interface ExtraField {
  label: string;
  hint: string | null;
}

/** The provider settings beyond endpoint, key and model, by the name the server gives them. Each is optional. */
const OPTION_FIELDS: Record<string, ExtraField> = {
  organization: {
    label: optionalLabel("Organization ID"),
    hint: "Needed only when the key belongs to more than one organization.",
  },
};

/** The request settings a kind of model accepts, by the name the server gives them. Each is optional. */
const PARAMETER_FIELDS: Record<string, ExtraField> = {
  dimensions: { label: optionalLabel("Output dimensions"), hint: "Leave blank for the model's own vector size." },
};

/** The token limits a model may state, by the name the server gives them. Each is a number of tokens. */
const TOKEN_LIMIT_FIELDS: Record<string, ExtraField> = {
  context_window_tokens: {
    label: "Context window",
    hint: "In tokens: the most the model takes in one request, its answer included.",
  },
  max_output_tokens: { label: "Maximum output", hint: "In tokens: the most the model writes in one answer." },
};

function fallbackLabel(name: string): string {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function optionalFallback(name: string): ExtraField {
  return { label: optionalLabel(fallbackLabel(name)), hint: null };
}

export function optionField(name: string): ExtraField {
  return OPTION_FIELDS[name] ?? optionalFallback(name);
}

export function parameterField(name: string): ExtraField {
  return PARAMETER_FIELDS[name] ?? optionalFallback(name);
}

/**
 * A token limit field. The limits are optional while LiteLLM or the server
 * environment has them for the model, and required once validation found that
 * nothing has.
 */
export function tokenLimitField(name: string, { required }: { required: boolean }): ExtraField {
  const field = TOKEN_LIMIT_FIELDS[name] ?? { label: fallbackLabel(name), hint: null };
  return required ? field : { ...field, label: optionalLabel(field.label) };
}

/** What a token limit is called where it is shown rather than edited. */
export function tokenLimitLabel(name: string): string {
  return TOKEN_LIMIT_FIELDS[name]?.label ?? fallbackLabel(name);
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

/** The token limits a model of `kind` may state with this provider. */
export function tokenLimitNames(provider: ModelProvider, kind: ModelKind): string[] {
  return provider.token_limits[kind] ?? [];
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
 * while that way of authenticating is chosen. A provider with an address of
 * its own uses it for a blank field, so the field is optional and says so.
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
    return {
      label: "Base URL",
      placeholder: "https://api.example.com/v1",
      hint: "The address the API's paths start from. A /chat/completions or /models path at its end is left out.",
      fixed: false,
    };
  }
  if (ownAddress === null) {
    return {
      label: optionalLabel("Base URL"),
      placeholder: "Decided by the route",
      hint: "Leave blank unless the route needs the address of a specific server.",
      fixed: false,
    };
  }
  return {
    label: optionalLabel("Base URL"),
    placeholder: ownAddress,
    hint: `Defaults to the ${provider.label} API at ${ownAddress}; change it only for a proxy or another server.`,
    fixed: false,
  };
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
  return {
    label: "Model",
    placeholder: "Model id",
    hint: provider.lists_models ? "Type a model id, or load the provider's models and type here to filter them." : null,
  };
}
