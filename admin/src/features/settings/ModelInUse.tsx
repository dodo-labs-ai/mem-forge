import { Loader2, RefreshCw } from "lucide-react";
import { useId } from "react";
import { errorMessage } from "@/lib/errors";
import { formatCount } from "@/lib/format";
import { ErrorNotice, PropertyList, type Property } from "@/patterns";
import { Button } from "@/ui/button";
import { useValidateActive } from "./api";
import { providerLabel, validationFacts, validationNotes, validationSummary } from "./model/modelStatus";
import { AUTH_LABELS, tokenLimitLabel } from "./model/providerFields";
import { maskedKey } from "./model/savedKey";
import type { ModelBinding, ModelKind, ModelProvider } from "./model/types";
import { StatusLine } from "./StatusLine";

const WITHOUT_A_MODEL: Record<ModelKind, string> = {
  generation: "No generation model is in use. A source sync stores content but writes no memories until one is.",
  embedding: "No embedding model is in use. Search ranks by matching words until one is.",
};

const KEY_SOURCES = { saved: "Saved in Settings", environment: "From the server environment" } as const;

function Code({ children }: { children: string }) {
  return <span className="font-mono text-[13px] break-all">{children}</span>;
}

function authenticationProperty(model: ModelBinding): Property {
  if (model.auth_strategy !== "api_key") return { label: "Authentication", value: AUTH_LABELS[model.auth_strategy] };
  const { is_set: isSet, last4, source } = model.api_key;
  return {
    label: "API key",
    value: !isSet ? "Not available to the server" : last4 ? <Code>{maskedKey(last4)}</Code> : "Set",
    hint: source ? KEY_SOURCES[source] : undefined,
  };
}

/** The settings and the observations of the latest validation, as labeled values. */
function modelProperties(model: ModelBinding, providers: readonly ModelProvider[]): Property[] {
  const properties: Property[] = [
    { label: "Provider", value: providerLabel(model.provider, providers) },
    { label: "Model", value: <Code>{model.model_id}</Code> },
  ];
  if (model.endpoint) properties.push({ label: "Base URL", value: <Code>{model.endpoint}</Code> });
  properties.push(authenticationProperty(model));
  for (const [name, tokens] of Object.entries(model.token_limits)) {
    properties.push({ label: tokenLimitLabel(name), value: `${formatCount(tokens)} tokens` });
  }
  const facts = model.validation?.passed ? validationFacts(model.validation) : {};
  if (facts.dimension !== undefined) properties.push({ label: "Vector size", value: facts.dimension });
  if (facts.latency_ms !== undefined) properties.push({ label: "Response time", value: `${facts.latency_ms} ms` });
  return properties;
}

interface ModelInUseProps {
  kind: ModelKind;
  /** The model the server uses, or `null` when the kind has none. */
  model: ModelBinding | null;
  providers: readonly ModelProvider[];
}

/**
 * The model MemForge is using and what its latest validation found. The model
 * stays in use whatever a validation finds, so validating again is always safe.
 */
export function ModelInUse({ kind, model, providers }: ModelInUseProps) {
  const validate = useValidateActive(kind);
  const headingId = useId();

  return (
    <div role="group" aria-labelledby={headingId} className="space-y-3 rounded-lg bg-surface-subtle p-3.5">
      <h3 id={headingId} className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
        In use
      </h3>
      {model === null ? (
        <p className="text-sm text-subtle-foreground">{WITHOUT_A_MODEL[kind]}</p>
      ) : (
        <>
          <PropertyList layout="row" items={modelProperties(model, providers)} />
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div role="status" className="min-w-0 flex-1 space-y-1.5">
              <StatusLine summary={validationSummary(model, "active")} />
              {validationNotes(kind, model.validation).map((note) => (
                <p key={note} className="text-xs text-tone-warn-foreground">
                  {note}
                </p>
              ))}
              {model.config_source === "environment" ? (
                <p className="text-xs text-muted-foreground">
                  Set by the MEMFORGE_* environment variables of the server. A model saved here takes their place.
                </p>
              ) : null}
            </div>
            <Button type="button" variant="outline" size="sm" onClick={() => validate.mutate()} disabled={validate.isPending}>
              {validate.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              {validate.isPending ? "Validating…" : "Validate again"}
            </Button>
          </div>
          {validate.isError ? <ErrorNotice title="Couldn't validate" message={errorMessage(validate.error)} /> : null}
        </>
      )}
    </div>
  );
}
