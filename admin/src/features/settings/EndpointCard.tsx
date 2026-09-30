import { Loader2, RefreshCw } from "lucide-react";
import { useId } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { errorMessage } from "@/lib/errors";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { useProbeEndpoint } from "./api";
import { ApiKeyField } from "./ApiKeyField";
import { ENVIRONMENT_PLACEHOLDER, ENDPOINT_INPUT_CLASS } from "./constants";
import { FormField } from "./FormField";
import { ModelSuggestions } from "./ModelSuggestions";
import { preferredModel } from "./model/modelSuggestions";
import { probeFailure, probeOutcome } from "./model/probeOutcome";
import { ENDPOINT_LABELS, probeRequest, type SettingsValues } from "./model/settingsForm";
import type { EndpointKind, LlmConfig } from "./model/types";
import { ProbeNotice } from "./ProbeNotice";

const DESCRIPTIONS: Record<EndpointKind, string> = {
  enrichment: "Extracts memories from source content and summarizes agent sessions.",
  embedding: "Turns memories into vectors for search.",
};

interface EndpointCardProps {
  kind: EndpointKind;
  config: LlmConfig;
  readOnly: boolean;
}

/** One model endpoint: where it is, how to authenticate, and which model to call. */
export function EndpointCard({ kind, config, readOnly }: EndpointCardProps) {
  const { register, setValue, getValues, trigger, control, formState } = useFormContext<SettingsValues>();
  const [baseUrl, model] = useWatch({ control, name: [`${kind}.baseUrl`, `${kind}.model`] });
  const probe = useProbeEndpoint();
  const titleId = useId();
  const ids = { baseUrl: `${kind}-base-url`, baseUrlError: `${kind}-base-url-error`, apiKey: `${kind}-api-key`, model: `${kind}-model` };
  const baseUrlError = formState.errors[kind]?.baseUrl?.message;
  const outcome = probe.data ? probeOutcome(probe.data) : probe.isError ? probeFailure(errorMessage(probe.error)) : null;
  const placeholder = (editable: string) => (readOnly ? ENVIRONMENT_PLACEHOLDER : editable);

  async function testConnection() {
    if (!(await trigger(`${kind}.baseUrl`))) return;
    probe.mutate(probeRequest(kind, getValues(kind)), {
      onSuccess: (result) => {
        const suggestion = result.ok ? preferredModel(kind, probeOutcome(result).models) : undefined;
        if (suggestion && getValues(`${kind}.model`).trim() === "") setValue(`${kind}.model`, suggestion);
      },
    });
  }

  function applySuggestedBaseUrl(suggestion: string) {
    setValue(`${kind}.baseUrl`, suggestion, { shouldValidate: true });
    probe.reset();
  }

  return (
    <section aria-labelledby={titleId} className="space-y-4 rounded-xl border bg-card p-5">
      <div className="space-y-0.5">
        <h2 id={titleId} className="text-base font-semibold text-foreground">
          {ENDPOINT_LABELS[kind]}
        </h2>
        <p className="text-sm text-muted-foreground">{DESCRIPTIONS[kind]}</p>
      </div>

      <FormField label="Base URL" htmlFor={ids.baseUrl}>
        <div className="flex items-center gap-2">
          <Input
            id={ids.baseUrl}
            placeholder={placeholder("https://api.example.com/v1")}
            spellCheck={false}
            aria-invalid={baseUrlError ? true : undefined}
            aria-describedby={baseUrlError ? ids.baseUrlError : undefined}
            readOnly={readOnly}
            className={ENDPOINT_INPUT_CLASS}
            {...register(`${kind}.baseUrl`, { onChange: () => probe.reset() })}
          />
          <Button
            type="button"
            variant="outline"
            onClick={testConnection}
            disabled={readOnly || baseUrl.trim() === "" || probe.isPending}
          >
            {probe.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            Test connection
          </Button>
        </div>
        {baseUrlError ? (
          <p id={ids.baseUrlError} className="text-xs text-tone-danger-foreground">
            {baseUrlError}
          </p>
        ) : null}
      </FormField>

      {outcome ? <ProbeNotice outcome={outcome} onUseBaseUrl={applySuggestedBaseUrl} /> : null}

      <FormField label="API key" htmlFor={ids.apiKey}>
        <ApiKeyField
          id={ids.apiKey}
          kind={kind}
          saved={{ set: config[`${kind}_api_key_set`], last4: config[`${kind}_api_key_last4`] ?? null }}
          readOnly={readOnly}
          onKeyChange={() => probe.reset()}
        />
      </FormField>

      <FormField label="Model" htmlFor={ids.model}>
        <Input
          id={ids.model}
          placeholder={placeholder("Model id")}
          spellCheck={false}
          readOnly={readOnly}
          className={ENDPOINT_INPUT_CLASS}
          {...register(`${kind}.model`)}
        />
        {outcome ? (
          <ModelSuggestions
            kind={kind}
            models={outcome.models}
            selected={model}
            onSelect={(choice) => setValue(`${kind}.model`, choice)}
          />
        ) : null}
      </FormField>
    </section>
  );
}
