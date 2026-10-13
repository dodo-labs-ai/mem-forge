import { zodResolver } from "@hookform/resolvers/zod";
import { Check, CircleCheck, ListRestart, Loader2, Undo2 } from "lucide-react";
import { useEffect, useId, useMemo } from "react";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import { errorMessage } from "@/lib/errors";
import { ErrorNotice, Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { useDiscardCandidate, useDiscoverModels, useSaveCandidate, useValidateCandidate } from "./api";
import { ApiKeyField } from "./ApiKeyField";
import { describedBy } from "./describedBy";
import { DiscoveryNotice } from "./DiscoveryNotice";
import { EndpointConflictNotice } from "./EndpointConflictNotice";
import { FormField } from "./FormField";
import { discoveryFailure, discoveryOutcome } from "./model/discoveryOutcome";
import {
  candidateRequest,
  discoveryRequest,
  environmentEndpointConflict,
  fieldProblems,
  formOrigin,
  isChanged,
  modelFormSchema,
  savedValues,
  valuesForProvider,
  type ModelFormValues,
} from "./model/modelForm";
import { preferredModel } from "./model/modelSuggestions";
import {
  AUTH_HINTS,
  AUTH_LABELS,
  endpointField,
  keylessProviders,
  modelField,
  optionField,
  parameterField,
  parameterNames,
  providersFor,
} from "./model/providerFields";
import { savedKeyFor } from "./model/savedKey";
import type { AuthStrategy, ModelKind, ModelKindSettings, ModelProvider } from "./model/types";
import { ModelSuggestions } from "./ModelSuggestions";
import { PendingEditNotice } from "./PendingEditNotice";

const INPUT_CLASS = "font-mono read-only:bg-surface-subtle";

interface ModelFormProps {
  kind: ModelKind;
  title: string;
  settings: ModelKindSettings;
  providers: readonly ModelProvider[];
  /** Tells the page whether the form holds changes that are not saved. */
  onUnsavedChange: (kind: ModelKind, unsaved: boolean) => void;
}

/**
 * The form that changes one model. It edits the saved edit when there is one
 * and the model in use otherwise. Its one action saves the form and validates
 * what was saved with a real request; only an edit that passes is put to use.
 */
export function ModelForm({ kind, title, settings, providers, onUnsavedChange }: ModelFormProps) {
  const { active, candidate } = settings;
  const stored = candidate ?? active;
  const saved = useMemo(() => savedValues(stored, kind, providers), [stored, kind, providers]);
  const choices = providersFor(kind, providers);
  const findProvider = (id: string) => choices.find((choice) => choice.id === id);
  const savedKeyOf = (values: ModelFormValues, provider: ModelProvider | undefined) =>
    savedKeyFor(settings, provider ? formOrigin(values, provider) : null);

  const form = useForm<ModelFormValues>({
    defaultValues: saved,
    resolver: zodResolver(
      modelFormSchema((values) => {
        const provider = findProvider(values.provider);
        return fieldProblems(values, { provider, kind, savedKey: savedKeyOf(values, provider) });
      }),
    ),
  });
  const { register, setValue, getValues, trigger, reset, control, formState } = form;
  const [providerId, endpoint, authStrategy, apiKey, model, options, parameters] = useWatch({
    control,
    name: ["provider", "endpoint", "authStrategy", "apiKey", "model", "options", "parameters"],
  });
  const values: ModelFormValues = { provider: providerId, endpoint, authStrategy, apiKey, model, options, parameters };
  const provider = findProvider(providerId);
  const changed = isChanged(saved, values, provider, kind);

  const save = useSaveCandidate(kind);
  const validate = useValidateCandidate(kind);
  const discard = useDiscardCandidate(kind);
  const discovery = useDiscoverModels(kind);
  const busy = save.isPending || validate.isPending || discard.isPending;
  const canSubmit = changed || candidate !== null || active === null;

  useEffect(() => {
    onUnsavedChange(kind, changed);
    return () => onUnsavedChange(kind, false);
  }, [kind, changed, onUnsavedChange]);

  const baseId = useId();
  const ids = {
    provider: `${baseId}-provider`,
    providerLabel: `${baseId}-provider-label`,
    endpoint: `${baseId}-endpoint`,
    auth: `${baseId}-auth`,
    authLabel: `${baseId}-auth-label`,
    apiKey: `${baseId}-api-key`,
    model: `${baseId}-model`,
  };
  const errors = formState.errors;
  const outcome = discovery.data
    ? discoveryOutcome(discovery.data)
    : discovery.isError
      ? discoveryFailure(errorMessage(discovery.error))
      : null;

  function showStored(next: ModelKindSettings) {
    reset(savedValues(next.candidate ?? next.active, kind, providers));
  }

  function validateSaved() {
    validate.mutate(undefined, { onSuccess: (result) => showStored(result.settings) });
  }

  const submit = form.handleSubmit((submitted) => {
    if (!provider) return;
    discovery.reset();
    if (!changed && candidate !== null) {
      validateSaved();
      return;
    }
    validate.reset();
    save.mutate(candidateRequest(submitted, provider, kind), {
      onSuccess: (next) => {
        showStored(next);
        validateSaved();
      },
    });
  });

  function discardEdit() {
    save.reset();
    validate.reset();
    discovery.reset();
    if (candidate === null) {
      reset(saved);
      return;
    }
    discard.mutate(undefined, { onSuccess: showStored });
  }

  function chooseProvider(id: string) {
    const next = findProvider(id);
    if (!next) return;
    reset(valuesForProvider(next, saved));
    discovery.reset();
  }

  function applyOwnAddress() {
    setValue("endpoint", provider?.default_endpoint ?? "", { shouldValidate: formState.isSubmitted });
    discovery.reset();
  }

  function chooseAuthStrategy(next: AuthStrategy) {
    setValue("authStrategy", next);
    if (next === "environment") applyOwnAddress();
    else if (formState.isSubmitted) void trigger("endpoint");
    discovery.reset();
  }

  async function loadModels() {
    if (!(await trigger("endpoint"))) return;
    discovery.mutate(discoveryRequest(getValues()), {
      onSuccess: (result) => {
        const suggestion = result.ok ? preferredModel(kind, discoveryOutcome(result).models) : undefined;
        if (suggestion && getValues("model").trim() === "") setValue("model", suggestion);
      },
    });
  }

  function applySuggestedEndpoint(suggestion: string) {
    setValue("endpoint", suggestion, { shouldValidate: true });
    discovery.reset();
  }

  const endpointCopy = provider ? endpointField(provider, authStrategy) : null;
  const modelCopy = provider ? modelField(provider) : null;
  const authHint = AUTH_HINTS[authStrategy];
  const savedKey = savedKeyOf(values, provider);
  const endpointConflict = provider ? environmentEndpointConflict(values, provider) : null;
  const endpointHint = endpointConflict === null ? endpointCopy?.hint : null;

  return (
    <FormProvider {...form}>
      <form onSubmit={submit} noValidate aria-label={`${title} settings`} className="space-y-4">
        <fieldset disabled={busy} className="min-w-0 space-y-4">
          <FormField label="Provider" htmlFor={ids.provider} labelId={ids.providerLabel}>
            <Select disabled={busy} value={providerId} onValueChange={(next) => (typeof next === "string" ? chooseProvider(next) : undefined)}>
              <SelectTrigger id={ids.provider} aria-labelledby={ids.providerLabel} className="w-full">
                <SelectValue>
                  {(selected: string) => findProvider(selected)?.label ?? (selected || "Choose a provider")}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {choices.map((choice) => (
                  <SelectItem key={choice.id} value={choice.id}>
                    {choice.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FormField>

          {provider && endpointCopy && modelCopy ? (
            <>
              <FormField
                label={endpointCopy.label}
                htmlFor={ids.endpoint}
                hint={endpointHint}
                hintId={`${ids.endpoint}-hint`}
                error={errors.endpoint?.message}
                errorId={`${ids.endpoint}-error`}
              >
                <Input
                  id={ids.endpoint}
                  placeholder={endpointCopy.placeholder}
                  spellCheck={false}
                  readOnly={endpointCopy.fixed}
                  aria-invalid={errors.endpoint ? true : undefined}
                  aria-describedby={describedBy(
                    errors.endpoint && `${ids.endpoint}-error`,
                    endpointHint && `${ids.endpoint}-hint`,
                  )}
                  className={INPUT_CLASS}
                  {...register("endpoint", { onChange: () => discovery.reset() })}
                />
              </FormField>

              {endpointConflict !== null ? (
                <EndpointConflictNotice
                  endpoint={endpointConflict}
                  provider={provider}
                  keylessProviders={keylessProviders(kind, providers)}
                  onUseOwnAddress={applyOwnAddress}
                />
              ) : null}

              {provider.auth_strategies.length > 1 ? (
                <FormField
                  label="Authentication"
                  htmlFor={ids.auth}
                  labelId={ids.authLabel}
                  hint={authHint}
                  hintId={`${ids.auth}-hint`}
                >
                  <Select
                    disabled={busy}
                    value={authStrategy}
                    onValueChange={(next) => {
                      const chosen = provider.auth_strategies.find((strategy) => strategy === next);
                      if (chosen) chooseAuthStrategy(chosen);
                    }}
                  >
                    <SelectTrigger
                      id={ids.auth}
                      aria-labelledby={ids.authLabel}
                      aria-describedby={describedBy(authHint && `${ids.auth}-hint`)}
                      className="w-full"
                    >
                      <SelectValue>{(selected: AuthStrategy) => AUTH_LABELS[selected] ?? selected}</SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {provider.auth_strategies.map((strategy) => (
                        <SelectItem key={strategy} value={strategy}>
                          {AUTH_LABELS[strategy]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </FormField>
              ) : null}

              {authStrategy === "api_key" ? (
                <FormField label="API key" htmlFor={ids.apiKey} error={errors.apiKey?.message} errorId={`${ids.apiKey}-error`}>
                  <ApiKeyField
                    id={ids.apiKey}
                    savedKey={savedKey}
                    hasTypedKey={apiKey.trim() !== ""}
                    errorId={errors.apiKey ? `${ids.apiKey}-error` : undefined}
                    onKeyChange={() => discovery.reset()}
                  />
                </FormField>
              ) : null}

              {provider.options.map((name) => {
                const copy = optionField(name);
                const id = `${baseId}-option-${name}`;
                return (
                  <FormField key={name} label={copy.label} htmlFor={id} hint={copy.hint} hintId={`${id}-hint`}>
                    <Input id={id} spellCheck={false} aria-describedby={`${id}-hint`} className={INPUT_CLASS} {...register(`options.${name}`)} />
                  </FormField>
                );
              })}

              <FormField
                label={modelCopy.label}
                htmlFor={ids.model}
                hint={modelCopy.hint}
                hintId={`${ids.model}-hint`}
                error={errors.model?.message}
                errorId={`${ids.model}-error`}
              >
                <div className="flex items-center gap-2">
                  <Input
                    id={ids.model}
                    placeholder={modelCopy.placeholder}
                    spellCheck={false}
                    aria-invalid={errors.model ? true : undefined}
                    aria-describedby={describedBy(errors.model && `${ids.model}-error`, modelCopy.hint && `${ids.model}-hint`)}
                    className={INPUT_CLASS}
                    {...register("model")}
                  />
                  {provider.lists_models ? (
                    <Button type="button" variant="outline" onClick={loadModels} disabled={discovery.isPending}>
                      {discovery.isPending ? <Loader2 className="animate-spin" /> : <ListRestart />}
                      Load models
                    </Button>
                  ) : null}
                </div>
              </FormField>

              {outcome ? <DiscoveryNotice outcome={outcome} onUseEndpoint={applySuggestedEndpoint} /> : null}
              {outcome ? (
                <ModelSuggestions kind={kind} models={outcome.models} typed={model} onSelect={(choice) => setValue("model", choice)} />
              ) : null}

              {parameterNames(provider, kind).map((name) => {
                const copy = parameterField(name);
                const id = `${baseId}-parameter-${name}`;
                const error = errors.parameters?.[name]?.message;
                return (
                  <FormField
                    key={name}
                    label={copy.label}
                    htmlFor={id}
                    hint={copy.hint}
                    hintId={`${id}-hint`}
                    error={error}
                    errorId={`${id}-error`}
                  >
                    <Input
                      id={id}
                      inputMode="numeric"
                      spellCheck={false}
                      aria-invalid={error ? true : undefined}
                      aria-describedby={describedBy(error && `${id}-error`, `${id}-hint`)}
                      className={`${INPUT_CLASS} max-w-40`}
                      {...register(`parameters.${name}`)}
                    />
                  </FormField>
                );
              })}
            </>
          ) : (
            <p className="text-sm text-tone-danger-foreground">
              Choose one of the listed providers. The saved one is not available on this server.
            </p>
          )}
        </fieldset>

        {candidate !== null && !busy ? (
          <PendingEditNotice kind={kind} edit={candidate} inUse={active} providers={providers} />
        ) : null}
        {validate.data?.promoted && active?.state === "ready" && !changed && !busy ? (
          <Notice tone="ok" icon={CircleCheck} title="Validation passed">
            MemForge now uses this model.
          </Notice>
        ) : null}
        {save.isError ? <ErrorNotice title="Couldn't save" message={errorMessage(save.error)} /> : null}
        {validate.isError ? <ErrorNotice title="Couldn't validate" message={errorMessage(validate.error)} /> : null}
        {discard.isError ? <ErrorNotice title="Couldn't discard the changes" message={errorMessage(discard.error)} /> : null}

        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" size="lg" disabled={!canSubmit || busy}>
            {save.isPending || validate.isPending ? <Loader2 className="animate-spin" /> : <Check />}
            {save.isPending ? "Saving…" : validate.isPending ? "Validating…" : "Save and validate"}
          </Button>
          {changed || candidate !== null ? (
            <Button type="button" variant="outline" size="lg" onClick={discardEdit} disabled={busy}>
              <Undo2 />
              Discard changes
            </Button>
          ) : null}
          <p role="status" className={validate.isPending ? "text-sm text-muted-foreground" : "text-sm text-tone-warn-foreground"}>
            {validate.isPending
              ? "Sending one real request to the model. This can take up to a minute."
              : changed
                ? "Unsaved changes"
                : ""}
          </p>
        </div>
      </form>
    </FormProvider>
  );
}
