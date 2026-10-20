import { zodResolver } from "@hookform/resolvers/zod";
import { Check, CircleCheck, ListRestart, Loader2, RefreshCw, Undo2 } from "lucide-react";
import { useEffect, useId, useMemo, useState, type ChangeEvent } from "react";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import { isReindexInProgress } from "@/api";
import { useFetchImageDocuments } from "@/features/sources";
import { errorMessage } from "@/lib/errors";
import { ErrorNotice, Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { useDiscardCandidate, useDiscoverModels, useSaveCandidate, useStartReindex, useValidateCandidate } from "./api";
import { ApiKeyField } from "./ApiKeyField";
import { describedBy } from "./describedBy";
import { DiscoveryNotice } from "./DiscoveryNotice";
import { EndpointConflictNotice } from "./EndpointConflictNotice";
import { FormField } from "./FormField";
import { discoveryFailure, discoveryOutcome, namedListedModel, onlyListedModel } from "./model/discoveryOutcome";
import {
  candidateRequest,
  discoveryRequest,
  environmentEndpointConflict,
  fieldProblems,
  formOrigin,
  hasTokenLimitValues,
  isChanged,
  listedLimitsGap,
  listedTokenLimits,
  modelFormSchema,
  nativeSchemaForTypedModel,
  offersTokenLimits,
  savedValues,
  tokenLimitsForChoice,
  tokenLimitsForTypedModel,
  valuesForProvider,
  type ListedLimits,
  type ModelFormValues,
} from "./model/modelForm";
import { awaitsTokenLimits, needsTokenLimits, validationFacts } from "./model/modelStatus";
import {
  AUTH_HINTS,
  AUTH_LABELS,
  endpointField,
  keylessProviders,
  modelField,
  optionField,
  parameterField,
  parameterNames,
  providerHint,
  providersFor,
  tokenLimitNames,
} from "./model/providerFields";
import { savedKeyFor } from "./model/savedKey";
import { rebuildLabel, startRefusalReason } from "./model/searchIndex";
import type { AuthStrategy, ModelKind, ModelKindSettings, ModelOption, ModelProvider, VectorIndex } from "./model/types";
import { ModelSuggestions } from "./ModelSuggestions";
import { PendingEditNotice } from "./PendingEditNotice";
import { ReReadImagesDialog } from "./ReReadImagesDialog";
import { TokenLimitFields } from "./TokenLimitFields";

const INPUT_CLASS = "font-mono read-only:bg-surface-subtle";

interface ModelFormProps {
  kind: ModelKind;
  title: string;
  settings: ModelKindSettings;
  providers: readonly ModelProvider[];
  /** The search index, for the embedding model, whose edits can wait for a rebuild of it. */
  vectorIndex?: VectorIndex;
  /** Tells the page whether the form holds changes that are not saved. */
  onUnsavedChange: (kind: ModelKind, unsaved: boolean) => void;
}

/**
 * The form that changes one model. It edits the saved edit when there is one
 * and the model in use otherwise. Its one action saves the form and validates
 * what was saved with a real request; only an edit that passes is put to use.
 *
 * Load models offers the provider's models of the kind. A model field that
 * names one of them, picked from the list or typed, fills the empty token
 * limit fields with the limits the listing gives for it, and saves what the
 * listing declares about the model answering in a JSON schema. When validation finds
 * that only the token limits are missing and the provider's listing can give
 * them, the step offers to fill them in from the listing.
 *
 * An embedding model whose vectors cannot be compared with the stored ones
 * passes validation and then waits: the action becomes the rebuild of the
 * search index, which puts the model to use when it ends. While a rebuild
 * runs the embedding model cannot be changed.
 */
export function ModelForm({ kind, title, settings, providers, vectorIndex, onUnsavedChange }: ModelFormProps) {
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
  const [providerId, endpoint, authStrategy, apiKey, model, options, parameters, tokenLimits, nativeSchema] = useWatch({
    control,
    name: ["provider", "endpoint", "authStrategy", "apiKey", "model", "options", "parameters", "tokenLimits", "nativeSchema"],
  });
  const values: ModelFormValues = {
    provider: providerId,
    endpoint,
    authStrategy,
    apiKey,
    model,
    options,
    parameters,
    tokenLimits,
    nativeSchema,
  };
  /** The token limits the form filled in from the model list, while it is the source of what the fields hold. */
  const [listedLimits, setListedLimits] = useState<ListedLimits | null>(null);
  /** The model whose token limits were last looked up in the model list from the token limits step. */
  const [limitsLookup, setLimitsLookup] = useState<string | null>(null);
  const provider = findProvider(providerId);
  const changed = isChanged(saved, values, provider, kind);

  const save = useSaveCandidate(kind);
  const validate = useValidateCandidate(kind);
  const discard = useDiscardCandidate(kind);
  const discovery = useDiscoverModels(kind);
  const rebuild = useStartReindex();
  const fetchImageDocuments = useFetchImageDocuments();
  /** The documents with an image, while the offer to re-read them is open. */
  const [reReadCount, setReReadCount] = useState<number | null>(null);
  const rebuilding = vectorIndex?.status === "reindexing";
  const busy = save.isPending || validate.isPending || discard.isPending || rebuild.isPending;
  const locked = busy || rebuilding;
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
    setListedLimits(null);
    setLimitsLookup(null);
  }

  function validateSaved() {
    const readTextOnly =
      kind === "generation" && active?.validation?.passed === true && validationFacts(active.validation).image_input === false;
    validate.mutate(undefined, {
      onSuccess: (result) => {
        showStored(result.settings);
        const promoted = result.promoted ? result.settings.active : null;
        if (readTextOnly && promoted?.validation && validationFacts(promoted.validation).image_input === true) {
          offerReRead();
        }
      },
    });
  }

  /**
   * Offers to re-read the documents with an image, which a model that read
   * text only read without their images. When they cannot be counted the
   * offer is left out; each source's menu still re-reads its documents.
   */
  function offerReRead() {
    fetchImageDocuments().then(
      ({ document_count: count }) => {
        if (count > 0) setReReadCount(count);
      },
      () => undefined,
    );
  }

  const submit = form.handleSubmit((submitted) => {
    if (!provider) return;
    discovery.reset();
    if (!changed && candidate !== null) {
      if (candidate.waits_for_reindex) rebuild.mutate();
      else validateSaved();
      return;
    }
    validate.reset();
    rebuild.reset();
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
    rebuild.reset();
    discovery.reset();
    setListedLimits(null);
    setLimitsLookup(null);
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
    setListedLimits(null);
    setLimitsLookup(null);
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

  function chooseModel(choice: ModelOption) {
    setValue("model", choice.id, { shouldValidate: formState.isSubmitted });
    applyListedModel(choice);
  }

  /**
   * Takes what the listing declares about `choice` answering in a JSON schema, and fills the
   * token limit fields from what it gives for `choice`, as far as they are empty or listed.
   */
  function applyListedModel(choice: ModelOption) {
    if (!provider) return;
    setValue("nativeSchema", choice.native_schema ?? null);
    const names = tokenLimitNames(provider, kind);
    const listed = listedTokenLimits(choice, names);
    const next = tokenLimitsForChoice(getValues("tokenLimits"), names, listedLimits?.limits ?? null, listed);
    if (next === null) return;
    setValue("tokenLimits", next, { shouldValidate: formState.isSubmitted });
    setListedLimits(Object.keys(listed).length > 0 ? { model: choice.id, limits: next } : null);
  }

  /** Clears the token limits the listing gave for another model than the one `typed` names. */
  function followTypedModel(typed: string) {
    if (listedLimits === null || typed.trim() === listedLimits.model) return;
    const next = provider ? tokenLimitsForTypedModel(getValues("tokenLimits"), tokenLimitNames(provider, kind), listedLimits, typed) : null;
    if (next !== null) setValue("tokenLimits", next, { shouldValidate: formState.isSubmitted });
    setListedLimits(null);
  }

  /**
   * Follows a new value of the model field. The field reports every value it
   * takes as an input event, typed, pasted or filled in by the browser's
   * autofill alike, and naming a listed model counts as choosing it.
   */
  function followModelField(typed: string) {
    const named = namedListedModel(outcome, typed);
    if (named) {
      applyListedModel(named);
      return;
    }
    setValue("nativeSchema", nativeSchemaForTypedModel(saved, providerId, typed));
    followTypedModel(typed);
  }

  async function loadModels() {
    if (!(await trigger("endpoint"))) return;
    discovery.mutate(discoveryRequest(getValues()), {
      onSuccess: (result) => {
        if (!result.ok) return;
        const loaded = discoveryOutcome(result);
        const typed = getValues("model");
        const named = namedListedModel(loaded, typed);
        const only = onlyListedModel(loaded);
        if (named) applyListedModel(named);
        else if (only && typed.trim() === "") chooseModel(only);
      },
    });
  }

  function fillLimitsFromList() {
    setLimitsLookup(getValues("model").trim());
    void loadModels();
  }

  function applySuggestedEndpoint(suggestion: string) {
    setValue("endpoint", suggestion, { shouldValidate: true });
    discovery.reset();
  }

  const endpointCopy = provider ? endpointField(provider, authStrategy) : null;
  const providerCopy = providerHint(provider);
  const modelCopy = provider ? modelField(provider) : null;
  const authHint = AUTH_HINTS[authStrategy];
  const savedKey = savedKeyOf(values, provider);
  const endpointConflict = provider ? environmentEndpointConflict(values, provider) : null;
  const endpointHint = endpointConflict === null ? endpointCopy?.hint : null;
  const limitNames =
    provider && (offersTokenLimits(stored) || hasTokenLimitValues(tokenLimits)) ? tokenLimitNames(provider, kind) : [];
  const limitsFromList =
    provider &&
    listedLimits !== null &&
    model.trim() === listedLimits.model &&
    limitNames.every((name) => (tokenLimits[name] ?? "").trim() === (listedLimits.limits[name] ?? ""))
      ? provider.label
      : null;
  const offersListedLimits =
    provider !== undefined &&
    provider.lists_token_limits &&
    limitNames.length > 0 &&
    candidate !== null &&
    candidate.provider === provider.id &&
    awaitsTokenLimits(candidate);
  const lookedUp = offersListedLimits && limitsLookup === model.trim() && outcome !== null && !discovery.isPending;
  const limitsGap = lookedUp ? listedLimitsGap(outcome, model, limitNames, provider.label) : null;
  const listFill = offersListedLimits
    ? {
        onFill: limitsFromList !== null || (lookedUp && outcome.tone !== "danger") ? null : fillLimitsFromList,
        loading: limitsLookup !== null && discovery.isPending,
        message: limitsGap,
      }
    : undefined;
  const rebuilt = rebuild.isSuccess && vectorIndex?.reindex?.state === "succeeded" && candidate === null;
  const rebuildAction = vectorIndex && candidate?.waits_for_reindex && !changed ? rebuildLabel(vectorIndex) : null;
  const rebuildingForEdit = rebuilding && candidate?.waits_for_reindex === true;
  const working = save.isPending || validate.isPending || rebuild.isPending || rebuildingForEdit;
  const actionLabel = save.isPending
    ? "Saving…"
    : validate.isPending
      ? "Validating…"
      : rebuild.isPending
        ? "Starting the rebuild…"
        : rebuildingForEdit
          ? "Rebuilding…"
          : (rebuildAction ?? "Save and validate");

  const listedImageInput =
    kind === "generation" ? (outcome?.models.find((option) => option.id === model.trim())?.image_input ?? null) : null;

  return (
    <FormProvider {...form}>
      <form onSubmit={submit} noValidate aria-label={`${title} settings`} className="space-y-4">
        <fieldset disabled={locked} className="min-w-0 space-y-4">
          <FormField
            label="Provider"
            htmlFor={ids.provider}
            labelId={ids.providerLabel}
            hint={providerCopy}
            hintId={`${ids.provider}-hint`}
          >
            <Select disabled={locked} value={providerId} onValueChange={(next) => (typeof next === "string" ? chooseProvider(next) : undefined)}>
              <SelectTrigger
                id={ids.provider}
                aria-labelledby={ids.providerLabel}
                aria-describedby={describedBy(providerCopy && `${ids.provider}-hint`)}
                className="w-full"
              >
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
                    disabled={locked}
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
                    <Input
                      id={id}
                      spellCheck={false}
                      aria-describedby={describedBy(copy.hint && `${id}-hint`)}
                      className={INPUT_CLASS}
                      {...register(`options.${name}`)}
                    />
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
                    {...register("model", { onChange: (event: ChangeEvent<HTMLInputElement>) => followModelField(event.target.value) })}
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
                <ModelSuggestions
                  kind={kind}
                  models={outcome.models}
                  kindFiltered={outcome.kindFiltered}
                  typed={model}
                  onSelect={chooseModel}
                />
              ) : null}
              {listedImageInput !== null && provider ? (
                <p className="text-xs text-muted-foreground">
                  Image input: {listedImageInput ? "supported" : "not supported"}, according to {provider.label}'s model
                  list.
                </p>
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
                      aria-describedby={describedBy(error && `${id}-error`, copy.hint && `${id}-hint`)}
                      className={`${INPUT_CLASS} max-w-40`}
                      {...register(`parameters.${name}`)}
                    />
                  </FormField>
                );
              })}

              {limitNames.length > 0 ? (
                <TokenLimitFields
                  id={`${baseId}-token-limits`}
                  names={limitNames}
                  missing={stored !== null && needsTokenLimits(stored)}
                  listedBy={limitsFromList}
                />
              ) : null}
            </>
          ) : (
            <p className="text-sm text-tone-danger-foreground">
              Choose one of the listed providers. The saved one is not available on this server.
            </p>
          )}
        </fieldset>

        {candidate !== null && !locked ? (
          <PendingEditNotice
            edit={candidate}
            inUse={active}
            providers={providers}
            vectorSearchOn={vectorIndex?.serves_search}
            listFill={listFill}
          />
        ) : null}
        {rebuilt && !changed && !locked ? (
          <Notice tone="ok" icon={CircleCheck} title="Search index rebuilt">
            MemForge now uses this model.
          </Notice>
        ) : null}
        {save.isError ? <ErrorNotice title="Couldn't save" message={errorMessage(save.error)} /> : null}
        {validate.isError ? <ErrorNotice title="Couldn't validate" message={errorMessage(validate.error)} /> : null}
        {discard.isError ? <ErrorNotice title="Couldn't discard the changes" message={errorMessage(discard.error)} /> : null}
        {rebuild.isError && !isReindexInProgress(rebuild.error) ? (
          <ErrorNotice title="Couldn't start the rebuild" message={startRefusalReason(rebuild.error)} />
        ) : null}

        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" size="lg" disabled={!canSubmit || locked}>
            {working ? <Loader2 className="animate-spin" /> : rebuildAction !== null ? <RefreshCw /> : <Check />}
            {actionLabel}
          </Button>
          {changed || candidate !== null ? (
            <Button type="button" variant="outline" size="lg" onClick={discardEdit} disabled={locked}>
              <Undo2 />
              Discard changes
            </Button>
          ) : null}
          <p
            role="status"
            className={validate.isPending || rebuilding ? "text-sm text-muted-foreground" : "text-sm text-tone-warn-foreground"}
          >
            {validate.isPending
              ? "Sending one real request to the model. This can take up to a minute."
              : rebuilding
                ? "The embedding model can be changed once the search index is rebuilt."
                : changed
                  ? "Unsaved changes"
                  : ""}
          </p>
        </div>
      </form>
      <ReReadImagesDialog
        documentCount={reReadCount}
        onOpenChange={(open) => {
          if (!open) setReReadCount(null);
        }}
      />
    </FormProvider>
  );
}
