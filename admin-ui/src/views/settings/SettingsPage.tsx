import { useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Eye, EyeOff, Loader2, RotateCw, SlidersHorizontal, X } from "lucide-react";
import client from "@/api/client";
import type { LlmConfig, LlmConfigProbeResponse } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

type LlmKind = "enrichment" | "embedding";
type KeyAction = "keep" | "replace" | "clear";

type LlmForm = {
  enrichment_model: string;
  enrichment_base_url: string;
  enrichment_api_key: string;
  embedding_model: string;
  embedding_base_url: string;
  embedding_api_key: string;
};

type ProbeState = {
  status: "idle" | "loading" | "success" | "info" | "error";
  message: string;
  latencyMs: number | null;
  suggestedBaseUrl: string | null;
  models: string[];
};

const emptyForm: LlmForm = {
  enrichment_model: "",
  enrichment_base_url: "",
  enrichment_api_key: "",
  embedding_model: "",
  embedding_base_url: "",
  embedding_api_key: "",
};

const idleProbe: ProbeState = {
  status: "idle",
  message: "",
  latencyMs: null,
  suggestedBaseUrl: null,
  models: [],
};

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<LlmForm>(emptyForm);
  const [keyActions, setKeyActions] = useState<Record<LlmKind, KeyAction>>({
    enrichment: "keep",
    embedding: "keep",
  });
  const [showKeys, setShowKeys] = useState<Record<LlmKind, boolean>>({
    enrichment: false,
    embedding: false,
  });
  const [probes, setProbes] = useState<Record<LlmKind, ProbeState>>({
    enrichment: idleProbe,
    embedding: idleProbe,
  });

  const configQuery = useQuery<LlmConfig>({
    queryKey: ["llm-config"],
    queryFn: () => client.get("/api/llm-config").then((response) => response.data),
  });

  useEffect(() => {
    const config = configQuery.data;
    if (!config) return;
    setForm({
      enrichment_model: config.enrichment_model ?? "",
      enrichment_base_url: config.enrichment_base_url ?? "",
      enrichment_api_key: "",
      embedding_model: config.embedding_model ?? "",
      embedding_base_url: config.embedding_base_url ?? "",
      embedding_api_key: "",
    });
    setKeyActions({ enrichment: "keep", embedding: "keep" });
    setProbes({ enrichment: idleProbe, embedding: idleProbe });
  }, [configQuery.data]);

  const updateConfig = useMutation({
    mutationFn: (payload: Partial<LlmConfig>) => client.put("/api/llm-config", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
    },
  });

  const updateField = (field: keyof LlmForm, value: string) => {
    setForm((current) => ({ ...current, [field]: value }));
    if (field === "enrichment_api_key") {
      setKeyActions((current) => ({ ...current, enrichment: value ? "replace" : "keep" }));
      setProbes((current) => ({ ...current, enrichment: idleProbe }));
    }
    if (field === "embedding_api_key") {
      setKeyActions((current) => ({ ...current, embedding: value ? "replace" : "keep" }));
      setProbes((current) => ({ ...current, embedding: idleProbe }));
    }
  };

  const setBaseUrl = (kind: LlmKind, value: string) => {
    updateField(`${kind}_base_url`, value);
    setProbes((current) => ({ ...current, [kind]: idleProbe }));
  };

  const clearKey = (kind: LlmKind) => {
    updateField(`${kind}_api_key`, "");
    setKeyActions((current) => ({ ...current, [kind]: "clear" }));
  };

  const keepKey = (kind: LlmKind) => {
    updateField(`${kind}_api_key`, "");
    setKeyActions((current) => ({ ...current, [kind]: "keep" }));
  };

  const probeEndpoint = async (kind: LlmKind) => {
    const baseUrl = form[`${kind}_base_url`].trim();
    if (!isHttpUrl(baseUrl)) {
      setProbes((current) => ({
        ...current,
        [kind]: {
          ...idleProbe,
          status: "error",
          message: "Enter a URL starting with http:// or https://.",
        },
      }));
      return;
    }

    const keyAction = keyActions[kind];
    const typedKey = form[`${kind}_api_key`];
    const apiKey = keyAction === "replace" ? typedKey : keyAction === "clear" ? "" : null;

    setProbes((current) => ({
      ...current,
      [kind]: { ...idleProbe, status: "loading", message: "Testing..." },
    }));

    try {
      const response = await client.post<LlmConfigProbeResponse>("/api/llm-config/probe", {
        kind,
        base_url: baseUrl,
        api_key: apiKey,
      });
      const data = response.data;
      const models = data.models.map((model) => model.id);
      const modelField = `${kind}_model` as keyof LlmForm;
      if (data.ok && models.length && !form[modelField].trim()) {
        const model = preferredModel(kind, models);
        setForm((current) => (current[modelField].trim() ? current : { ...current, [modelField]: model }));
      }
      setProbes((current) => ({
        ...current,
        [kind]: {
          status: data.ok ? (data.models_supported && models.length ? "success" : "info") : "error",
          message: data.message,
          latencyMs: data.latency_ms,
          suggestedBaseUrl: data.suggested_base_url,
          models,
        },
      }));
    } catch (error) {
      setProbes((current) => ({
        ...current,
        [kind]: {
          ...idleProbe,
          status: "error",
          message: apiErrorMessage(error),
        },
      }));
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    updateConfig.mutate(buildPayload(form, keyActions));
  };

  return (
    <div className="max-w-3xl space-y-4">
      <PageHeader title="Settings" description="Configure the LLM endpoints MemForge uses for enrichment and embeddings." />

      <AsyncBoundary
        isLoading={configQuery.isLoading}
        isError={configQuery.isError}
        error={configQuery.error}
        onRetry={() => configQuery.refetch()}
        isEmpty={!configQuery.data}
        empty={
          <EmptyState
            icon={SlidersHorizontal}
            title="No configuration found"
            description="The admin API did not return model configuration."
          />
        }
      >
        <form onSubmit={handleSubmit} className="space-y-4">
          <LlmSection
            kind="enrichment"
            title="Enrichment"
            description="Used for content extraction and agent-session output."
            config={configQuery.data}
            form={form}
            keyAction={keyActions.enrichment}
            showKey={showKeys.enrichment}
            probe={probes.enrichment}
            onChange={updateField}
            onBaseUrlChange={setBaseUrl}
            onProbe={probeEndpoint}
            onClearKey={clearKey}
            onKeepKey={keepKey}
            onToggleKey={() => setShowKeys((current) => ({ ...current, enrichment: !current.enrichment }))}
          />

          <LlmSection
            kind="embedding"
            title="Embedding"
            description="Used for vector search over documents and memories."
            config={configQuery.data}
            form={form}
            keyAction={keyActions.embedding}
            showKey={showKeys.embedding}
            probe={probes.embedding}
            onChange={updateField}
            onBaseUrlChange={setBaseUrl}
            onProbe={probeEndpoint}
            onClearKey={clearKey}
            onKeepKey={keepKey}
            onToggleKey={() => setShowKeys((current) => ({ ...current, embedding: !current.embedding }))}
          />

          {updateConfig.isError && (
            <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
              Couldn't save. Check the values and try again.
            </div>
          )}

          <div className="flex items-center gap-2 pt-1">
            <Button type="submit" disabled={updateConfig.isPending}>
              {updateConfig.isPending && <Loader2 className="size-4 animate-spin" />}
              Save changes
            </Button>
            {updateConfig.isSuccess && (
              <span className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                <Check className="size-4" />
                Saved
              </span>
            )}
          </div>
        </form>
      </AsyncBoundary>
    </div>
  );
}

function LlmSection({
  kind,
  title,
  description,
  config,
  form,
  keyAction,
  showKey,
  probe,
  onChange,
  onBaseUrlChange,
  onProbe,
  onClearKey,
  onKeepKey,
  onToggleKey,
}: {
  kind: LlmKind;
  title: string;
  description: string;
  config?: LlmConfig;
  form: LlmForm;
  keyAction: KeyAction;
  showKey: boolean;
  probe: ProbeState;
  onChange: (field: keyof LlmForm, value: string) => void;
  onBaseUrlChange: (kind: LlmKind, value: string) => void;
  onProbe: (kind: LlmKind) => void;
  onClearKey: (kind: LlmKind) => void;
  onKeepKey: (kind: LlmKind) => void;
  onToggleKey: () => void;
}) {
  const baseUrlField = `${kind}_base_url` as keyof LlmForm;
  const apiKeyField = `${kind}_api_key` as keyof LlmForm;
  const modelField = `${kind}_model` as keyof LlmForm;
  const keySet = Boolean(config?.[`${kind}_api_key_set`]);
  const keyLast4 = config?.[`${kind}_api_key_last4`] ?? null;
  const baseUrlId = `${kind}-base-url`;
  const apiKeyId = `${kind}-api-key`;
  const modelId = `${kind}-model`;
  const modelOptions = suggestedModels(kind, probe.models);

  return (
    <DataSurface>
      <div className="space-y-4 p-4">
        <div>
          <h2 className="text-base font-semibold">{title}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        </div>
        <Separator />

        <Field label="Base URL" htmlFor={baseUrlId}>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              id={baseUrlId}
              value={form[baseUrlField]}
              onChange={(event) => onBaseUrlChange(kind, event.target.value)}
              placeholder="https://api.example.com/v1"
              className="font-mono"
            />
            <Button
              type="button"
              variant="outline"
              onClick={() => onProbe(kind)}
              disabled={!form[baseUrlField].trim() || probe.status === "loading"}
              className="sm:w-40"
            >
              {probe.status === "loading" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <RotateCw className="size-4" />
              )}
              Test connection
            </Button>
          </div>
        </Field>

        <Field label="API key" htmlFor={apiKeyId}>
          <div className="relative">
            <Input
              id={apiKeyId}
              type={showKey ? "text" : "password"}
              value={form[apiKeyField]}
              onChange={(event) => onChange(apiKeyField, event.target.value)}
              placeholder={keySet ? "Leave blank to keep saved key" : "Required for hosted providers"}
              className="pr-10 font-mono"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={showKey ? "Hide API key" : "Show API key"}
              onClick={onToggleKey}
              className="absolute right-1 top-1/2 -translate-y-1/2"
            >
              {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </Button>
          </div>
          <KeyStatus
            keySet={keySet}
            keyLast4={keyLast4}
            keyAction={keyAction}
            hasTypedKey={Boolean(form[apiKeyField])}
            onClear={() => onClearKey(kind)}
            onKeep={() => onKeepKey(kind)}
          />
        </Field>

        <Field label="Model" htmlFor={modelId}>
          <Input
            id={modelId}
            value={form[modelField]}
            onChange={(event) => onChange(modelField, event.target.value)}
            placeholder="Model id"
            className="font-mono"
          />
          {modelOptions.length > 0 && (
            <ModelSuggestions
              kind={kind}
              models={modelOptions}
              totalModels={probe.models.length}
              selectedModel={form[modelField]}
              onSelect={(model) => onChange(modelField, model)}
            />
          )}
        </Field>

        <ProbeBanner probe={probe} onUseSuggestion={(url) => onBaseUrlChange(kind, url)} />
      </div>
    </DataSurface>
  );
}

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-xs font-medium text-muted-foreground">
        {label}
      </label>
      {children}
    </div>
  );
}

function KeyStatus({
  keySet,
  keyLast4,
  keyAction,
  hasTypedKey,
  onClear,
  onKeep,
}: {
  keySet: boolean;
  keyLast4: string | null;
  keyAction: KeyAction;
  hasTypedKey: boolean;
  onClear: () => void;
  onKeep: () => void;
}) {
  if (keyAction === "clear") {
    return (
      <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-amber-700">
        <span>Saved key will be removed on save.</span>
        <Button type="button" variant="ghost" size="sm" onClick={onKeep}>
          Undo
        </Button>
      </div>
    );
  }
  if (keySet && hasTypedKey) {
    return (
      <p className="mt-1 text-sm text-muted-foreground">
        Will replace saved key{keyLast4 ? ` (${maskLast4(keyLast4)})` : ""} on save.
      </p>
    );
  }
  if (keySet) {
    return (
      <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <span>Saved key{keyLast4 ? ` (${maskLast4(keyLast4)})` : ""} in use. Leave blank to keep it.</span>
        <Button type="button" variant="ghost" size="sm" onClick={onClear}>
          <X className="size-3.5" />
          Remove
        </Button>
      </div>
    );
  }
  if (hasTypedKey) {
    return <p className="mt-1 text-sm text-muted-foreground">New key will be saved.</p>;
  }
  return <p className="mt-1 text-sm text-muted-foreground">Leave blank only for local endpoints that do not require auth.</p>;
}

function ProbeBanner({
  probe,
  onUseSuggestion,
}: {
  probe: ProbeState;
  onUseSuggestion: (url: string) => void;
}) {
  if (probe.status === "idle" || probe.status === "loading") return null;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-2 rounded-lg border p-3 text-sm",
        probe.status === "success" && "border-emerald-200 bg-emerald-50 text-emerald-800",
        probe.status === "info" && "border-amber-200 bg-amber-50 text-amber-800",
        probe.status === "error" && "border-destructive/20 bg-destructive/10 text-destructive",
      )}
    >
      {probe.status === "success" && <Check className="size-4" />}
      <span>
        {probe.message}
        {probe.latencyMs !== null ? ` (${probe.latencyMs} ms)` : ""}
      </span>
      {probe.suggestedBaseUrl && (
        <Button type="button" variant="outline" size="sm" onClick={() => onUseSuggestion(probe.suggestedBaseUrl!)}>
          Use host.docker.internal
        </Button>
      )}
    </div>
  );
}

function ModelSuggestions({
  kind,
  models,
  totalModels,
  selectedModel,
  onSelect,
}: {
  kind: LlmKind;
  models: string[];
  totalModels: number;
  selectedModel: string;
  onSelect: (model: string) => void;
}) {
  const filtered = models.length !== totalModels;
  return (
    <div className="mt-2 space-y-2">
      <p className="text-sm text-muted-foreground">
        {filtered
          ? `${models.length} ${kind} suggestion${models.length === 1 ? "" : "s"} from ${totalModels} returned models.`
          : `${totalModels} model${totalModels === 1 ? "" : "s"} available from this endpoint.`}{" "}
        Select one below or type any model id.
      </p>
      <div className="flex flex-wrap gap-2">
        {models.map((model) => {
          const selected = model === selectedModel;
          return (
            <Button
              key={model}
              type="button"
              variant={selected ? "secondary" : "outline"}
              size="sm"
              aria-pressed={selected}
              onClick={() => onSelect(model)}
              className="h-auto max-w-full justify-start whitespace-normal break-all px-2 py-1 font-mono text-xs"
            >
              {model}
            </Button>
          );
        })}
      </div>
    </div>
  );
}

function buildPayload(form: LlmForm, keyActions: Record<LlmKind, KeyAction>): Partial<LlmConfig> {
  const payload: Partial<LlmConfig> = {
    enrichment_model: form.enrichment_model.trim(),
    enrichment_base_url: form.enrichment_base_url.trim(),
    embedding_model: form.embedding_model.trim(),
    embedding_base_url: form.embedding_base_url.trim(),
  };

  const enrichmentKey = keyPayload("enrichment", form, keyActions);
  const embeddingKey = keyPayload("embedding", form, keyActions);
  if (enrichmentKey !== undefined) payload.enrichment_api_key = enrichmentKey;
  if (embeddingKey !== undefined) payload.embedding_api_key = embeddingKey;
  return payload;
}

function keyPayload(kind: LlmKind, form: LlmForm, keyActions: Record<LlmKind, KeyAction>): string | undefined {
  if (keyActions[kind] === "clear") return "";
  if (keyActions[kind] === "replace") return form[`${kind}_api_key`];
  return undefined;
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function preferredModel(kind: LlmKind, models: string[]): string {
  const suggestions = suggestedModels(kind, models);
  return suggestions[0] ?? models[0];
}

function suggestedModels(kind: LlmKind, models: string[]): string[] {
  if (kind === "embedding") {
    const embeddingModels = models.filter(isEmbeddingModel);
    return embeddingModels.length ? embeddingModels : models;
  }
  const generationModels = models.filter((model) => !isEmbeddingModel(model));
  return generationModels.length ? generationModels : models;
}

function isEmbeddingModel(model: string): boolean {
  return /embed/i.test(model);
}

function apiErrorMessage(error: unknown): string {
  if (typeof error === "object" && error && "response" in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof response?.data?.detail === "string") return response.data.detail;
  }
  return "Could not reach the admin API.";
}

function maskLast4(last4: string): string {
  return `****${last4}`;
}
