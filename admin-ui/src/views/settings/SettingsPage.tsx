import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Eye, EyeOff, Pencil } from "lucide-react";
import client from "@/api/client";
import type { LlmConfig } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const [form, setForm] = useState<Partial<LlmConfig>>({});

  const configQuery = useQuery<LlmConfig>({
    queryKey: ["llm-config"],
    queryFn: () => client.get("/api/llm-config").then((response) => response.data),
  });

  const updateConfig = useMutation({
    mutationFn: (payload: Partial<LlmConfig>) => client.put("/api/llm-config", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
      setEditing(false);
    },
  });

  const startEditing = () => {
    if (configQuery.data) setForm({ ...configQuery.data });
    setEditing(true);
  };

  const updateField = (field: keyof LlmConfig, value: string) => {
    setForm((current) => ({ ...current, [field]: value || null }));
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    updateConfig.mutate(form);
  };

  return (
    <div className="max-w-3xl space-y-4">
      <PageHeader
        title="Settings"
        description="LLM and embedding model configuration."
        actions={
          !editing ? (
            <Button type="button" variant="outline" onClick={startEditing} disabled={!configQuery.data}>
              <Pencil className="size-4" />
              Edit
            </Button>
          ) : (
            <Button type="button" variant="outline" onClick={() => setShowKeys((value) => !value)}>
              {showKeys ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              {showKeys ? "Hide keys" : "Show keys"}
            </Button>
          )
        }
      />

      <DataSurface>
        <div className="border-b p-4">
          <h2 className="text-base font-semibold">AI Configuration</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            These values are stored by the admin API and used by extraction and embedding clients.
          </p>
        </div>

        <AsyncBoundary
          isLoading={configQuery.isLoading}
          isError={configQuery.isError}
          error={configQuery.error}
          onRetry={() => configQuery.refetch()}
          isEmpty={!configQuery.data}
          empty={
            <EmptyState
              icon={Pencil}
              title="No configuration found"
              description="The admin API did not return LLM configuration."
            />
          }
        >
          <div className="p-4">
            {!editing ? (
              <div className="space-y-3">
                <ConfigRow label="Enrichment model" value={configQuery.data?.enrichment_model || "-"} />
                <ConfigRow label="Enrichment base URL" value={configQuery.data?.enrichment_base_url || "-"} />
                <ConfigRow label="Enrichment API key" value={maskKey(configQuery.data?.enrichment_api_key ?? null)} />
                <Separator className="my-4" />
                <ConfigRow label="Embedding model" value={configQuery.data?.embedding_model || "-"} />
                <ConfigRow label="Embedding base URL" value={configQuery.data?.embedding_base_url || "-"} />
                <ConfigRow label="Embedding API key" value={maskKey(configQuery.data?.embedding_api_key ?? null)} />
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <Field label="Enrichment model">
                  <Input value={form.enrichment_model ?? ""} onChange={(event) => updateField("enrichment_model", event.target.value)} />
                </Field>
                <Field label="Enrichment base URL">
                  <Input value={form.enrichment_base_url ?? ""} onChange={(event) => updateField("enrichment_base_url", event.target.value)} placeholder="https://..." />
                </Field>
                <Field label="Enrichment API key">
                  <Input type={showKeys ? "text" : "password"} value={form.enrichment_api_key ?? ""} onChange={(event) => updateField("enrichment_api_key", event.target.value)} placeholder="sk-..." />
                </Field>
                <Separator />
                <Field label="Embedding model">
                  <Input value={form.embedding_model ?? ""} onChange={(event) => updateField("embedding_model", event.target.value)} />
                </Field>
                <Field label="Embedding base URL">
                  <Input value={form.embedding_base_url ?? ""} onChange={(event) => updateField("embedding_base_url", event.target.value)} placeholder="https://..." />
                </Field>
                <Field label="Embedding API key">
                  <Input type={showKeys ? "text" : "password"} value={form.embedding_api_key ?? ""} onChange={(event) => updateField("embedding_api_key", event.target.value)} placeholder="sk-..." />
                </Field>

                {updateConfig.isError && (
                  <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                    Failed to save configuration. Please check the values and try again.
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button type="submit" disabled={updateConfig.isPending}>
                    <Check className="size-4" />
                    Save
                  </Button>
                  <Button type="button" variant="outline" onClick={() => setEditing(false)}>
                    Cancel
                  </Button>
                </div>
              </form>
            )}
          </div>
        </AsyncBoundary>
      </DataSurface>
    </div>
  );
}

function maskKey(key: string | null): string {
  if (!key) return "-";
  if (key.length <= 8) return "****";
  return `${key.slice(0, 4)}...${key.slice(-4)}`;
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 py-1 sm:grid-cols-[12rem_1fr] sm:items-center">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate font-mono text-sm text-foreground">{value}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}
