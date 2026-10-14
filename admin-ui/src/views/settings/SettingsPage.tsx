import { useQuery } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { ModelBinding, ModelKind, ModelSettings, VectorIndex } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Badge } from "@/components/ui/badge";

/** Where the image serves the new admin UI's Settings page, beside this one at `/`. */
const NEW_ADMIN_SETTINGS_PATH = "/v2/settings";

const KINDS: Array<{ kind: ModelKind; title: string; description: string }> = [
  { kind: "generation", title: "Generation model", description: "Reads source content and writes memories." },
  { kind: "embedding", title: "Embedding model", description: "Turns memories into vectors for search." },
];

const STATE_LABELS: Record<ModelBinding["state"], string> = {
  ready: "Ready",
  not_validated: "Not validated",
  needs_revalidation: "Needs revalidation",
};

/** The state of the search index in one line, or `null` when there is no index to speak of. */
function searchIndexLine({ status, index }: VectorIndex): string | null {
  if (status === "ready") {
    return index ? `Search index: ready, with ${index.vector_count.toLocaleString("en-US")} memories.` : "Search index: ready.";
  }
  if (status === "reindex_required") {
    return "Search index: has to be rebuilt for this model. Vector search is off until it is; start the rebuild in Settings in the new admin.";
  }
  if (status === "reindexing") {
    return "Search index: being rebuilt. Source syncs and changes to memories wait until the rebuild has finished.";
  }
  if (status === "awaiting_validation") return "Search index: none yet. It is built once the model has passed validation.";
  return null;
}

/**
 * @deprecated admin-ui-v1: shows the two models in use and the state of the
 * search index. Models are changed and validated, and the index is rebuilt,
 * on the Settings page of the new admin UI.
 */
export function SettingsPage() {
  const settingsQuery = useQuery<ModelSettings>({
    queryKey: ["model-settings"],
    queryFn: () => resourceClient.get("/model-settings").then((response) => response.data),
  });
  const settings = settingsQuery.data;

  return (
    <div className="max-w-3xl space-y-4">
      <PageHeader title="Settings" description="The models MemForge uses to write memories and to search them." />

      <AsyncBoundary
        isLoading={settingsQuery.isLoading}
        isError={settingsQuery.isError}
        error={settingsQuery.error}
        onRetry={() => settingsQuery.refetch()}
        isEmpty={!settings}
        empty={
          <EmptyState
            icon={SlidersHorizontal}
            title="No model settings found"
            description="The admin API did not return the model settings."
          />
        }
      >
        {settings && (
          <>
            <p className="text-sm text-muted-foreground">
              {settings.writable ? (
                <>
                  To change or validate a model, open{" "}
                  <a href={NEW_ADMIN_SETTINGS_PATH} className="font-medium text-foreground underline underline-offset-4">
                    Settings in the new admin
                  </a>
                  .
                </>
              ) : (
                "The deployment environment manages these models, so they cannot be changed in the admin."
              )}
            </p>
            {KINDS.map(({ kind, title, description }) => (
              <ModelSection
                key={kind}
                title={title}
                description={description}
                model={settings[kind].active}
                note={kind === "embedding" ? searchIndexLine(settings.vector_index) : null}
              />
            ))}
          </>
        )}
      </AsyncBoundary>
    </div>
  );
}

interface ModelSectionProps {
  title: string;
  description: string;
  model: ModelBinding | null;
  /** One more line about the model, such as the state of the index it builds. */
  note: string | null;
}

function ModelSection({ title, description, model, note }: ModelSectionProps) {
  return (
    <DataSurface>
      <section aria-label={title} className="space-y-4 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">{title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          </div>
          <Badge variant={model?.state === "ready" ? "secondary" : "outline"}>
            {model ? STATE_LABELS[model.state] : "Not set up"}
          </Badge>
        </div>
        {model && (
          <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Provider</dt>
            <dd>{model.provider}</dd>
            <dt className="text-muted-foreground">Endpoint</dt>
            <dd className="break-all font-mono">{model.endpoint ?? "Decided by the provider"}</dd>
            <dt className="text-muted-foreground">Model</dt>
            <dd className="break-all font-mono">{model.model_id}</dd>
          </dl>
        )}
        {note && <p className="text-sm text-muted-foreground">{note}</p>}
      </section>
    </DataSurface>
  );
}
