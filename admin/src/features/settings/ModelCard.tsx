import { useId } from "react";
import { StatusBadge } from "@/patterns";
import { KIND_DESCRIPTIONS, KIND_TITLES, WITHOUT_A_MODEL } from "./model/kinds";
import { stateBadge } from "./model/modelStatus";
import type { ModelKind, ModelKindSettings, ModelProvider, VectorIndex } from "./model/types";
import { ModelForm } from "./ModelForm";
import { ModelInUse } from "./ModelInUse";
import { SearchIndex } from "./SearchIndex";

interface ModelCardProps {
  kind: ModelKind;
  settings: ModelKindSettings;
  providers: readonly ModelProvider[];
  /** The search index, on the card of the embedding model, which builds it. */
  vectorIndex?: VectorIndex;
  /** False when the deployment manages the models, so the card only shows them. */
  writable: boolean;
  onUnsavedChange: (kind: ModelKind, unsaved: boolean) => void;
}

/**
 * One kind of model: the one in use with its state, and the form that changes
 * it. While none is set up, the card says what MemForge does without one. The
 * embedding model also shows the search index it builds.
 */
export function ModelCard({ kind, settings, providers, vectorIndex, writable, onUnsavedChange }: ModelCardProps) {
  const titleId = useId();
  const badge = stateBadge(settings.active);
  const title = KIND_TITLES[kind];

  return (
    <section aria-labelledby={titleId} className="space-y-4 rounded-xl border bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-0.5">
          <h2 id={titleId} className="text-base font-semibold text-foreground">
            {title}
          </h2>
          <p className="text-sm text-muted-foreground">{KIND_DESCRIPTIONS[kind]}</p>
          {settings.active === null ? <p className="text-sm text-subtle-foreground">{WITHOUT_A_MODEL[kind]}</p> : null}
        </div>
        <StatusBadge tone={badge.tone}>{badge.label}</StatusBadge>
      </div>
      {settings.active !== null ? <ModelInUse kind={kind} model={settings.active} providers={providers} /> : null}
      {vectorIndex ? <SearchIndex vectorIndex={vectorIndex} embedding={settings} providers={providers} /> : null}
      {writable ? (
        <ModelForm
          kind={kind}
          title={title}
          settings={settings}
          providers={providers}
          vectorIndex={vectorIndex}
          onUnsavedChange={onUnsavedChange}
        />
      ) : null}
    </section>
  );
}
