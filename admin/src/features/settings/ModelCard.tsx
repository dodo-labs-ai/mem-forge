import { useId } from "react";
import { StatusBadge } from "@/patterns";
import { KIND_DESCRIPTIONS, KIND_TITLES } from "./model/kinds";
import { stateBadge } from "./model/modelStatus";
import type { ModelKind, ModelKindSettings, ModelProvider } from "./model/types";
import { ModelForm } from "./ModelForm";
import { ModelInUse } from "./ModelInUse";

interface ModelCardProps {
  kind: ModelKind;
  settings: ModelKindSettings;
  providers: readonly ModelProvider[];
  /** False when the deployment manages the models, so the card only shows them. */
  writable: boolean;
  onUnsavedChange: (kind: ModelKind, unsaved: boolean) => void;
}

/** One kind of model: the one in use with its state, and the form that changes it. */
export function ModelCard({ kind, settings, providers, writable, onUnsavedChange }: ModelCardProps) {
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
        </div>
        <StatusBadge tone={badge.tone}>{badge.label}</StatusBadge>
      </div>
      <ModelInUse kind={kind} model={settings.active} providers={providers} />
      {writable ? (
        <ModelForm kind={kind} title={title} settings={settings} providers={providers} onUnsavedChange={onUnsavedChange} />
      ) : null}
    </section>
  );
}
