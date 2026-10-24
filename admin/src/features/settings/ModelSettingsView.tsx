import { Lock } from "lucide-react";
import { useCallback, useState } from "react";
import { MODEL_KINDS } from "@/api";
import { Notice, useUnsavedChangesGuard } from "@/patterns";
import { LeaveDialog } from "./LeaveDialog";
import type { ModelKind, ModelProvider, ModelSettings } from "./model/types";
import { ModelCard } from "./ModelCard";

interface ModelSettingsViewProps {
  settings: ModelSettings;
  providers: readonly ModelProvider[];
}

/**
 * The generation and the embedding model, each on its own card with its own
 * form; the embedding model's card also holds the search index. When the
 * deployment environment manages the models, the server says so up front and
 * the cards only show what is in use.
 */
export function ModelSettingsView({ settings, providers }: ModelSettingsViewProps) {
  const [unsaved, setUnsaved] = useState<Record<ModelKind, boolean>>({ generation: false, embedding: false });
  const changed = MODEL_KINDS.filter((kind) => unsaved[kind]);
  const blocker = useUnsavedChangesGuard(changed.length > 0);
  const noteUnsaved = useCallback(
    (kind: ModelKind, value: boolean) => setUnsaved((current) => (current[kind] === value ? current : { ...current, [kind]: value })),
    [],
  );

  return (
    <div className="space-y-4">
      {settings.writable ? null : (
        <Notice tone="live" icon={Lock}>
          <strong className="font-semibold">Managed by the deployment environment.</strong> These settings come from
          environment variables and cannot be changed here. Change them in the environment and restart the service.
        </Notice>
      )}
      {MODEL_KINDS.map((kind) => (
        <ModelCard
          key={kind}
          kind={kind}
          settings={settings[kind]}
          providers={providers}
          vectorIndex={kind === "embedding" ? settings.vector_index : undefined}
          writable={settings.writable}
          onUnsavedChange={noteUnsaved}
        />
      ))}
      <LeaveDialog blocker={blocker} changed={changed} />
    </div>
  );
}
