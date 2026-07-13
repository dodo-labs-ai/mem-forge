import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import { resourceClient } from "@/api/client";
import type { GeneConfigSchema, Source } from "@/api/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

import { SchemaSourceSetup } from "./SchemaSourceSetup";
import { isSchemaSourceType } from "./sourceSetupAdapters";
import { TeamsSourceSetup } from "./TeamsSourceSetup";

export function SourceSetupDialog({
  open,
  onOpenChange,
  sourceType,
  source,
  onSaved,
  initialFocus,
  onRequestAccessChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceType: string | null;
  source?: Source | null;
  onSaved?: (sourceId: string) => void;
  initialFocus?: { step: "project" };
  onRequestAccessChange?: (source: Source) => void;
}) {
  if (!sourceType) return null;
  if (sourceType === "teams") {
    return (
      <TeamsSetupDialog
        open={open}
        onOpenChange={onOpenChange}
        source={source}
        onSaved={onSaved}
        initialFocus={initialFocus}
        onRequestAccessChange={onRequestAccessChange}
      />
    );
  }
  if (!isSchemaSourceType(sourceType)) {
    throw new Error(`No source setup UI registered for ${sourceType}`);
  }
  return (
    <SchemaSourceSetup
      open={open}
      onOpenChange={onOpenChange}
      sourceType={sourceType}
      source={source}
      onSaved={onSaved}
      initialFocus={initialFocus}
      onRequestAccessChange={onRequestAccessChange}
    />
  );
}

function TeamsSetupDialog({
  open,
  onOpenChange,
  source,
  onSaved,
  initialFocus,
  onRequestAccessChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  source?: Source | null;
  onSaved?: (sourceId: string) => void;
  initialFocus?: { step: "project" };
  onRequestAccessChange?: (source: Source) => void;
}) {
  const canConfigure = source ? source.capabilities?.can_configure === true : true;
  const schemaQuery = useQuery<GeneConfigSchema>({
    queryKey: ["gene-config-schema", "teams"],
    queryFn: () => resourceClient.get("/genes/teams/config-schema").then((response) => response.data),
    enabled: open && canConfigure,
  });
  if (!canConfigure) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden p-0 sm:max-w-3xl">
        {schemaQuery.isPending ? (
          <div className="flex items-center justify-center gap-2 p-12 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading source setup...
          </div>
        ) : schemaQuery.isError || !schemaQuery.data ? (
          <div className="p-4">
            <DialogHeader><DialogTitle>Microsoft Teams setup</DialogTitle></DialogHeader>
            <div className="mt-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
              Failed to load Teams source settings.
            </div>
          </div>
        ) : (
          <TeamsSourceSetup
            source={source}
            schema={schemaQuery.data}
            onOpenChange={onOpenChange}
            onSaved={onSaved}
            initialFocus={initialFocus}
            onRequestAccessChange={onRequestAccessChange}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
