import { Undo2 } from "lucide-react";
import { useState } from "react";
import { formatRelative } from "@/lib/format";
import { ErrorNotice, StatusBadge } from "@/patterns";
import { Button } from "@/ui/button";
import { useRestoreRelation } from "./api";
import { DetailCard } from "./DetailCard";
import { DismissRelationDialog } from "./DismissRelationDialog";
import { RelationLabelBadge } from "./MemoryBadges";
import { actionErrorMessage } from "./model/actionErrors";
import { RELATION_DISMISS_ACTIONS, dismissedLabelsText, relationSentence } from "./model/relations";
import type { MemoryDetail, MemoryRelation } from "./model/types";
import { RelatedMemorySummary } from "./RelationCells";

interface RelatedMemoriesCardProps {
  memory: MemoryDetail;
  typeLabels: Readonly<Record<string, string>>;
}

/**
 * The current relations of one Memory to Memories in other documents, and
 * the dismissals a person can undo. A relation is a hint for readers;
 * dismissing it changes no Memory.
 */
export function RelatedMemoriesCard({ memory, typeLabels }: RelatedMemoriesCardProps) {
  const relations = memory.relations ?? [];
  const dismissed = memory.dismissed_relations ?? [];
  const restore = useRestoreRelation(memory);
  const [dismissing, setDismissing] = useState<MemoryRelation | null>(null);
  if (relations.length === 0 && dismissed.length === 0) return null;

  return (
    <DetailCard title="Related memories in other documents" meta={`${relations.length} current`}>
      <div className="flex flex-col gap-2">
        {relations.map((relation) => (
          <div key={`${relation.label}:${relation.counterpart.memory_id}`} className="flex flex-col gap-2 rounded-lg border px-3.5 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <RelationLabelBadge label={relation.label} />
              <span className="text-sm text-subtle-foreground">{relationSentence(relation)}</span>
              {relation.decided_by === "review" ? <StatusBadge tone="ok">Confirmed by a reviewer</StatusBadge> : null}
              <Button size="sm" variant="outline" className="ml-auto" onClick={() => setDismissing(relation)}>
                {RELATION_DISMISS_ACTIONS[relation.label]}
              </Button>
            </div>
            <RelatedMemorySummary memory={relation.counterpart} typeLabels={typeLabels} />
          </div>
        ))}
        {dismissed.map((dismissal) => (
          <div key={dismissal.counterpart.memory_id} className="flex flex-col gap-2 rounded-lg border border-dashed px-3.5 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted-foreground">
                {dismissedLabelsText(dismissal.labels)}: dismissed by {dismissal.dismissed_by} {formatRelative(dismissal.dismissed_at)}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto"
                disabled={restore.isPending}
                onClick={() => restore.mutate(dismissal)}
              >
                <Undo2 />
                Undo
              </Button>
            </div>
            <RelatedMemorySummary memory={dismissal.counterpart} typeLabels={typeLabels} />
            {dismissal.note ? <p className="text-xs text-muted-foreground">Note: {dismissal.note}</p> : null}
          </div>
        ))}
        {restore.isError ? (
          <ErrorNotice title="Could not undo the dismissal" message={actionErrorMessage(restore.error)} />
        ) : null}
      </div>
      <DismissRelationDialog memory={memory} relation={dismissing} onClose={() => setDismissing(null)} />
    </DetailCard>
  );
}
