import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { resourceClient } from "@/api/client";
import type { DismissedRelation, Memory, MemoryRelation, RelatedMemory } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { timeAgo } from "@/utils/date";
import {
  RELATION_DISMISS_ACTIONS,
  RELATION_LABEL_NAMES,
  relatedMemorySources,
  relationSentence,
} from "./relations";

function errorDetail(error: unknown): string | null {
  if (!error) return null;
  const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
  if (typeof detail === "string") return detail;
  return error instanceof Error ? error.message : "The request failed.";
}

function CounterpartLine({ memory }: { memory: RelatedMemory }) {
  const sources = relatedMemorySources(memory);
  const recorded = memory.evidence_time;
  return (
    <div className="space-y-1">
      <Link to={`/memories/${memory.memory_id}`} className="text-sm text-primary hover:underline">
        {memory.summary}
      </Link>
      {(sources || recorded) && (
        <div className="text-xs text-muted-foreground">
          {sources}
          {sources && recorded ? " · " : ""}
          {recorded ? `recorded ${recorded}` : ""}
        </div>
      )}
    </div>
  );
}

/**
 * Current Cross-Document Relations of one Memory and the dismissals a person
 * can undo. A relation is a hint for readers; dismissing it changes no Memory.
 */
export function MemoryRelationsCard({ memory }: { memory: Memory }) {
  const queryClient = useQueryClient();
  const relations = memory.relations ?? [];
  const dismissed = memory.dismissed_relations ?? [];
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["memory", memory.id] });
    queryClient.invalidateQueries({ queryKey: ["memory-relations"] });
  };
  const dismiss = useMutation({
    mutationFn: (relation: MemoryRelation) =>
      resourceClient.post(
        `/memories/${memory.id}/relations/${relation.counterpart.memory_id}/dismissal`,
        {
          label: relation.label,
          expected_content_hash: memory.content_hash,
          counterpart_expected_content_hash: relation.counterpart.content_hash,
        },
      ),
    onSettled: refresh,
  });
  const restore = useMutation({
    mutationFn: (dismissal: DismissedRelation) =>
      resourceClient.delete(`/memories/${memory.id}/relations/${dismissal.counterpart.memory_id}/dismissal`),
    onSettled: refresh,
  });

  if (relations.length === 0 && dismissed.length === 0) return null;
  const failure = errorDetail(dismiss.error) ?? errorDetail(restore.error);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Related Memories in other documents</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {failure && <p className="text-sm text-destructive">{failure}</p>}
        {relations.map((relation) => (
          <div key={relation.counterpart.memory_id} className="rounded-lg border p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={relation.label === "contradicts" ? "destructive" : "secondary"}>
                  {RELATION_LABEL_NAMES[relation.label]}
                </Badge>
                <span className="text-xs text-muted-foreground">{relationSentence(relation)}</span>
                {relation.decided_by === "review" && <Badge variant="outline">Confirmed by a reviewer</Badge>}
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={dismiss.isPending || restore.isPending}
                onClick={() => dismiss.mutate(relation)}
              >
                {RELATION_DISMISS_ACTIONS[relation.label]}
              </Button>
            </div>
            <CounterpartLine memory={relation.counterpart} />
            {relation.reason && <p className="mt-2 text-xs text-muted-foreground">{relation.reason}</p>}
          </div>
        ))}
        {dismissed.map((dismissal) => (
          <div key={dismissal.counterpart.memory_id} className="rounded-lg border border-dashed p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">
                {dismissal.labels.map((label) => RELATION_DISMISS_ACTIONS[label]).join(", ")}: dismissed by{" "}
                {dismissal.dismissed_by} {timeAgo(dismissal.dismissed_at)}
              </span>
              <Button
                variant="ghost"
                size="sm"
                disabled={dismiss.isPending || restore.isPending}
                onClick={() => restore.mutate(dismissal)}
              >
                Undo
              </Button>
            </div>
            <CounterpartLine memory={dismissal.counterpart} />
            {dismissal.note && <p className="mt-2 text-xs text-muted-foreground">{dismissal.note}</p>}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
