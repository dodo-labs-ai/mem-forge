import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { Source, SourceAgentEvaluationResponse } from "@/api/types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { OnlineEvaluationPresentation } from "@/views/evaluation/OnlineEvaluationPresentation";

/** Compatibility wrapper; contextual Source actions now navigate to /evaluation. */
export function SourceOnlineEvaluationDialog({
  source,
  onOpenChange,
}: {
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}) {
  const open = Boolean(source);
  const [days, setDays] = useState(1);
  const evaluationQuery = useQuery<SourceAgentEvaluationResponse>({
    queryKey: ["source-agent-evaluation", source?.id, days],
    queryFn: () => {
      if (!source) throw new Error("source is required");
      return resourceClient
        .get(`/sources/${source.id}/agent-evaluation`, { params: { days } })
        .then((response) => response.data);
    },
    enabled: open,
  });

  if (!source) return null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] grid-rows-[auto_minmax(0,1fr)] overflow-hidden sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>Online evaluation</DialogTitle>
          <DialogDescription>
            Detect live-traffic problems, inspect representative cases, and hand a bounded diagnosis task to an agent.
          </DialogDescription>
        </DialogHeader>
        {evaluationQuery.isPending ? (
          <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading evaluation results...
          </div>
        ) : evaluationQuery.isError || !evaluationQuery.data ? (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            Failed to load online evaluation results.
          </div>
        ) : (
          <OnlineEvaluationPresentation
            data={evaluationQuery.data}
            days={days}
            onDaysChange={setDays}
            scopeName={source.name}
          />
        )}
        <DialogFooter showCloseButton />
      </DialogContent>
    </Dialog>
  );
}
