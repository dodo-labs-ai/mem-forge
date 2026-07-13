import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import { resourceClient } from "@/api/client";
import type { Source, SourceAccessPolicy } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { SourceAccessSelection } from "./SourceAccessSection";

export function SourceAccessChangeDialog({
  source,
  onOpenChange,
}: {
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={Boolean(source)} onOpenChange={onOpenChange}>
      {source && (
        <SourceAccessChangeDialogContent
          key={`${source.id}:${source.access_policy}`}
          source={source}
          onClose={() => onOpenChange(false)}
        />
      )}
    </Dialog>
  );
}

function SourceAccessChangeDialogContent({
  source,
  onClose,
}: {
  source: Source;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [targetPolicy, setTargetPolicy] = useState<SourceAccessPolicy>(
    () => oppositePolicy(source.access_policy),
  );

  const changeAccess = useMutation({
    mutationFn: async () => {
      return resourceClient.post(
        `/sources/${source.id}/access-transitions`,
        { target_policy: targetPolicy },
        { headers: { "Idempotency-Key": crypto.randomUUID() } },
      );
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["sources"] });
      onClose();
    },
  });

  const currentLabel = source.access_policy === "private"
    ? "Only you can use it now."
    : "Everyone in this workspace can use it now.";

  return (
    <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Change access for {source.name}</DialogTitle>
          <DialogDescription>
            {currentLabel} Changing access updates the source and its existing memories as one durable operation.
          </DialogDescription>
        </DialogHeader>

        <SourceAccessSelection value={targetPolicy} onChange={setTargetPolicy} />

        <div className="rounded-lg border border-amber-200 bg-amber-50/60 p-3 text-xs leading-relaxed text-amber-900 dark:border-amber-900 dark:bg-amber-900/20 dark:text-amber-100">
          Sync pauses while existing memories are updated. Until the operation completes, only the source owner can query them.
        </div>

        {changeAccess.isError && (
          <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            {accessChangeError(changeAccess.error)}
          </div>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            disabled={changeAccess.isPending}
            onClick={onClose}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={
              changeAccess.isPending
              || targetPolicy === source.access_policy
            }
            onClick={() => changeAccess.mutate()}
          >
            {changeAccess.isPending && <Loader2 className="size-4 animate-spin" />}
            Change access
          </Button>
        </DialogFooter>
    </DialogContent>
  );
}

function oppositePolicy(policy: SourceAccessPolicy): SourceAccessPolicy {
  return policy === "private" ? "workspace" : "private";
}

function accessChangeError(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return "Access could not be changed. Review the source state and try again.";
}
