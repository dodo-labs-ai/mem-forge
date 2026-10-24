import { useState } from "react";
import { PauseCircle } from "lucide-react";
import { toast } from "sonner";
import { errorMessage } from "@/lib/errors";
import { ChoiceGroup, Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { useChangeSourceAccess } from "./api";
import { ACCESS_CHOICES, accessChangeRefusal, oppositeAccessPolicy, type AccessPolicy } from "./model/sourceAccess";
import type { Source } from "./model/types";

interface SourceAccessChangeDialogProps {
  /** The source whose access changes; null while the dialog is closed. */
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}

/**
 * Changes who can use a source in a shared workspace. The change covers the
 * source and every memory it already made, as one durable operation that
 * the source list follows until it ends.
 */
export function SourceAccessChangeDialog({ source, onOpenChange }: SourceAccessChangeDialogProps) {
  return (
    <Dialog open={source !== null} onOpenChange={onOpenChange}>
      {source ? (
        <AccessChangeContent key={`${source.id}:${source.access_policy}`} source={source} onClose={() => onOpenChange(false)} />
      ) : null}
    </Dialog>
  );
}

function AccessChangeContent({ source, onClose }: { source: Source; onClose: () => void }) {
  const changeAccess = useChangeSourceAccess();
  const [target, setTarget] = useState<AccessPolicy>(() => oppositeAccessPolicy(source.access_policy));
  // One id per dialog, so a repeated click starts the same change.
  const [requestId] = useState(() => crypto.randomUUID());
  const current = source.access_policy === "private" ? "Only you can use it now." : "Everyone in this workspace can use it now.";

  function change() {
    changeAccess.mutate(
      { sourceId: source.id, targetPolicy: target, requestId },
      {
        onSuccess: () => {
          toast.success(`Changing who can use ${source.name}`);
          onClose();
        },
      },
    );
  }

  return (
    <DialogContent showCloseButton={false} className="sm:max-w-xl">
      <DialogHeader>
        <DialogTitle>Change who can use {source.name}</DialogTitle>
        <DialogDescription>
          {current} The change covers the source and every memory it already made, as one operation.
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-1.5">
        <p id="source-access-change-label" className="text-sm font-medium">
          Who can use this source?
        </p>
        <ChoiceGroup
          id="source-access-change"
          aria-labelledby="source-access-change-label"
          choices={ACCESS_CHOICES}
          value={target}
          onValueChange={(next) => setTarget(next as AccessPolicy)}
          disabled={changeAccess.isPending}
        />
      </div>
      <Notice tone="warn" icon={PauseCircle}>
        Syncing pauses while the existing memories are updated. Until the change is finished, only the source owner can
        find the source or query its memories.
      </Notice>
      {changeAccess.isError ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          {accessChangeRefusal(errorMessage(changeAccess.error))}
        </p>
      ) : null}
      <DialogFooter>
        <Button variant="outline" disabled={changeAccess.isPending} onClick={onClose}>
          Cancel
        </Button>
        <Button disabled={changeAccess.isPending || target === source.access_policy} onClick={change}>
          Change access
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}
