import { useState } from "react";
import { toast } from "sonner";
import { formatCount } from "@/lib/format";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Label } from "@/ui/label";
import { Textarea } from "@/ui/textarea";
import { useDismissRelation } from "./api";
import { DISMISSAL_NOTE_MAX_CHARS } from "./constants";
import { actionErrorMessage } from "./model/actionErrors";
import { RELATION_DISMISS_ACTIONS, dismissalDescription, dismissalTitle } from "./model/relations";
import type { MemoryDetail, MemoryRelation } from "./model/types";

interface DismissRelationDialogProps {
  memory: MemoryDetail;
  /** The relation to dismiss; the dialog is open while one is set. */
  relation: MemoryRelation | null;
  onClose: () => void;
}

/** Asks before a person says a relation does not hold, with an optional note for others. */
export function DismissRelationDialog({ memory, relation, onClose }: DismissRelationDialogProps) {
  const dismiss = useDismissRelation(memory);
  const [note, setNote] = useState("");

  function close() {
    if (dismiss.isPending) return;
    setNote("");
    dismiss.reset();
    onClose();
  }

  async function confirm() {
    if (!relation) return;
    try {
      await dismiss.mutateAsync({ relation, note });
      toast.success(`Marked as ${RELATION_DISMISS_ACTIONS[relation.label].toLowerCase()}`);
      close();
    } catch {
      // The dialog stays open and shows the reason.
    }
  }

  return (
    <Dialog open={relation !== null} onOpenChange={(open) => (open ? undefined : close())}>
      {relation ? (
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{dismissalTitle(relation.label)}</DialogTitle>
            <DialogDescription>{dismissalDescription(relation.label)}</DialogDescription>
          </DialogHeader>
          <dl className="grid gap-1 rounded-lg border px-3.5 py-3 text-sm">
            <dt className="text-xs font-semibold text-muted-foreground">This memory</dt>
            <dd className="mb-2">{memory.content}</dd>
            <dt className="text-xs font-semibold text-muted-foreground">Related memory</dt>
            <dd>{relation.counterpart.summary}</dd>
          </dl>
          <div className="grid gap-1.5">
            <Label htmlFor="dismissal-note">Note (optional)</Label>
            <Textarea
              id="dismissal-note"
              rows={3}
              value={note}
              maxLength={DISMISSAL_NOTE_MAX_CHARS}
              onChange={(event) => setNote(event.target.value)}
              placeholder={`Why this is ${RELATION_DISMISS_ACTIONS[relation.label].toLowerCase()}`}
              aria-describedby="dismissal-note-hint"
            />
            <p id="dismissal-note-hint" className="text-xs text-muted-foreground">
              Shown to others next to the dismissal. Up to {formatCount(DISMISSAL_NOTE_MAX_CHARS)} characters.
            </p>
          </div>
          {dismiss.isError ? (
            <p role="alert" className="rounded-lg bg-tone-danger-soft px-3.5 py-3 text-sm text-tone-danger-foreground">
              {actionErrorMessage(dismiss.error)}
            </p>
          ) : null}
          <DialogFooter>
            <Button variant="outline" onClick={close} disabled={dismiss.isPending}>
              Cancel
            </Button>
            <Button onClick={() => void confirm()} disabled={dismiss.isPending}>
              {RELATION_DISMISS_ACTIONS[relation.label]}
            </Button>
          </DialogFooter>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}
