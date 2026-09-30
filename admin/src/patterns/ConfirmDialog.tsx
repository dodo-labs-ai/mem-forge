import { useState, type ReactNode } from "react";
import { errorMessage } from "@/lib/errors";
import { Button } from "@/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/ui/dialog";
import { Input } from "@/ui/input";
import { Label } from "@/ui/label";

interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  /** `danger` styles the confirm button for irreversible actions. */
  tone?: "default" | "danger";
  /** When set, the user must type this text before confirming. */
  confirmationText?: string;
  /** Runs the action. The dialog stays open and shows the error if it throws. */
  onConfirm: () => Promise<unknown>;
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  tone = "default",
  confirmationText,
  onConfirm,
}: ConfirmDialogProps) {
  const [typed, setTyped] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const confirmed = confirmationText === undefined || typed === confirmationText;

  function reset(nextOpen: boolean) {
    if (pending) return;
    if (!nextOpen) {
      setTyped("");
      setError(null);
    }
    onOpenChange(nextOpen);
  }

  async function confirm() {
    setPending(true);
    setError(null);
    try {
      await onConfirm();
      setTyped("");
      onOpenChange(false);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setPending(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={reset}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        {confirmationText !== undefined ? (
          <div className="space-y-1.5">
            <Label htmlFor="confirm-dialog-text">
              Type <span className="font-mono font-medium">{confirmationText}</span> to confirm
            </Label>
            <Input
              id="confirm-dialog-text"
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              autoComplete="off"
            />
          </div>
        ) : null}
        {error ? (
          <p role="alert" className="text-sm text-tone-danger-foreground">
            {error}
          </p>
        ) : null}
        <DialogFooter>
          <Button variant="outline" onClick={() => reset(false)} disabled={pending}>
            Cancel
          </Button>
          <Button
            variant={tone === "danger" ? "destructive" : "default"}
            onClick={confirm}
            disabled={!confirmed || pending}
          >
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
