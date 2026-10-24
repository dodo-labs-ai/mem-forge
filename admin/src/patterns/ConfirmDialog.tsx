import { Copy } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
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
  /** The label of the button that closes the dialog without acting. */
  cancelLabel?: string;
  /** `danger` styles the confirm button for irreversible actions. */
  tone?: "default" | "danger";
  /** When set, the user must type this text before confirming. */
  confirmationText?: string;
  /**
   * When set with `confirmationText`, confirming takes two steps: the
   * description comes first with this button, and the typed confirmation
   * second, so the user reads the consequences before committing to them.
   */
  continueLabel?: string;
  /** False while the description is not ready to act on, for example while it loads. */
  ready?: boolean;
  /** Runs the action. The dialog stays open and shows the error if it throws. */
  onConfirm: () => Promise<unknown>;
}

function copyConfirmationText(text: string) {
  navigator.clipboard.writeText(text).then(
    () => toast.success("Name copied"),
    () => toast.error("Could not copy"),
  );
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  cancelLabel = "Cancel",
  tone = "default",
  confirmationText,
  continueLabel,
  ready = true,
  onConfirm,
}: ConfirmDialogProps) {
  const [typed, setTyped] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [continued, setContinued] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const confirmed = confirmationText === undefined || typed === confirmationText;
  const reviewing = confirmationText !== undefined && continueLabel !== undefined && !continued;
  const actionVariant = tone === "danger" ? "destructive" : "default";

  function clear() {
    setTyped("");
    setContinued(false);
    setError(null);
  }

  function reset(nextOpen: boolean) {
    if (pending) return;
    if (!nextOpen) clear();
    onOpenChange(nextOpen);
  }

  async function confirm() {
    setPending(true);
    setError(null);
    try {
      await onConfirm();
      clear();
      onOpenChange(false);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setPending(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={reset}>
      {/* A dialog that asks for the text at once opens on its input, not on the Copy button before it. */}
      <DialogContent
        showCloseButton={false}
        initialFocus={confirmationText !== undefined && continueLabel === undefined ? inputRef : undefined}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        {confirmationText !== undefined && !reviewing ? (
          <div className="space-y-1.5">
            <div className="flex items-start gap-2">
              {/* A sentence, not a flex row: a long name wraps inside it and can be selected. */}
              <Label htmlFor="confirm-dialog-text" className="block min-w-0 flex-1 leading-normal select-text">
                Type <span className="font-mono font-medium break-words select-all">{confirmationText}</span> to
                confirm
              </Label>
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                aria-label={`Copy ${confirmationText}`}
                onClick={() => copyConfirmationText(confirmationText)}
              >
                <Copy aria-hidden />
              </Button>
            </div>
            <Input
              ref={inputRef}
              id="confirm-dialog-text"
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              autoComplete="off"
              autoFocus={continueLabel !== undefined}
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
            {cancelLabel}
          </Button>
          {reviewing ? (
            <Button variant={actionVariant} onClick={() => setContinued(true)} disabled={!ready}>
              {continueLabel}
            </Button>
          ) : (
            <Button variant={actionVariant} onClick={confirm} disabled={!ready || !confirmed || pending}>
              {confirmLabel}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
