import { zodResolver } from "@hookform/resolvers/zod";
import { Archive } from "lucide-react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Label } from "@/ui/label";
import { Textarea } from "@/ui/textarea";
import { useRetireMemory } from "./api";
import { actionErrorMessage } from "./model/actionErrors";
import type { MemoryDetail } from "./model/types";

const retireSchema = z.object({ reason: z.string().trim().min(1, "Enter why this memory should be retired.") });
type RetireForm = z.infer<typeof retireSchema>;

interface RetireMemoryDialogProps {
  memory: MemoryDetail;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RetireMemoryDialog({ memory, open, onOpenChange }: RetireMemoryDialogProps) {
  const retire = useRetireMemory(memory);
  const form = useForm<RetireForm>({ resolver: zodResolver(retireSchema), defaultValues: { reason: "" } });
  const reasonError = form.formState.errors.reason?.message;

  function close(nextOpen: boolean) {
    if (retire.isPending) return;
    if (!nextOpen) {
      form.reset();
      retire.reset();
    }
    onOpenChange(nextOpen);
  }

  const submit = form.handleSubmit(async ({ reason }) => {
    await retire.mutateAsync(reason);
    toast.success("Memory retired");
    close(false);
  });

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent>
        <form onSubmit={(event) => void submit(event).catch(() => undefined)} className="grid gap-4">
          <DialogHeader>
            <DialogTitle>Retire this memory?</DialogTitle>
            <DialogDescription>It stays in history but leaves search and memory views. Agents stop receiving it.</DialogDescription>
          </DialogHeader>
          <blockquote className="rounded-lg border bg-surface-subtle px-3.5 py-3 text-sm text-foreground">{memory.content}</blockquote>
          <div className="grid gap-1.5">
            <Label htmlFor="retire-reason">
              Reason <span className="text-tone-danger">*</span>
            </Label>
            <Textarea
              id="retire-reason"
              rows={3}
              placeholder="For example: no longer true after the workflow change"
              aria-invalid={reasonError ? true : undefined}
              aria-describedby="retire-reason-hint"
              {...form.register("reason")}
            />
            <p id="retire-reason-hint" className="text-xs text-muted-foreground">
              {reasonError ?? "Recorded in the lifecycle history."}
            </p>
          </div>
          {retire.isError ? (
            <p role="alert" className="rounded-lg bg-tone-danger-soft px-3.5 py-3 text-sm text-tone-danger-foreground">
              {actionErrorMessage(retire.error)}
            </p>
          ) : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => close(false)} disabled={retire.isPending}>
              Cancel
            </Button>
            <Button type="submit" variant="destructive" disabled={retire.isPending}>
              <Archive />
              Retire memory
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
