import { zodResolver } from "@hookform/resolvers/zod";
import { Controller, useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { z } from "zod";
import { memoryPath, reviewPath } from "@/lib/paths";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Input } from "@/ui/input";
import { Label } from "@/ui/label";
import { Textarea } from "@/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/ui/toggle-group";
import { useProposeCorrection } from "./api";
import { actionErrorMessage } from "./model/actionErrors";
import type { CorrectionKind, MemoryDetail } from "./model/types";

const CORRECTION_KINDS: { value: CorrectionKind; label: string }[] = [
  { value: "revision", label: "Revision (same fact, fixed wording)" },
  { value: "supersession", label: "Supersession (the fact changed)" },
];

const correctionSchema = z.object({
  replacementContent: z.string().trim().min(1, "Write the corrected statement."),
  replacementKind: z.enum(["revision", "supersession"]),
  provenance: z.string().trim().min(1, "Add a link or note that supports the correction."),
  reason: z.string().trim().min(1, "Say why the current memory is wrong."),
});
type CorrectionForm = z.infer<typeof correctionSchema>;

interface ProposeCorrectionDialogProps {
  memory: MemoryDetail;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function FieldHint({ id, error, hint }: { id: string; error?: string; hint?: string }) {
  const text = error ?? hint;
  if (!text) return null;
  return (
    <p id={id} className={error ? "text-xs text-tone-danger-foreground" : "text-xs text-muted-foreground"}>
      {text}
    </p>
  );
}

export function ProposeCorrectionDialog({ memory, open, onOpenChange }: ProposeCorrectionDialogProps) {
  const navigate = useNavigate();
  const propose = useProposeCorrection(memory);
  const form = useForm<CorrectionForm>({
    resolver: zodResolver(correctionSchema),
    defaultValues: { replacementContent: memory.content, replacementKind: "supersession", provenance: "", reason: "" },
  });
  const { errors } = form.formState;

  function close(nextOpen: boolean) {
    if (propose.isPending) return;
    if (!nextOpen) {
      form.reset();
      propose.reset();
    }
    onOpenChange(nextOpen);
  }

  const submit = form.handleSubmit(async (values) => {
    const result = await propose.mutateAsync(values);
    close(false);
    if (result.outcome === "applied") {
      toast.success("Correction applied");
      navigate(memoryPath(result.replacement_memory_id));
    } else {
      toast.success("Correction sent for review", {
        action: { label: "Open review", onClick: () => navigate(reviewPath(result.review_id)) },
      });
    }
  });

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="sm:max-w-xl">
        <form onSubmit={(event) => void submit(event).catch(() => undefined)} className="grid gap-4">
          <DialogHeader>
            <DialogTitle>Propose a correction</DialogTitle>
            <DialogDescription>
              Write the corrected statement. If you manage every source behind this memory, it applies right away.
              Otherwise it opens a review.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-1.5">
            <span className="text-sm font-medium">Current</span>
            <p className="rounded-lg bg-surface-subtle px-3 py-2.5 text-sm text-subtle-foreground">{memory.content}</p>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="correction-content">
              Corrected statement <span className="text-tone-danger">*</span>
            </Label>
            <Textarea
              id="correction-content"
              rows={3}
              aria-invalid={errors.replacementContent ? true : undefined}
              aria-describedby="correction-content-hint"
              {...form.register("replacementContent")}
            />
            <FieldHint id="correction-content-hint" error={errors.replacementContent?.message} />
          </div>
          <div className="grid gap-1.5">
            <span id="correction-kind-label" className="text-sm font-medium">
              Kind
            </span>
            <Controller
              control={form.control}
              name="replacementKind"
              render={({ field }) => (
                <ToggleGroup
                  value={[field.value]}
                  onValueChange={(values: string[]) => {
                    const kind = CORRECTION_KINDS.find((option) => option.value === values[0]);
                    if (kind) field.onChange(kind.value);
                  }}
                  variant="outline"
                  size="sm"
                  aria-labelledby="correction-kind-label"
                  className="w-full"
                >
                  {CORRECTION_KINDS.map((kind) => (
                    <ToggleGroupItem key={kind.value} value={kind.value} className="flex-1">
                      {kind.label}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              )}
            />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="correction-provenance">
              Evidence <span className="text-tone-danger">*</span>
            </Label>
            <Input
              id="correction-provenance"
              aria-invalid={errors.provenance ? true : undefined}
              aria-describedby="correction-provenance-hint"
              {...form.register("provenance")}
            />
            <FieldHint
              id="correction-provenance-hint"
              error={errors.provenance?.message}
              hint="Link or note that supports the correction."
            />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="correction-reason">
              Reason <span className="text-tone-danger">*</span>
            </Label>
            <Input
              id="correction-reason"
              placeholder="Why the current memory is wrong"
              aria-invalid={errors.reason ? true : undefined}
              aria-describedby="correction-reason-hint"
              {...form.register("reason")}
            />
            <FieldHint id="correction-reason-hint" error={errors.reason?.message} />
          </div>
          {propose.isError ? (
            <p role="alert" className="rounded-lg bg-tone-danger-soft px-3.5 py-3 text-sm text-tone-danger-foreground">
              {actionErrorMessage(propose.error)}
            </p>
          ) : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => close(false)} disabled={propose.isPending}>
              Cancel
            </Button>
            <Button type="submit" disabled={propose.isPending}>
              Submit correction
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
