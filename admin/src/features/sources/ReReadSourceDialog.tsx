import { toast } from "sonner";
import { errorMessage } from "@/lib/errors";
import { formatCount, pluralize } from "@/lib/format";
import { ConfirmDialog } from "@/patterns";
import { useReReadPreview, useReReadSource } from "./api";
import type { Source } from "./model/types";

interface ReReadSourceDialogProps {
  /** The source to re-read; null while the dialog is closed. */
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}

/**
 * Confirms re-reading every document of a source with the current models,
 * with the number of documents and the model calls it takes. A sync reads only
 * documents that changed, so this is how a source is read again after a model
 * changes.
 */
export function ReReadSourceDialog({ source, onOpenChange }: ReReadSourceDialogProps) {
  const preview = useReReadPreview(source?.id ?? null);
  const reRead = useReReadSource();
  const report = preview.data;
  const count = report?.available_unit_count ?? 0;

  async function confirm() {
    await reRead.mutateAsync(source!.id);
    toast.success(`Re-reading ${pluralize(count, "document")} in ${source!.name}`);
  }

  return (
    <ConfirmDialog
      open={source !== null}
      onOpenChange={onOpenChange}
      title={report ? `Re-read all ${pluralize(count, "document")}?` : "Re-read all documents?"}
      description={
        preview.isError ? (
          errorMessage(preview.error)
        ) : !report ? (
          "Counting the documents…"
        ) : (
          <>
            <span className="block">
              Every document in {source?.name} is processed again with the current models. This takes about{" "}
              {formatCount(report.estimated_model_calls)} model calls.
            </span>
            {report.unavailable_unit_count > 0 ? (
              <span className="mt-2 block">
                {pluralize(report.unavailable_unit_count, "document")} can't be re-read from what was stored and stay
                as they are.
              </span>
            ) : null}
            <span className="mt-2 block">Memories stay available while this runs.</span>
          </>
        )
      }
      confirmLabel="Re-read documents"
      ready={report !== undefined && count > 0}
      onConfirm={confirm}
    />
  );
}
