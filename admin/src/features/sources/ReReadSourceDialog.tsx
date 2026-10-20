import { toast } from "sonner";
import { ApiError } from "@/api";
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

/** What the server's refusal of a re-read means for the source. */
const REFUSALS: Record<string, string> = {
  source_has_no_documents: "This source has no documents yet. Sync it first.",
  projection_scope_transition_open: "The source's scope is changing. Re-read it once that change is finished.",
};

function refusal(error: unknown): string {
  const message = errorMessage(error);
  return REFUSALS[message] ?? message;
}

/** Whether the dry run failed to answer, as a server error or a lost request does, rather than refused. */
function unanswered(error: unknown): boolean {
  return !(error instanceof ApiError) || error.status >= 500;
}

/**
 * Confirms re-reading every document of a source with the current models. A
 * sync reads only documents that changed, so this is how a source is read
 * again after a model changes. A dry run first counts the documents that can
 * be re-read from what was stored, and the model calls they take, and only
 * those are re-read. When the dry run cannot answer, re-reading is still
 * offered for every document; when the server refuses it, the refusal shows.
 */
export function ReReadSourceDialog({ source, onOpenChange }: ReReadSourceDialogProps) {
  const preview = useReReadPreview(source?.id ?? null);
  const reRead = useReReadSource();
  const report = preview.data;
  const readable = report?.units.filter((unit) => unit.available).map((unit) => unit.document_id) ?? [];
  const refused = preview.isError ? refusal(preview.error) : null;
  const unestimated = preview.isError && unanswered(preview.error);

  async function confirm() {
    await reRead.mutateAsync({ sourceId: source!.id, documentIds: report ? readable : null });
    toast.success(
      report ? `Re-reading ${pluralize(readable.length, "document")} in ${source!.name}` : `Re-reading ${source!.name}`,
    );
  }

  function description() {
    if (unestimated) {
      return (
        <>
          <span className="block">Couldn't count the documents and model calls: {refused}</span>
          <span className="mt-2 block">
            Re-reading still processes every document of {source?.name}. Any that can't be re-read from what was stored
            show as failed items.
          </span>
        </>
      );
    }
    if (refused) return refused;
    if (!report) return "Counting the documents…";
    if (readable.length === 0) return "None of the documents of this source can be re-read from what was stored.";
    return (
      <>
        <span className="block">
          Every document in {source?.name} is processed again with the current models. This takes about{" "}
          {formatCount(report.estimated_model_calls)} model calls.
        </span>
        {report.unavailable_unit_count > 0 ? (
          <span className="mt-2 block">
            {pluralize(report.unavailable_unit_count, "document")} can't be re-read from what was stored and stay as
            they are.
          </span>
        ) : null}
        <span className="mt-2 block">Memories stay available while this runs.</span>
      </>
    );
  }

  return (
    <ConfirmDialog
      open={source !== null}
      onOpenChange={onOpenChange}
      title={report && readable.length > 0 ? `Re-read all ${pluralize(readable.length, "document")}?` : "Re-read all documents?"}
      description={description()}
      confirmLabel="Re-read documents"
      ready={unestimated || readable.length > 0}
      onConfirm={confirm}
    />
  );
}
