import { toast } from "sonner";
import { useReReadImageDocuments } from "@/features/sources";
import { pluralize } from "@/lib/format";
import { ConfirmDialog } from "@/patterns";

interface ReReadImagesDialogProps {
  /** The documents with an image; null while the dialog is closed. */
  documentCount: number | null;
  onOpenChange: (open: boolean) => void;
}

/**
 * Offers to re-read the documents with an image once a generation model that
 * reads images replaces one that read text only. Those documents were read
 * without their images, and a sync reads only documents that changed.
 */
export function ReReadImagesDialog({ documentCount, onOpenChange }: ReReadImagesDialogProps) {
  const reRead = useReReadImageDocuments();
  const count = documentCount ?? 0;

  async function confirm() {
    const queued = await reRead.mutateAsync();
    toast.success(`Re-reading ${pluralize(queued.document_count, "document")}`, {
      description:
        (queued.unqueued ?? []).length > 0
          ? "Some sources are busy. Re-read them later from their menu on the Sources page."
          : "See progress on the Sources page.",
    });
  }

  return (
    <ConfirmDialog
      open={documentCount !== null}
      onOpenChange={onOpenChange}
      title="Re-read documents with images?"
      description={
        <>
          <span className="block">
            This model supports image input. The previous one didn't, so the images in{" "}
            {pluralize(count, "document")} haven't been read yet.
          </span>
          <span className="mt-2 block">Re-reading uses model calls. You can also do it later from a source's menu.</span>
        </>
      }
      confirmLabel={`Re-read ${pluralize(count, "document")}`}
      cancelLabel="Later"
      onConfirm={confirm}
    />
  );
}
