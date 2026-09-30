import { Archive, ArrowRight, Link2, MoreHorizontal, Pencil } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { reviewPath } from "@/lib/paths";
import { Button, buttonVariants } from "@/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/ui/dropdown-menu";
import { memoryStatus } from "./model/memoryPresentation";
import type { MemoryDetail } from "./model/types";
import { ProposeCorrectionDialog } from "./ProposeCorrectionDialog";
import { RetireMemoryDialog } from "./RetireMemoryDialog";

/** Why Retire is not offered for this memory, or null when it is. */
function retireBlockedReason(memory: MemoryDetail): string | null {
  if (memoryStatus(memory.status) !== "active") return "Only active memories can be retired.";
  if (memory.source_backed) return "This memory is backed by a source. Remove it from the source, or propose a correction.";
  return null;
}

async function copyLink() {
  try {
    await navigator.clipboard.writeText(window.location.href);
    toast.success("Link copied");
  } catch {
    toast.error("Could not copy the link");
  }
}

interface MemoryActionsProps {
  memory: MemoryDetail;
  /** The open review this memory waits for. */
  reviewId: string | undefined;
}

/** At most one primary action for the memory's state, and a menu of the rest. */
export function MemoryActions({ memory, reviewId }: MemoryActionsProps) {
  const [dialog, setDialog] = useState<"correct" | "retire" | null>(null);
  const status = memoryStatus(memory.status);
  const retireBlocked = retireBlockedReason(memory);

  return (
    <div className="flex items-center gap-2">
      {status === "active" ? (
        <Button variant="outline" onClick={() => setDialog("correct")}>
          <Pencil />
          Propose correction
        </Button>
      ) : null}
      {status === "pending_review" && reviewId ? (
        <Link to={reviewPath(reviewId)} className={buttonVariants()}>
          <ArrowRight />
          Open review
        </Link>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button size="icon" variant="outline" aria-label="More actions" />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          <DropdownMenuItem onClick={() => void copyLink()}>
            <Link2 />
            Copy link
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            disabled={retireBlocked !== null}
            onClick={() => setDialog("retire")}
            className="items-start"
          >
            <Archive className="mt-0.5" />
            <span className="flex flex-col gap-0.5">
              <span>Retire memory</span>
              {retireBlocked ? <span className="text-xs text-muted-foreground">{retireBlocked}</span> : null}
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ProposeCorrectionDialog
        key={`correct:${memory.content_hash}`}
        memory={memory}
        open={dialog === "correct"}
        onOpenChange={(open) => setDialog(open ? "correct" : null)}
      />
      <RetireMemoryDialog
        key={`retire:${memory.content_hash}`}
        memory={memory}
        open={dialog === "retire"}
        onOpenChange={(open) => setDialog(open ? "retire" : null)}
      />
    </div>
  );
}
