import { CircleAlert, Lock, RefreshCw, Users } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";
import { DrawerSection, Notice, StatusBadge } from "@/patterns";
import { Button } from "@/ui/button";
import { useSettleSourceAccess } from "./api";
import {
  accessChangeRefusal,
  accessChoiceLabel,
  accessTransitionProgress,
  presentSourceAccess,
} from "./model/sourceAccess";
import type { Source } from "./model/types";

/** Who can use a source, or the state of its access change, on its row and in its drawer. */
export function SourceAccessLabel({ source }: { source: Source }) {
  const view = presentSourceAccess(source);
  if (view.kind === "changing") return <StatusBadge tone="live">{view.label}</StatusBadge>;
  if (view.kind === "failed") return <StatusBadge tone="danger">{view.label}</StatusBadge>;
  const Icon = view.personal ? Lock : Users;
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-subtle-foreground">
      <Icon aria-hidden className="size-3.5 text-muted-foreground" />
      {view.label}
    </span>
  );
}

/**
 * The access change a source is applying: how many of its memories are
 * updated, or, when it failed, why, with Retry and Revert. Only the owner of
 * a source can see it while its access changes, so whoever sees a failed
 * change can settle it.
 */
export function SourceAccessTransitionSection({ source }: { source: Source }) {
  const settle = useSettleSourceAccess();
  const transition = source.access_state === "changing" ? source.access_transition : null;
  if (!transition) return null;
  const target = accessChoiceLabel(transition.target_policy);

  if (transition.status !== "failed") {
    const progress = accessTransitionProgress(transition);
    return (
      <DrawerSection title="Access change">
        <Notice tone="live" icon={RefreshCw} title={`Changing access to ${target}`}>
          Syncing is paused while the existing memories are updated. Until then, only you can find the source or query
          its memories.
        </Notice>
        <div className="space-y-1.5">
          <div
            role="progressbar"
            aria-label="Access change"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress.percent ?? undefined}
            aria-valuetext={progress.text}
            className="h-1.5 overflow-hidden rounded-full bg-muted"
          >
            <div
              className={cn("h-full rounded-full bg-tone-live transition-[width]", progress.percent === null && "w-1/4 animate-pulse")}
              style={progress.percent === null ? undefined : { width: `${progress.percent}%` }}
            />
          </div>
          <p className="text-sm text-foreground">{progress.text}</p>
        </div>
      </DrawerSection>
    );
  }

  const previous = accessChoiceLabel(transition.previous_policy);

  function run(kind: "retry" | "revert") {
    settle.mutate(
      { source, settle: kind },
      {
        onSuccess: () =>
          toast.success(kind === "retry" ? `Retrying the access change of ${source.name}` : `Returning ${source.name} to ${previous}`),
        onError: (error) =>
          toast.error(kind === "retry" ? "Could not retry the access change" : "Could not revert the access change", {
            description: accessChangeRefusal(errorMessage(error)),
          }),
      },
    );
  }

  return (
    <DrawerSection title="Access change">
      <Notice tone="danger" icon={CircleAlert} title={`The change to ${target} failed`}>
        <p>{transition.error_message || "Some existing memories were not updated."}</p>
        <p>
          Until it is retried or reverted, syncing stays paused and only you can use the source.
        </p>
      </Notice>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" disabled={settle.isPending} onClick={() => run("retry")}>
          Retry
        </Button>
        <Button size="sm" variant="outline" disabled={settle.isPending} onClick={() => run("revert")}>
          Revert to {previous}
        </Button>
      </div>
    </DrawerSection>
  );
}
