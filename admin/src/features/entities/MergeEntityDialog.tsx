import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { errorMessage } from "@/lib/errors";
import { pluralize } from "@/lib/format";
import { PropertyList, SearchBox } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Skeleton } from "@/ui/skeleton";
import { useEntity, useEntityList, useMergeEntity } from "./api";
import { canonicalNameHint, entityName, isCurationForbidden } from "./model/entityPresentation";
import type { Entity, EntityDetail } from "./model/types";

/** How many matching names the target search lists; a longer list asks for a narrower search. */
const CANDIDATE_LIMIT = 8;

interface MergeEntityDialogProps {
  /** The entity that is merged away; its aliases and memory links move to the target. */
  entity: EntityDetail;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Receives a refusal because the caller is no longer a workspace admin; the dialog closes. */
  onForbidden: (error: Error) => void;
}

function linkedMemories(count: number): string {
  return `${pluralize(count, "linked memory", "linked memories")} now`;
}

function TargetCandidates({ entity, onChoose }: { entity: EntityDetail; onChoose: (target: Entity) => void }) {
  const [search, setSearch] = useState("");
  const list = useEntityList({ search, limit: CANDIDATE_LIMIT + 1, offset: 0 });
  const candidates = (list.data?.data ?? []).filter((candidate) => candidate.id !== entity.id).slice(0, CANDIDATE_LIMIT);
  return (
    <div className="space-y-3">
      <SearchBox label="Search for the target entity" value={search} onCommit={setSearch} className="max-w-none" />
      {list.isError ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          {errorMessage(list.error)}
        </p>
      ) : list.isPending ? (
        <Skeleton className="h-24 w-full" />
      ) : candidates.length === 0 ? (
        <p className="text-sm text-muted-foreground">No other entity matches.</p>
      ) : (
        <ul aria-label="Matching entities" className="divide-y rounded-lg border">
          {candidates.map((candidate) => {
            const canonical = canonicalNameHint(candidate);
            return (
              <li key={candidate.id}>
                <button
                  type="button"
                  onClick={() => onChoose(candidate)}
                  className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-muted/50"
                >
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-sm font-medium text-foreground">{entityName(candidate)}</span>
                    {canonical ? <span className="truncate font-mono text-xs text-muted-foreground">{canonical}</span> : null}
                  </span>
                  <ChevronRight aria-hidden className="size-4 shrink-0 text-muted-foreground" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

interface MergeSummaryProps {
  entity: EntityDetail;
  target: Entity;
}

/**
 * Each entity's linked memory count as it is now, read from its detail. The
 * counts are not added up: a memory linked to both counts once after the
 * merge, so the target can end with fewer than their sum.
 */
function MergeSummary({ entity, target }: MergeSummaryProps) {
  const targetDetail = useEntity(target.id);
  const targetCount = targetDetail.isError ? (
    <span className="text-tone-danger-foreground">Could not read it. {errorMessage(targetDetail.error)}</span>
  ) : targetDetail.isPending ? (
    <Skeleton className="h-4 w-32" />
  ) : (
    linkedMemories(targetDetail.data.linked_memory_count)
  );
  return (
    <div className="space-y-3">
      <PropertyList
        items={[
          { label: `Merged away: ${entityName(entity)}`, value: linkedMemories(entity.linked_memory_count) },
          { label: `Kept: ${entityName(target)}`, value: targetCount },
        ]}
      />
      <p className="text-sm text-muted-foreground">
        These are each entity’s current links, not a total after the merge: a memory linked to both counts once
        afterwards. {entityName(entity)}’s {pluralize(entity.aliases.length, "alias", "aliases")} and memory links move
        to {entityName(target)}, and {entityName(entity)} is removed. This cannot be undone.
      </p>
    </div>
  );
}

/**
 * Merges this entity into another the admin picks by name. The second step
 * shows both entities' current linked memory counts before the merge, which
 * has no dry run.
 */
export function MergeEntityDialog({ entity, open, onOpenChange, onForbidden }: MergeEntityDialogProps) {
  const [target, setTarget] = useState<Entity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const merge = useMergeEntity(entity.id);

  function close(nextOpen: boolean) {
    if (merge.isPending) return;
    if (!nextOpen) {
      setTarget(null);
      setError(null);
    }
    onOpenChange(nextOpen);
  }

  async function confirm() {
    if (!target) return;
    setError(null);
    try {
      await merge.mutateAsync(target.id);
      close(false);
    } catch (caught) {
      if (isCurationForbidden(caught)) {
        close(false);
        onForbidden(caught);
      } else {
        setError(errorMessage(caught));
      }
    }
  }

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{target ? `Merge into ${entityName(target)}?` : `Merge ${entityName(entity)}`}</DialogTitle>
          <DialogDescription>
            {target
              ? `${entityName(entity)} becomes part of ${entityName(target)}.`
              : `Choose the entity that keeps ${entityName(entity)}’s aliases and memory links.`}
          </DialogDescription>
        </DialogHeader>
        {target ? <MergeSummary entity={entity} target={target} /> : <TargetCandidates entity={entity} onChoose={setTarget} />}
        {error ? (
          <p role="alert" className="text-sm text-tone-danger-foreground">
            {error}
          </p>
        ) : null}
        <DialogFooter>
          {target ? (
            <>
              <Button variant="outline" onClick={() => setTarget(null)} disabled={merge.isPending}>
                Back
              </Button>
              <Button variant="destructive" onClick={() => void confirm()} disabled={merge.isPending}>
                Merge
              </Button>
            </>
          ) : (
            <Button variant="outline" onClick={() => close(false)}>
              Cancel
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
