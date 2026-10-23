import { ArrowLeft, GitMerge, Plus, Search, Trash2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { HTTP_STATUS, isApiErrorStatus } from "@/api";
import { errorMessage } from "@/lib/errors";
import { formatCount, formatDate } from "@/lib/format";
import { ENTITIES_PATH } from "@/lib/paths";
import { BackLink, EmptyState, ErrorNotice, StatCard } from "@/patterns";
import { Button, buttonVariants } from "@/ui/button";
import { Input } from "@/ui/input";
import { Skeleton } from "@/ui/skeleton";
import { useAddEntityAlias, useEntity, useRemoveEntityAlias } from "./api";
import { MergeEntityDialog } from "./MergeEntityDialog";
import { aliasOrigin, canonicalNameHint, entityName, isCurationForbidden } from "./model/entityPresentation";
import type { EntityAlias, EntityDetail } from "./model/types";

function BackToEntities() {
  return <BackLink label="Entities" render={<Link to={ENTITIES_PATH} />} />;
}

/** The route's entity id, or null when it is not one, so no request is made for it. */
function parseEntityId(raw: string): number | null {
  const id = Number(raw);
  return raw !== "" && Number.isSafeInteger(id) && id > 0 ? id : null;
}

interface AliasesProps {
  entity: EntityDetail;
  /** Receives a failed alias change; null hides the add and remove controls. */
  curation: { onError: (error: unknown) => void } | null;
}

interface AliasRowProps {
  alias: EntityAlias;
  onRemove: ((alias: string) => void) | null;
  /** True while this alias's removal is in flight, so a second click cannot send it again. */
  removing: boolean;
}

function AliasRow({ alias, onRemove, removing }: AliasRowProps) {
  return (
    <li className="flex items-center gap-3 px-4 py-2.5">
      <span className="min-w-0 flex-1 truncate text-sm text-foreground">{alias.alias}</span>
      <span className="shrink-0 text-xs text-muted-foreground">{aliasOrigin(alias)}</span>
      <span className="w-28 shrink-0 text-right text-xs text-muted-foreground">
        {alias.created_at ? formatDate(alias.created_at) : "—"}
      </span>
      {onRemove ? (
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label={`Remove alias ${alias.alias}`}
          disabled={removing}
          onClick={() => onRemove(alias.alias)}
        >
          <Trash2 />
        </Button>
      ) : null}
    </li>
  );
}

function Aliases({ entity, curation }: AliasesProps) {
  const [draft, setDraft] = useState("");
  const addAlias = useAddEntityAlias(entity.id);
  const removeAlias = useRemoveEntityAlias(entity.id);
  const alias = draft.trim();

  function add(event: FormEvent) {
    event.preventDefault();
    if (!alias || !curation) return;
    addAlias.mutate(alias, { onSuccess: () => setDraft(""), onError: curation.onError });
  }

  const remove = curation
    ? (name: string) => removeAlias.mutate(name, { onError: curation.onError })
    : null;

  return (
    <section aria-labelledby="entity-aliases-title" className="overflow-hidden rounded-lg border bg-surface">
      <div className="space-y-0.5 border-b px-4 py-3">
        <h2 id="entity-aliases-title" className="text-sm font-semibold text-foreground">
          Aliases
        </h2>
        <p className="text-xs text-muted-foreground">
          Other names that resolve to this entity when memories are extracted and searched, in every project.
        </p>
      </div>
      {curation ? (
        <form onSubmit={add} className="flex gap-2 border-b px-4 py-3">
          <Input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Add an alias"
            aria-label="New alias"
            className="flex-1"
          />
          <Button type="submit" variant="outline" disabled={!alias || addAlias.isPending}>
            <Plus />
            Add
          </Button>
        </form>
      ) : null}
      {entity.aliases.length === 0 ? (
        <p className="px-4 py-3 text-sm text-muted-foreground">No aliases yet.</p>
      ) : (
        <ul aria-label="Aliases" className="divide-y">
          {entity.aliases.map((item) => (
            <AliasRow
              key={item.alias_normalized}
              alias={item}
              onRemove={remove}
              removing={removeAlias.isPending && removeAlias.variables === item.alias}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

interface EntityDetailViewProps {
  entity: EntityDetail;
  /** Reads the entity again, such as after a refusal that may mean the caller's role changed. */
  onRefetch: () => void;
}

function EntityDetailView({ entity, onRefetch }: EntityDetailViewProps) {
  const [merging, setMerging] = useState(false);
  const [changeError, setChangeError] = useState<string | null>(null);
  const canonical = canonicalNameHint(entity);

  function onChangeError(error: unknown) {
    setChangeError(errorMessage(error));
    // A refusal for curation means the role changed elsewhere; the detail then hides the controls.
    if (isCurationForbidden(error)) onRefetch();
  }

  return (
    <div className="space-y-6">
      <BackToEntities />
      <header className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1 space-y-1.5">
          <h1 className="truncate text-xl font-semibold tracking-tight text-foreground">{entityName(entity)}</h1>
          {canonical ? (
            <p className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Canonical name</span>
              <span className="font-mono text-subtle-foreground">{canonical}</span>
            </p>
          ) : null}
        </div>
        {entity.can_curate ? (
          <Button variant="outline" onClick={() => setMerging(true)}>
            <GitMerge />
            Merge into…
          </Button>
        ) : null}
      </header>

      {changeError ? <ErrorNotice title="Could not change the entity" message={changeError} /> : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <StatCard
          label="Linked memories"
          value={formatCount(entity.linked_memory_count)}
          detail="Memories you can see that name this entity"
        />
        <StatCard
          label="First seen"
          value={entity.created_at ? formatDate(entity.created_at) : "—"}
          detail="When MemForge first recorded it"
        />
      </div>

      <Aliases entity={entity} curation={entity.can_curate ? { onError: onChangeError } : null} />

      {entity.can_curate ? (
        <MergeEntityDialog entity={entity} open={merging} onOpenChange={setMerging} onForbidden={onChangeError} />
      ) : null}
    </div>
  );
}

function EntityNotFound() {
  return (
    <div className="space-y-6">
      <BackToEntities />
      <div className="rounded-lg border bg-surface">
        <EmptyState
          icon={Search}
          title="Entity not found"
          description="It may have been merged into another entity, or no memory you can see names it."
          action={
            <Link to={ENTITIES_PATH} className={buttonVariants({ variant: "outline" })}>
              <ArrowLeft />
              Back to Entities
            </Link>
          }
        />
      </div>
    </div>
  );
}

export function EntityDetailPage() {
  const { entityId = "" } = useParams<{ entityId: string }>();
  const id = parseEntityId(entityId);
  const entityQuery = useEntity(id);

  if (id === null || isApiErrorStatus(entityQuery.error, HTTP_STATUS.notFound)) return <EntityNotFound />;
  if (entityQuery.isPending) {
    return (
      <div aria-busy="true" aria-label="Loading entity" className="space-y-6">
        <BackToEntities />
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }
  if (entityQuery.isError) {
    return (
      <div className="space-y-6">
        <BackToEntities />
        <ErrorNotice
          title="Could not load the entity"
          message={errorMessage(entityQuery.error)}
          onRetry={() => void entityQuery.refetch()}
        />
      </div>
    );
  }
  // Keyed by id: a merge opens the target on this route, which starts with no dialog or error open.
  return <EntityDetailView key={entityQuery.data.id} entity={entityQuery.data} onRefetch={() => void entityQuery.refetch()} />;
}
