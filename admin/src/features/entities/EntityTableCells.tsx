import { Link } from "react-router-dom";
import { formatDate } from "@/lib/format";
import { entityPath } from "@/lib/paths";
import { canonicalNameHint, entityName } from "./model/entityPresentation";
import type { Entity } from "./model/types";

const NO_VALUE = "—";

export function EntityNameCell({ entity }: { entity: Entity }) {
  const canonical = canonicalNameHint(entity);
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <Link
        to={entityPath(entity.id)}
        onClick={(event) => event.stopPropagation()}
        className="truncate font-medium text-foreground hover:underline"
      >
        {entityName(entity)}
      </Link>
      {canonical ? <span className="truncate font-mono text-xs text-muted-foreground">{canonical}</span> : null}
    </div>
  );
}

export function EntityFirstSeenCell({ entity }: { entity: Entity }) {
  return (
    <div className="text-right text-sm text-muted-foreground">
      {entity.created_at ? formatDate(entity.created_at) : NO_VALUE}
    </div>
  );
}
