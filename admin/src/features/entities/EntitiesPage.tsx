import { Tags } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { errorMessage } from "@/lib/errors";
import { pluralize } from "@/lib/format";
import { ENTITIES_SEARCH_PARAM, entityPath } from "@/lib/paths";
import { DataTable, EmptyState, ErrorNotice, PageHeader, Pagination, SearchBox } from "@/patterns";
import { useEntityList } from "./api";
import { ENTITY_COLUMNS } from "./entityColumns";

/** Rows per page of the entity list. */
const ENTITY_PAGE_SIZE = 50;
/** The page parameter of the list, one-based; the search parameter is shared with links from other pages. */
const PAGE_PARAM = "page";
const FIRST_PAGE = 1;

function readPage(params: URLSearchParams): number {
  const page = Number(params.get(PAGE_PARAM));
  return Number.isInteger(page) && page > FIRST_PAGE ? page : FIRST_PAGE;
}

export function EntitiesPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const search = params.get(ENTITIES_SEARCH_PARAM) ?? "";
  const page = readPage(params);
  const list = useEntityList({ search, limit: ENTITY_PAGE_SIZE, offset: (page - FIRST_PAGE) * ENTITY_PAGE_SIZE });
  const total = list.data?.total;

  function update(next: { search: string; page: number }, options?: { replace?: boolean }) {
    const query = new URLSearchParams();
    if (next.search) query.set(ENTITIES_SEARCH_PARAM, next.search);
    if (next.page > FIRST_PAGE) query.set(PAGE_PARAM, String(next.page));
    setParams(query, options);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Entities"
        description="The people, systems and terms your memories name. Aliases let different spellings resolve to one entity."
      />

      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <SearchBox
            label="Search entities"
            value={search}
            onCommit={(next) => update({ search: next, page: FIRST_PAGE }, { replace: true })}
          />
          {total !== undefined ? (
            <span className="ml-auto text-sm text-muted-foreground tabular-nums">{pluralize(total, "entity", "entities")}</span>
          ) : null}
        </div>

        {list.isError ? (
          <ErrorNotice title="Could not load entities" message={errorMessage(list.error)} onRetry={() => void list.refetch()} />
        ) : (
          <DataTable
            aria-label="Entities"
            columns={ENTITY_COLUMNS}
            data={list.data?.data ?? []}
            getRowId={(entity) => String(entity.id)}
            onRowClick={(entity) => navigate(entityPath(entity.id))}
            loading={list.isPending}
            empty={
              <EmptyState
                icon={Tags}
                title={search ? "No entities match" : "No entities yet"}
                description={search ? "Search matches display and canonical names, not aliases." : "Entities appear as sources sync and memories name them."}
              />
            }
          />
        )}

        {total !== undefined ? (
          <Pagination
            page={page}
            pageSize={ENTITY_PAGE_SIZE}
            total={total}
            itemName="entity"
            itemNamePlural="entities"
            onPageChange={(next) => update({ search, page: next })}
          />
        ) : null}
      </div>
    </div>
  );
}
