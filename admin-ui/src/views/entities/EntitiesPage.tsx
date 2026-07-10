import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ChevronRight, Database, RefreshCw } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { Entity, PaginatedResponse } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { FilterSelect } from "@/components/admin/FilterSelect";
import { PageHeader } from "@/components/admin/PageHeader";
import { Pagination } from "@/components/admin/Pagination";
import { SearchInput } from "@/components/admin/SearchInput";
import { Toolbar } from "@/components/admin/Toolbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ENTITY_PAGE_SIZE } from "@/lib/constants";

export function EntitiesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const search = searchParams.get("search") ?? "";
  const tag = searchParams.get("tag") ?? "all";
  const [page, setPage] = useState(0);
  const navigate = useNavigate();

  const updateFilter = (key: "search" | "tag", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (!value || value === "all") {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    setSearchParams(next, { replace: true });
    setPage(0);
  };

  const entitiesQuery = useQuery<PaginatedResponse<Entity>>({
    queryKey: ["entities", search, tag, page],
    queryFn: () =>
      resourceClient
        .get("/entities", {
          params: {
            search: search || undefined,
            tag: tag !== "all" ? tag : undefined,
            limit: ENTITY_PAGE_SIZE,
            offset: page * ENTITY_PAGE_SIZE,
          },
        })
        .then((response) => response.data),
  });

  const entities = entitiesQuery.data?.data ?? [];
  const total = entitiesQuery.data?.total ?? 0;
  const allTags = Array.from(
    new Set(entities.flatMap((entity) => entity.tags)),
  ).sort();

  return (
    <div className="space-y-4">
      <PageHeader
        title="Entities"
        description="Canonical names, aliases, and linked memory context."
        actions={
          <Button type="button" variant="outline" onClick={() => entitiesQuery.refetch()}>
            <RefreshCw className="size-4" />
            Refresh
          </Button>
        }
      />

      <DataSurface>
        <div className="flex flex-col gap-3 border-b p-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h2 className="text-base font-semibold">Entity List</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {total.toLocaleString()} names in the current result set.
            </p>
          </div>
          <Toolbar className="lg:justify-end">
            <SearchInput
              value={search}
              onChange={(value) => updateFilter("search", value)}
              placeholder="Filter entities..."
            />
            <FilterSelect
              value={tag}
              onChange={(value) => updateFilter("tag", value)}
              label="Filter by tag"
              options={[
                { value: "all", label: "All tags" },
                ...allTags.map((value) => ({ value, label: value })),
              ]}
            />
          </Toolbar>
        </div>
        <AsyncBoundary
          isLoading={entitiesQuery.isLoading}
          isError={entitiesQuery.isError}
          error={entitiesQuery.error}
          onRetry={() => entitiesQuery.refetch()}
          isEmpty={entities.length === 0}
          empty={
            <EmptyState
              icon={Database}
              title="No entities found"
              description="Sync a source to start tracking entities."
            />
          }
        >
          <ul className="divide-y divide-border">
            {entities.map((entity) => {
              const display = entity.display_name || entity.canonical_name;
              const showCanonical =
                entity.display_name &&
                entity.display_name !== entity.canonical_name;
              return (
                <li key={entity.id}>
                  <button
                    type="button"
                    onClick={() => navigate(`/entities/${entity.id}`)}
                    className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-accent/50"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-foreground">
                          {display}
                        </span>
                        {entity.tags.map((entityTag) => (
                          <Badge
                            key={entityTag}
                            variant="secondary"
                            className="text-[11px]"
                          >
                            {entityTag}
                          </Badge>
                        ))}
                      </div>
                      {showCanonical && (
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          {entity.canonical_name}
                        </p>
                      )}
                    </div>
                    <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                  </button>
                </li>
              );
            })}
          </ul>
        </AsyncBoundary>
        <Pagination
          page={page}
          pageSize={ENTITY_PAGE_SIZE}
          total={total}
          onPageChange={setPage}
          itemLabel="names"
        />
      </DataSurface>
    </div>
  );
}
