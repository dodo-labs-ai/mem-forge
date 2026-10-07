import { RefreshCw } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useProjects } from "@/api";
import { useSourceTypeLabels, useSources } from "@/features/sources";
import { REVIEW_PATH } from "@/lib/paths";
import { PageHeader } from "@/patterns";
import { Button } from "@/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/ui/tabs";
import { useMemoryStats, useOpenReviewCount, useRefreshMemories, useRelationCounts } from "./api";
import { MemoryListView } from "./MemoryListView";
import { MemoryStatCards } from "./MemoryStatCards";
import { MemoryTableContext } from "./memoryTableContext";
import {
  changePageState,
  readPageState,
  relationsPageState,
  writePageState,
  type MemoriesPageState,
  type MemoriesView,
} from "./model/listState";
import { RelationListView } from "./RelationListView";

const CONFLICTS_SEARCH = `?${writePageState(relationsPageState("contradicts"))}`;
const EMPTY_LIST: never[] = [];

export function MemoriesPage() {
  const [params, setParams] = useSearchParams();
  const state = readPageState(params);
  const stats = useMemoryStats();
  const openReviews = useOpenReviewCount();
  const relationCounts = useRelationCounts();
  const projects = useProjects();
  const { sources } = useSources();
  const typeLabels = useSourceTypeLabels();
  const refresh = useRefreshMemories();

  function update(next: MemoriesPageState, options?: { replace?: boolean }) {
    setParams(writePageState(next), options);
  }

  const viewSwitch = (
    <Tabs value={state.view} onValueChange={(view: MemoriesView) => update(changePageState(state, { view }))}>
      <TabsList aria-label="View">
        <TabsTrigger value="memories" className="px-3">
          Memories
        </TabsTrigger>
        <TabsTrigger value="relations" className="px-3">
          Relations
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Memories"
        description="What MemForge has learned from your sources and agent sessions. Open a memory to see its evidence."
        actions={
          <Button variant="outline" onClick={() => void refresh()}>
            <RefreshCw />
            Refresh
          </Button>
        }
      />

      <MemoryStatCards
        stats={stats.data}
        openReviews={openReviews.total}
        relationCounts={relationCounts.counts}
        reviewQueuePath={REVIEW_PATH}
        conflictsPath={CONFLICTS_SEARCH}
      />

      <MemoryTableContext.Provider value={{ projects: projects.data ?? EMPTY_LIST, typeLabels }}>
        {state.view === "relations" ? (
          <RelationListView state={state} onStateChange={update} viewSwitch={viewSwitch} counts={relationCounts.counts} />
        ) : (
          <MemoryListView
            state={state}
            onStateChange={update}
            viewSwitch={viewSwitch}
            projects={projects.data ?? EMPTY_LIST}
            sources={sources}
          />
        )}
      </MemoryTableContext.Provider>
    </div>
  );
}
