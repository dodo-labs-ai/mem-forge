import { CheckCircle2, RefreshCw, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { Navigate, useSearchParams } from "react-router-dom";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";
import { EmptyState, ErrorNotice, PageHeader, Pagination, SegmentedControl } from "@/patterns";
import { Button } from "@/ui/button";
import { Checkbox } from "@/ui/checkbox";
import { Skeleton } from "@/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/tabs";
import { useReviewQueue } from "./api";
import { BatchDecisionDialog } from "./BatchDecisionDialog";
import { decisionOptions } from "./model/reviewDecision";
import {
  REVIEW_KINDS,
  REVIEW_QUEUE_PAGE_SIZE,
  REVIEW_TABS,
  emptyQueueMessage,
  lastQueuePage,
  queueSummary,
  queueViewFromSearch,
  searchForQueueView,
  type ReviewKind,
  type ReviewQueueView,
  type ReviewTab,
} from "./model/reviewQueue";
import type { ReviewDecision, ReviewListItem } from "./model/types";
import { ReviewQueueRow } from "./ReviewQueueRow";

const LOADING_ROW_COUNT = 3;
const EMPTY_REVIEWS: ReviewListItem[] = [];

function QueueSkeleton() {
  return (
    <div className="space-y-3 p-4.5" aria-busy>
      {Array.from({ length: LOADING_ROW_COUNT }, (_, index) => (
        <Skeleton key={index} className="h-16 w-full" />
      ))}
    </div>
  );
}

/** Reviews picked for a bulk decision, shown only under the tab and kind they were picked in. */
interface Selection {
  scope: string;
  reviews: ReadonlyMap<string, ReviewListItem>;
}

const NO_SELECTION: Selection = { scope: "", reviews: new Map() };

function selectionScope(view: ReviewQueueView): string {
  return `${view.tab}:${view.kind}`;
}

function QueueSummaryRow({
  tab,
  total,
  selectAll,
}: {
  tab: ReviewTab;
  total: number;
  selectAll?: { checked: boolean; disabled: boolean; onCheckedChange: (checked: boolean) => void };
}) {
  const summary = queueSummary(tab, total);
  return (
    <div className="flex items-center gap-3 border-b px-4.5 py-3.5">
      {selectAll ? (
        <Checkbox
          checked={selectAll.checked}
          disabled={selectAll.disabled}
          onCheckedChange={(checked) => selectAll.onCheckedChange(checked === true)}
          aria-label="Select every review on this page that you can decide"
        />
      ) : null}
      <span
        aria-hidden
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-lg",
          tab === "waiting" ? "bg-tone-warn-soft text-tone-warn-foreground" : "bg-muted text-muted-foreground",
        )}
      >
        <ShieldCheck className="size-4" />
      </span>
      <div className="min-w-0 space-y-0.5">
        <p className="text-sm font-semibold text-foreground">{summary.title}</p>
        <p className="text-xs text-muted-foreground">{summary.description}</p>
      </div>
    </div>
  );
}

function SelectionBar({
  count,
  labels,
  onDecide,
  onClear,
}: {
  count: number;
  labels: { approve: string; reject: string };
  onDecide: (decision: ReviewDecision) => void;
  onClear: () => void;
}) {
  return (
    <div
      role="region"
      aria-label="Bulk decision"
      className="flex items-center gap-3 border-b bg-primary px-4.5 py-2.5 text-sm text-primary-foreground"
    >
      <span className="font-medium">{count} selected</span>
      <div className="ml-auto flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          className="border-primary-foreground/30 bg-transparent text-primary-foreground hover:bg-primary-foreground/10 hover:text-primary-foreground"
          onClick={() => onDecide("reject")}
        >
          {labels.reject}
        </Button>
        <Button size="sm" variant="secondary" onClick={() => onDecide("approve")}>
          {labels.approve}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="text-primary-foreground/80 hover:bg-primary-foreground/10 hover:text-primary-foreground"
          onClick={onClear}
        >
          Clear
        </Button>
      </div>
    </div>
  );
}

export function ReviewQueuePage() {
  const [search, setSearch] = useSearchParams();
  const view = queueViewFromSearch(search);
  const queue = useReviewQueue(view);
  const reviews = queue.data?.data ?? EMPTY_REVIEWS;
  const total = queue.data?.total ?? 0;

  const scope = selectionScope(view);
  const [selectionState, setSelection] = useState<Selection>(NO_SELECTION);
  const selection = selectionState.scope === scope ? selectionState.reviews : NO_SELECTION.reviews;
  const [pendingDecision, setPendingDecision] = useState<ReviewDecision | null>(null);

  const selectable = view.tab === "waiting";
  // Only reviews the caller can decide go into a bulk decision.
  const decidable = reviews.filter((review) => review.can_decide);
  const pageSelected = decidable.filter((review) => selection.has(review.id)).length;
  const firstSelected = selection.values().next().value;
  const selectedOptions = firstSelected ? decisionOptions(firstSelected.presentation) : null;

  function showView(next: Partial<ReviewQueueView>) {
    setSearch(searchForQueueView({ ...view, page: 1, ...next }));
  }

  function select(changes: ReviewListItem[], selected: boolean) {
    const next = new Map(selection);
    for (const review of changes) {
      if (selected) next.set(review.id, review);
      else next.delete(review.id);
    }
    setSelection({ scope, reviews: next });
  }

  function clearSelection() {
    setSelection(NO_SELECTION);
  }

  const empty = emptyQueueMessage(view.tab);
  // A page can empty while earlier pages still hold reviews, for example after
  // deciding the last reviews of the last page.
  const lastPage = lastQueuePage(total);
  const pastLastPage = reviews.length === 0 && view.page > lastPage;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Review queue"
        description="Decisions MemForge could not make on its own. Open a review to compare the current and proposed memory, then choose what to keep."
        actions={
          <Button variant="outline" onClick={() => void queue.refetch()} disabled={queue.isFetching}>
            <RefreshCw aria-hidden className={cn(queue.isFetching && "animate-spin")} />
            Refresh list
          </Button>
        }
      />

      <Tabs
        value={view.tab}
        onValueChange={(tab: ReviewTab) => showView({ tab })}
        className="gap-0 overflow-hidden rounded-xl border bg-surface"
      >
        <div className="flex flex-wrap items-center gap-3 border-b px-4.5 py-3">
          <TabsList aria-label="Review status">
            {REVIEW_TABS.map((tab) => (
              <TabsTrigger key={tab.value} value={tab.value} className="px-3">
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-sm text-muted-foreground" aria-hidden>
              Show
            </span>
            <SegmentedControl<ReviewKind>
              aria-label="Show"
              options={REVIEW_KINDS}
              value={view.kind}
              onValueChange={(kind) => showView({ kind })}
            />
          </div>
        </div>

        <TabsContent value={view.tab}>
          {queue.isError ? (
            <div className="p-4.5">
              <ErrorNotice
                title="Could not load reviews"
                message={errorMessage(queue.error)}
                onRetry={() => void queue.refetch()}
              />
            </div>
          ) : queue.isPending || (queue.isPlaceholderData && reviews.length === 0) ? (
            <QueueSkeleton />
          ) : pastLastPage ? (
            <Navigate replace to={{ search: searchForQueueView({ ...view, page: lastPage }).toString() }} />
          ) : reviews.length === 0 ? (
            <EmptyState
              icon={view.tab === "waiting" ? CheckCircle2 : undefined}
              title={empty.title}
              description={empty.description}
            />
          ) : (
            <>
              {selection.size > 0 && selectedOptions ? (
                <SelectionBar
                  count={selection.size}
                  labels={{ approve: selectedOptions.useLatest.label, reject: selectedOptions.keepCurrent.label }}
                  onDecide={setPendingDecision}
                  onClear={clearSelection}
                />
              ) : null}
              <QueueSummaryRow
                tab={view.tab}
                total={total}
                selectAll={
                  selectable
                    ? {
                        checked: decidable.length > 0 && pageSelected === decidable.length,
                        disabled: decidable.length === 0,
                        onCheckedChange: (checked) => select(decidable, checked),
                      }
                    : undefined
                }
              />
              <ul aria-label="Reviews">
                {reviews.map((review) => (
                  <ReviewQueueRow
                    key={review.id}
                    review={review}
                    queueSearch={search.toString()}
                    selection={
                      selectable
                        ? { selected: selection.has(review.id), onSelectedChange: (selected) => select([review], selected) }
                        : undefined
                    }
                  />
                ))}
              </ul>
              <div className="border-t px-4.5 py-2.5">
                <Pagination
                  page={view.page}
                  pageSize={REVIEW_QUEUE_PAGE_SIZE}
                  total={total}
                  itemName="review"
                  onPageChange={(page) => showView({ page })}
                />
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>

      {pendingDecision !== null ? (
        <BatchDecisionDialog
          decision={pendingDecision}
          reviews={[...selection.values()]}
          onOpenChange={(open) => {
            if (!open) setPendingDecision(null);
          }}
          onApplied={clearSelection}
        />
      ) : null}
    </div>
  );
}
