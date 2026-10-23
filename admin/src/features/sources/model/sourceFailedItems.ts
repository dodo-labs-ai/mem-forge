import { formatCount, pluralize } from "@/lib/format";
import type { SourceSyncActivity } from "./sourceSyncActivity";
import type { Source } from "./types";

/** One item of a source that could not be synced, with the step that failed and why. */
export interface SourceFailedItem {
  documentId: string;
  title: string;
  /** The step that failed, as a person reads it; null when the server does not say. */
  stage: string | null;
  reason: string;
  /** An earlier version failed this item, and the queued or running sync retries it once. */
  retryingAfterUpgrade: boolean;
}

export interface SourceFailedItems {
  /** Every failed item of the source; `items` lists the most recent of them. */
  count: number;
  items: SourceFailedItem[];
  /**
   * Items an earlier version failed that a queued or running sync retries once
   * after the upgrade; zero while no sync is on its way.
   */
  retryingAfterUpgrade: number;
}

const STAGE_LABELS: Record<string, string> = {
  fetch: "Reading",
  normalize: "Preparing",
  projection: "Splitting into sections",
  derivation: "Creating memories",
  lifecycle: "Saving memories",
  lifecycle_commit: "Saving memories",
  removal: "Removing",
};

/** How a person reads the step at which an item failed. */
export function failedItemStageLabel(stage: string): string {
  const known = STAGE_LABELS[stage];
  if (known) return known;
  const words = stage.replace(/_/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Processing";
}

export function failedItemsTitle(count: number): string {
  return `Some items failed (${formatCount(count)})`;
}

export function retryingAfterUpgradeTitle(count: number): string {
  return `Retrying ${pluralize(count, "item")} after upgrade`;
}

function syncOnItsWay(activity: SourceSyncActivity | undefined): boolean {
  return activity !== undefined && ["queued", "active", "recovering"].includes(activity.state);
}

/**
 * The source's failed items. The server's record of failed items is the
 * authority: it keeps an item until a sync commits it or the item leaves the
 * source. A source without that record shows the items its last sync could
 * not read. Null when nothing failed.
 */
export function sourceFailedItems(source: Source, activity: SourceSyncActivity | undefined): SourceFailedItems | null {
  const recorded = source.failed_items;
  if (recorded != null) return recordedFailedItems(recorded, activity);
  const lastRunItems = activity?.error?.items ?? [];
  if (lastRunItems.length === 0) return null;
  return {
    count: lastRunItems.length,
    retryingAfterUpgrade: 0,
    items: lastRunItems.map((item) => ({
      documentId: item.doc_id,
      title: item.title || item.doc_id,
      stage: null,
      reason: item.error,
      retryingAfterUpgrade: false,
    })),
  };
}

type RecordedFailedItems = NonNullable<Source["failed_items"]>;

/** Failed items as the server's failure record lists them; null when it lists none. */
export function recordedFailedItems(
  recorded: RecordedFailedItems,
  activity: SourceSyncActivity | undefined,
): SourceFailedItems | null {
  if (recorded.count === 0) return null;
  const retrying = syncOnItsWay(activity) ? recorded.retrying_after_upgrade : 0;
  return {
    count: recorded.count,
    retryingAfterUpgrade: retrying,
    items: recorded.items.map((item) => ({
      documentId: item.document_id,
      title: item.title || item.document_id,
      stage: failedItemStageLabel(item.stage),
      reason: item.message,
      retryingAfterUpgrade: retrying > 0 && item.retrying_after_upgrade,
    })),
  };
}
