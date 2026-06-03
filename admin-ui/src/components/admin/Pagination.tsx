import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getPageMeta } from "@/lib/pagination";

export function Pagination({
  page,
  pageSize,
  total,
  onPageChange,
  itemLabel = "rows",
}: {
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  itemLabel?: string;
}) {
  const { totalPages, pageStart, pageEnd, hasPrev, hasNext } = getPageMeta(page, pageSize, total);

  if (totalPages <= 1) return null;

  return (
    <div className="flex flex-col gap-3 border-t p-4 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-sm text-muted-foreground">
        Showing {pageStart.toLocaleString()}-{pageEnd.toLocaleString()} of {total.toLocaleString()}{" "}
        {itemLabel}
      </p>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">
          Page {(page + 1).toLocaleString()} of {totalPages.toLocaleString()}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={!hasPrev}
          onClick={() => onPageChange(page - 1)}
        >
          <ChevronLeft className="size-4" />
          Prev
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!hasNext}
          onClick={() => onPageChange(page + 1)}
        >
          Next
          <ChevronRight className="size-4" />
        </Button>
      </div>
    </div>
  );
}
