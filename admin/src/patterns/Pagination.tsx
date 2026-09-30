import { formatCount, pluralize } from "@/lib/format";
import { Button } from "@/ui/button";

interface PaginationProps {
  /** One-based page number. */
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  /** What the list counts, such as "memory". */
  itemName: string;
  itemNamePlural?: string;
}

const FIRST_PAGE = 1;

/** "Showing 51–100 of 3,344 memories" with previous and next page buttons. */
export function Pagination({ page, pageSize, total, onPageChange, itemName, itemNamePlural }: PaginationProps) {
  if (total === 0) return null;
  const pageCount = Math.max(FIRST_PAGE, Math.ceil(total / pageSize));
  const first = (page - FIRST_PAGE) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return (
    <nav aria-label="Pagination" className="flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
      <p>
        Showing {formatCount(first)}–{formatCount(last)} of {pluralize(total, itemName, itemNamePlural)}
      </p>
      <div className="flex items-center gap-2">
        <span>
          Page {formatCount(page)} of {formatCount(pageCount)}
        </span>
        <Button size="sm" variant="outline" disabled={page <= FIRST_PAGE} onClick={() => onPageChange(page - 1)}>
          Previous
        </Button>
        <Button size="sm" variant="outline" disabled={page >= pageCount} onClick={() => onPageChange(page + 1)}>
          Next
        </Button>
      </div>
    </nav>
  );
}
