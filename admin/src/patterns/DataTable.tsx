import { ChevronDown, ChevronRight } from "lucide-react";
import { useState, type ReactNode } from "react";
import {
  createColumnHelper,
  tableFeatures,
  useTable,
  type Cell,
  type ColumnDef,
  type Row,
  type RowData,
} from "@tanstack/react-table";
import { cn } from "@/lib/cn";
import { Skeleton } from "@/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/ui/table";

declare module "@tanstack/react-table" {
  // Declaration merging must repeat the library's type parameters.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TFeatures, TData, TValue> {
    /** Classes for the header and body cells of this column, such as a width. */
    className?: string;
  }
}

/** The table features every DataTable uses. Sorting and filtering happen before rows reach the table. */
export const dataTableFeatures = tableFeatures({});
export type DataTableFeatures = typeof dataTableFeatures;
export type DataColumn<T extends RowData> = ColumnDef<DataTableFeatures, T, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** Column helper bound to the DataTable features. */
export function dataColumns<T extends RowData>() {
  return createColumnHelper<DataTableFeatures, T>();
}

export interface DataTableGroup<T> {
  id: string;
  /** Content of the full-width group header row. */
  header: ReactNode;
  rows: T[];
}

interface DataTableProps<T extends RowData> {
  /**
   * Must keep its identity between renders (define it at module level): cell
   * functions render as components, so new functions remount every row.
   */
  columns: DataColumn<T>[];
  /** Flat rows, or rows split into groups with a header row each. */
  data: T[] | DataTableGroup<T>[];
  getRowId: (row: T) => string;
  onRowClick?: (row: T) => void;
  /** Row id to show as selected, such as the item open in a drawer. */
  selectedRowId?: string | null;
  rowClassName?: (row: T) => string | undefined;
  loading?: boolean;
  empty?: ReactNode;
  "aria-label": string;
}

const LOADING_ROW_COUNT = 4;
const NO_GROUP_ID = "__rows";

function isGrouped<T>(data: T[] | DataTableGroup<T>[]): data is DataTableGroup<T>[] {
  const first = data[0];
  return first !== undefined && typeof first === "object" && first !== null && "rows" in first && "header" in first;
}

export function DataTable<T extends RowData>({
  columns,
  data,
  getRowId,
  onRowClick,
  selectedRowId,
  rowClassName,
  loading = false,
  empty,
  "aria-label": ariaLabel,
}: DataTableProps<T>) {
  const groups: DataTableGroup<T>[] = isGrouped(data) ? data : [{ id: NO_GROUP_ID, header: null, rows: data as T[] }];
  const rows = groups.flatMap((group) => group.rows);
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set());
  const table = useTable({
    features: dataTableFeatures,
    columns,
    data: rows,
    getRowId: (row: T) => getRowId(row),
  });
  const rowsById = new Map(table.getRowModel().rows.map((row) => [row.id, row]));
  const columnCount = table.getAllLeafColumns().length;

  function toggle(groupId: string) {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }

  function renderRow(row: Row<DataTableFeatures, T>, groupId: string) {
    const original = row.original;
    const clickable = onRowClick !== undefined;
    return (
      <TableRow
        key={`${groupId}:${row.id}`}
        data-state={selectedRowId === row.id ? "selected" : undefined}
        className={cn(clickable && "cursor-pointer", rowClassName?.(original))}
        onClick={clickable ? () => onRowClick(original) : undefined}
      >
        {row.getAllCells().map((cell: Cell<DataTableFeatures, T, unknown>) => (
          <TableCell key={cell.id} className={cn("py-2.5 align-middle", cell.column.columnDef.meta?.className)}>
            <table.FlexRender cell={cell} />
          </TableCell>
        ))}
      </TableRow>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border bg-surface">
      <Table aria-label={ariaLabel}>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="hover:bg-transparent">
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className={cn("h-9 text-xs font-medium text-muted-foreground", header.column.columnDef.meta?.className)}
                >
                  {header.isPlaceholder ? null : <table.FlexRender header={header} />}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {loading
            ? Array.from({ length: LOADING_ROW_COUNT }, (_, index) => (
                <TableRow key={`loading-${index}`} className="hover:bg-transparent">
                  <TableCell colSpan={columnCount}>
                    <Skeleton className="h-6 w-full" />
                  </TableCell>
                </TableRow>
              ))
            : null}
          {!loading && rows.length === 0 ? (
            <TableRow className="hover:bg-transparent">
              <TableCell colSpan={columnCount}>{empty}</TableCell>
            </TableRow>
          ) : null}
          {!loading
            ? groups.map((group) => {
                const isCollapsed = collapsed.has(group.id);
                return [
                  group.header !== null ? (
                    <TableRow key={`group:${group.id}`} className="bg-surface-subtle hover:bg-surface-subtle">
                      <TableCell colSpan={columnCount} className="py-2">
                        <button
                          type="button"
                          aria-expanded={!isCollapsed}
                          onClick={() => toggle(group.id)}
                          className="flex w-full items-center gap-2 text-left"
                        >
                          {isCollapsed ? (
                            <ChevronRight aria-hidden className="size-4 shrink-0 text-muted-foreground" />
                          ) : (
                            <ChevronDown aria-hidden className="size-4 shrink-0 text-muted-foreground" />
                          )}
                          <div className="min-w-0 flex-1">{group.header}</div>
                        </button>
                      </TableCell>
                    </TableRow>
                  ) : null,
                  ...(isCollapsed
                    ? []
                    : group.rows.map((item) => {
                        const row = rowsById.get(getRowId(item));
                        return row ? renderRow(row, group.id) : null;
                      })),
                ];
              })
            : null}
        </TableBody>
      </Table>
    </div>
  );
}
