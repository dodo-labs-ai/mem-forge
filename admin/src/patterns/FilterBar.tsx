import { Search } from "lucide-react";
import type { ReactNode } from "react";
import { Input } from "@/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/ui/toggle-group";

export interface FilterOption<T extends string> {
  value: T;
  label: string;
  count?: number;
}

interface FilterBarProps<T extends string> {
  search: string;
  onSearchChange: (value: string) => void;
  searchPlaceholder: string;
  filters?: FilterOption<T>[];
  filter?: T;
  onFilterChange?: (value: T) => void;
  /** Controls placed at the end of the bar, such as sorting. */
  trailing?: ReactNode;
}

export function FilterBar<T extends string>({
  search,
  onSearchChange,
  searchPlaceholder,
  filters,
  filter,
  onFilterChange,
  trailing,
}: FilterBarProps<T>) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="relative w-full max-w-xs">
        <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder={searchPlaceholder}
          aria-label={searchPlaceholder}
          className="pl-8"
        />
      </div>
      {filters && filter !== undefined && onFilterChange ? (
        <ToggleGroup
          value={[filter]}
          onValueChange={(values: string[]) => {
            const next = filters.find((option) => option.value === values[0]);
            if (next) onFilterChange(next.value);
          }}
          variant="outline"
          size="sm"
          aria-label="Filter"
        >
          {filters.map((option) => (
            <ToggleGroupItem key={option.value} value={option.value}>
              {option.label}
              {option.count !== undefined ? (
                <span className="tabular-nums text-muted-foreground">{option.count}</span>
              ) : null}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      ) : null}
      {trailing ? <div className="ml-auto flex items-center gap-2">{trailing}</div> : null}
    </div>
  );
}
