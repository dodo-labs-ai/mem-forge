import { SlidersHorizontal, X } from "lucide-react";
import { useId } from "react";
import { Button } from "@/ui/button";
import { Popover, PopoverContent, PopoverDescription, PopoverHeader, PopoverTitle, PopoverTrigger } from "@/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/ui/toggle-group";
import { activeFilterKeys, type FilterKey, type MemoryFilters, type ProjectScope } from "./model/listState";
import {
  DEFAULT_LIST_STATUS,
  MEMORY_STATUS_LABELS,
  MEMORY_TYPES,
  MEMORY_TYPE_LABELS,
  STATUS_FILTERS,
} from "./model/memoryPresentation";

interface Option {
  value: string;
  label: string;
}

/** The select value that stands for "no filter", which the field names with its `defaultLabel`. */
const NO_FILTER = "none";

const FILTER_NAMES: Record<FilterKey, string> = { type: "Type", status: "Status", source: "Source", project: "Project" };

const TYPE_OPTIONS: Option[] = MEMORY_TYPES.map((type) => ({ value: type, label: MEMORY_TYPE_LABELS[type] }));
const STATUS_OPTIONS: Option[] = STATUS_FILTERS.map((status) => ({ value: status, label: MEMORY_STATUS_LABELS[status] }));

interface FilterFieldProps {
  label: string;
  /** What the list shows without this filter, such as "All types". */
  defaultLabel: string;
  value: string | null;
  options: Option[];
  onChange: (value: string | null) => void;
}

function FilterField({ label, defaultLabel, value, options, onChange }: FilterFieldProps) {
  const labelId = useId();
  const choices = [{ value: NO_FILTER, label: defaultLabel }, ...options];
  return (
    <div className="flex flex-col gap-1.5">
      <span id={labelId} className="text-xs font-medium text-muted-foreground">
        {label}
      </span>
      <Select
        value={value ?? NO_FILTER}
        onValueChange={(next) => {
          if (typeof next === "string") onChange(next === NO_FILTER ? null : next);
        }}
      >
        <SelectTrigger size="sm" aria-labelledby={labelId} className="w-full">
          <SelectValue>{(selected: string) => choices.find((choice) => choice.value === selected)?.label ?? selected}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {choices.map((choice) => (
            <SelectItem key={choice.value} value={choice.value}>
              {choice.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

interface MemoryFiltersPopoverProps {
  filters: MemoryFilters;
  sourceOptions: Option[];
  projectOptions: Option[];
  onChange: (change: Partial<MemoryFilters>) => void;
  onClear: () => void;
}

export function MemoryFiltersPopover({ filters, sourceOptions, projectOptions, onChange, onClear }: MemoryFiltersPopoverProps) {
  const activeCount = activeFilterKeys(filters).length;
  const project = projectOptions.find((option) => option.value === filters.project);
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant={activeCount > 0 ? "secondary" : "outline"}
            aria-label={activeCount > 0 ? `Filters, ${activeCount} active` : "Filters"}
          />
        }
      >
        <SlidersHorizontal />
        Filters
        {activeCount > 0 ? (
          <span className="rounded-full bg-foreground px-1.5 text-xs text-background tabular-nums">{activeCount}</span>
        ) : null}
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 gap-3 p-3">
        <div className="flex items-start justify-between gap-3">
          <PopoverHeader>
            <PopoverTitle>Filter memories</PopoverTitle>
            <PopoverDescription>Narrow the current result set.</PopoverDescription>
          </PopoverHeader>
          {activeCount > 0 ? (
            <Button size="xs" variant="ghost" onClick={onClear}>
              Clear all
            </Button>
          ) : null}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <FilterField label="Type" defaultLabel="All types" value={filters.type} options={TYPE_OPTIONS} onChange={(type) => onChange({ type })} />
          <FilterField
            label="Status"
            defaultLabel={MEMORY_STATUS_LABELS[DEFAULT_LIST_STATUS]}
            value={filters.status}
            options={STATUS_OPTIONS}
            onChange={(status) => onChange({ status })}
          />
          <FilterField
            label="Source"
            defaultLabel="All sources"
            value={filters.source}
            options={sourceOptions}
            onChange={(source) => onChange({ source })}
          />
          <FilterField
            label="Project"
            defaultLabel="All projects"
            value={filters.project}
            options={projectOptions}
            onChange={(projectKey) => onChange({ project: projectKey })}
          />
        </div>
        {filters.project !== null ? (
          <div className="flex flex-col gap-1.5 border-t pt-3">
            <span className="text-xs font-medium text-muted-foreground">Project scope</span>
            <ToggleGroup
              value={[filters.projectScope]}
              onValueChange={(values: string[]) => {
                const scope = values[0] as ProjectScope | undefined;
                if (scope) onChange({ projectScope: scope });
              }}
              variant="outline"
              size="sm"
              aria-label="Project scope"
              className="w-full"
            >
              <ToggleGroupItem value="project-first" className="min-w-0 flex-1">
                <span className="truncate">{project?.label ?? filters.project} on top</span>
              </ToggleGroupItem>
              <ToggleGroupItem value="project" className="flex-1">
                Only this project
              </ToggleGroupItem>
            </ToggleGroup>
            <p className="text-xs text-muted-foreground">
              Applies to search. Only this project also keeps team-wide memories.
            </p>
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

interface ActiveFilterChipsProps {
  filters: MemoryFilters;
  valueLabels: Record<FilterKey, (value: string) => string>;
  onRemove: (key: FilterKey) => void;
  onClear: () => void;
}

/** The filters in force, each removable on its own. */
export function ActiveFilterChips({ filters, valueLabels, onRemove, onClear }: ActiveFilterChipsProps) {
  const keys = activeFilterKeys(filters);
  if (keys.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      {keys.map((key) => (
        <span key={key} className="inline-flex h-7 items-center gap-1 rounded-md border bg-surface pr-1 pl-2.5 text-xs">
          <span className="text-muted-foreground">{FILTER_NAMES[key]}</span>
          <span className="font-medium text-foreground">{valueLabels[key](filters[key]!)}</span>
          <Button size="icon-xs" variant="ghost" aria-label={`Remove ${FILTER_NAMES[key]} filter`} onClick={() => onRemove(key)}>
            <X />
          </Button>
        </span>
      ))}
      <Button size="xs" variant="ghost" onClick={onClear}>
        Clear all
      </Button>
    </div>
  );
}
