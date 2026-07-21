import { useId } from "react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface FilterOption {
  value: string;
  label: string;
}

interface MemoryFiltersPopoverProps {
  type: string;
  status: string;
  source: string;
  project: string;
  projectLabel: string;
  narrowProject: boolean;
  typeOptions: FilterOption[];
  statusOptions: FilterOption[];
  sourceOptions: FilterOption[];
  projectOptions: FilterOption[];
  onTypeChange: (value: string) => void;
  onStatusChange: (value: string) => void;
  onSourceChange: (value: string) => void;
  onProjectChange: (value: string) => void;
  onNarrowProjectChange: (value: boolean) => void;
  onClear: () => void;
}

function FilterField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: FilterOption[];
  onChange: (value: string) => void;
}) {
  const labelId = useId();
  const selectedLabel = options.find((option) => option.value === value)?.label ?? value;

  return (
    <div className="space-y-1.5">
      <span id={labelId} className="text-xs font-medium text-muted-foreground">
        {label}
      </span>
      <Select<string>
        value={value}
        onValueChange={(nextValue) => nextValue && onChange(nextValue)}
      >
        <SelectTrigger size="sm" aria-labelledby={labelId}>
          <SelectValue>{selectedLabel}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function MemoryFiltersPopover({
  type,
  status,
  source,
  project,
  projectLabel,
  narrowProject,
  typeOptions,
  statusOptions,
  sourceOptions,
  projectOptions,
  onTypeChange,
  onStatusChange,
  onSourceChange,
  onProjectChange,
  onNarrowProjectChange,
  onClear,
}: MemoryFiltersPopoverProps) {
  const activeFilterCount = [type, status, source, project].filter(
    (value) => value !== "all",
  ).length;
  const hasProjectFilter = project !== "all";

  return (
    <PopoverPrimitive.Root>
      <PopoverPrimitive.Trigger
        render={
          <Button
            type="button"
            variant={activeFilterCount > 0 ? "secondary" : "outline"}
            size="sm"
            aria-label={
              activeFilterCount > 0
                ? `Filters, ${activeFilterCount} active`
                : "Filters"
            }
          />
        }
      >
        <SlidersHorizontal className="size-3.5" />
        Filters
        {activeFilterCount > 0 && (
          <span className="ml-0.5 rounded-full bg-foreground px-1.5 text-[10px] leading-4 text-background">
            {activeFilterCount}
          </span>
        )}
      </PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Positioner sideOffset={6} align="end" className="z-50">
          <PopoverPrimitive.Popup
            className={cn(
              "w-[min(20rem,calc(100vw-2rem))] rounded-lg border bg-popover p-3 text-sm text-popover-foreground shadow-md outline-none",
              "data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",
            )}
          >
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="font-medium">Filter memories</div>
                <div className="text-xs text-muted-foreground">
                  Narrow the current result set.
                </div>
              </div>
              {activeFilterCount > 0 && (
                <Button type="button" variant="ghost" size="xs" onClick={onClear}>
                  Clear all
                </Button>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2">
              <FilterField
                label="Type"
                value={type}
                options={typeOptions}
                onChange={onTypeChange}
              />
              <FilterField
                label="Status"
                value={status}
                options={statusOptions}
                onChange={onStatusChange}
              />
              <FilterField
                label="Source"
                value={source}
                options={sourceOptions}
                onChange={onSourceChange}
              />
              <FilterField
                label="Project"
                value={project}
                options={projectOptions}
                onChange={onProjectChange}
              />
            </div>

            {hasProjectFilter && (
              <div className="mt-3 border-t pt-3">
                <div className="mb-1.5 text-xs font-medium text-muted-foreground">
                  Project scope
                </div>
                <div
                  role="group"
                  aria-label="Project scope"
                  className="grid grid-cols-2 rounded-md border bg-background p-0.5 text-xs"
                >
                  <button
                    type="button"
                    onClick={() => onNarrowProjectChange(false)}
                    aria-pressed={!narrowProject}
                    className={cn(
                      "truncate rounded px-2 py-1.5 transition-colors",
                      !narrowProject
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                    title={`${projectLabel} on top`}
                  >
                    {projectLabel} on top
                  </button>
                  <button
                    type="button"
                    onClick={() => onNarrowProjectChange(true)}
                    aria-pressed={narrowProject}
                    className={cn(
                      "rounded px-2 py-1.5 transition-colors",
                      narrowProject
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    Only this project
                  </button>
                </div>
              </div>
            )}
          </PopoverPrimitive.Popup>
        </PopoverPrimitive.Positioner>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
