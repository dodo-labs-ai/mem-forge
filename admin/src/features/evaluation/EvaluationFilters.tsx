import { X } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { ASSESSMENT_LABELS, type EvaluationParams } from "./model/evaluationParams";
import { checkName, RESULT_LABELS, sourceTypeLabel } from "./model/presentation";

/** The Select value that stands for "no filter". */
const ALL = "all";

interface EvaluationFiltersProps {
  params: EvaluationParams;
  sourceTypes: string[];
  criteria: string[];
  typeLabels: Readonly<Record<string, string>>;
  /** Name of the source the page is scoped to, when it is. */
  scopedSourceName: string | null;
  onChange: (change: Partial<EvaluationParams>) => void;
}

interface FilterSelectProps<T extends string> {
  label: string;
  value: T | null;
  allLabel: string;
  options: { value: T; label: string }[];
  className: string;
  onChange: (value: T | null) => void;
}

function FilterSelect<T extends string>({ label, value, allLabel, options, className, onChange }: FilterSelectProps<T>) {
  const labels = new Map<string, string>(options.map((option) => [option.value, option.label]));
  return (
    <Select
      value={value ?? ALL}
      onValueChange={(next) => {
        if (next === null) return;
        onChange(options.find((option) => option.value === next)?.value ?? null);
      }}
    >
      <SelectTrigger aria-label={label} className={className}>
        <SelectValue>{(current: string) => labels.get(current) ?? allLabel}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>{allLabel}</SelectItem>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function EvaluationFilters({ params, sourceTypes, criteria, typeLabels, scopedSourceName, onChange }: EvaluationFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-2.5 border-b px-4 py-3">
      <FilterSelect
        label="Filter by source type"
        value={params.sourceType}
        allLabel="All source types"
        options={sourceTypes.map((type) => ({ value: type, label: sourceTypeLabel(type, typeLabels) }))}
        className="w-48"
        onChange={(sourceType) => onChange({ sourceType, sourceId: null })}
      />
      <FilterSelect
        label="Filter by criterion"
        value={params.criterion}
        allLabel="All criteria"
        options={criteria.map((criterion) => ({ value: criterion, label: checkName(criterion) }))}
        className="w-56"
        onChange={(criterion) => onChange({ criterion })}
      />
      <FilterSelect
        label="Filter by status"
        value={params.label}
        allLabel="All statuses"
        options={ASSESSMENT_LABELS.map((label) => ({ value: label, label: RESULT_LABELS[label].text }))}
        className="w-44"
        onChange={(label) => onChange({ label })}
      />
      {params.sourceId ? (
        <span className="inline-flex h-8 items-center gap-1.5 rounded-lg border bg-secondary pr-1 pl-2.5 text-sm">
          <span className="text-muted-foreground">Source</span>
          <span className="font-medium text-foreground">{scopedSourceName ?? params.sourceId}</span>
          <button
            type="button"
            aria-label="Show all sources"
            onClick={() => onChange({ sourceId: null })}
            className="grid size-6 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <X className="size-3.5" />
          </button>
        </span>
      ) : null}
    </div>
  );
}
