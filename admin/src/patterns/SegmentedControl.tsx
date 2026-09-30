import { cn } from "@/lib/cn";
import { ToggleGroup, ToggleGroupItem } from "@/ui/toggle-group";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  /** The name read out when the visible label is an abbreviation, such as "Last 7 days" for "7d". */
  accessibleLabel?: string;
}

type SegmentedControlProps<T extends string> = {
  options: ReadonlyArray<SegmentedOption<T>>;
  value: T;
  onValueChange: (value: T) => void;
  className?: string;
} & ({ "aria-label": string } | { "aria-labelledby": string });

/** One choice out of a few, shown side by side. Choosing one changes what the page shows. */
export function SegmentedControl<T extends string>({
  options,
  value,
  onValueChange,
  className,
  ...labelling
}: SegmentedControlProps<T>) {
  return (
    <ToggleGroup
      {...labelling}
      value={[value]}
      onValueChange={(values: string[]) => {
        const next = options.find((option) => option.value === values[0]);
        if (next) onValueChange(next.value);
      }}
      spacing={0}
      className={cn("gap-0.5 rounded-lg bg-muted p-[3px]", className)}
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          value={option.value}
          aria-label={option.accessibleLabel}
          size="sm"
          className="h-7 rounded-md px-3 font-normal text-subtle-foreground hover:bg-transparent aria-pressed:bg-background aria-pressed:font-medium aria-pressed:text-foreground aria-pressed:shadow-sm"
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
