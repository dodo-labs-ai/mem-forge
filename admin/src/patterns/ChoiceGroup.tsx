import { cn } from "@/lib/cn";
import { RadioGroup, RadioGroupItem } from "@/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";

export interface Choice {
  value: string;
  label: string;
  /** A line under the label of a radio card. */
  description?: string;
}

/** The most choices drawn as radio cards; more become a select. */
export const MAX_RADIO_CHOICES = 4;

interface ChoiceGroupProps {
  /** Prefix of the ids the cards give their label and description. */
  id: string;
  choices: readonly Choice[];
  value: string;
  onValueChange: (value: string) => void;
  /** The id of the visible label that names the whole group. */
  "aria-labelledby": string;
  "aria-describedby"?: string;
  "aria-invalid"?: boolean;
  disabled?: boolean;
  className?: string;
}

/**
 * One choice out of a set that configures something. Up to four choices are
 * radio cards, each with its label and an optional description, so the
 * person sees every option at once; more choices become a select.
 */
export function ChoiceGroup({
  id,
  choices,
  value,
  onValueChange,
  "aria-labelledby": labelledBy,
  "aria-describedby": describedBy,
  "aria-invalid": invalid,
  disabled,
  className,
}: ChoiceGroupProps) {
  if (choices.length > MAX_RADIO_CHOICES) {
    const labels = new Map(choices.map((choice) => [choice.value, choice.label]));
    return (
      <Select
        value={value || null}
        disabled={disabled}
        onValueChange={(next) => {
          if (typeof next === "string") onValueChange(next);
        }}
      >
        <SelectTrigger
          id={id}
          aria-labelledby={labelledBy}
          aria-describedby={describedBy}
          aria-invalid={invalid || undefined}
          className={cn("w-full", className)}
        >
          <SelectValue>{(current: string | null) => (current ? (labels.get(current) ?? current) : "Choose…")}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {choices.map((choice) => (
            <SelectItem key={choice.value} value={choice.value}>
              {choice.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  return (
    <RadioGroup
      value={value}
      disabled={disabled}
      onValueChange={(next) => {
        if (typeof next === "string") onValueChange(next);
      }}
      aria-labelledby={labelledBy}
      aria-describedby={describedBy}
      aria-invalid={invalid || undefined}
      className={cn("grid-cols-1 sm:grid-cols-2", className)}
    >
      {choices.map((choice) => {
        const labelId = `${id}-${choice.value}-label`;
        const descriptionId = choice.description ? `${id}-${choice.value}-description` : undefined;
        return (
          <label
            key={choice.value}
            className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-input px-3 py-2.5 transition-colors hover:bg-muted/50 has-data-checked:border-primary has-data-checked:bg-muted/40 has-disabled:cursor-not-allowed has-disabled:opacity-50"
          >
            <RadioGroupItem value={choice.value} aria-labelledby={labelId} aria-describedby={descriptionId} className="mt-0.5" />
            <span className="min-w-0 space-y-0.5">
              <span id={labelId} className="block text-sm font-medium text-foreground">
                {choice.label}
              </span>
              {choice.description ? (
                <span id={descriptionId} className="block text-xs text-muted-foreground">
                  {choice.description}
                </span>
              ) : null}
            </span>
          </label>
        );
      })}
    </RadioGroup>
  );
}
