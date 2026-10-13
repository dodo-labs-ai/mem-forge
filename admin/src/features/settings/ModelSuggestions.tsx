import { Button } from "@/ui/button";
import { modelSuggestionList } from "./model/modelSuggestions";
import type { ModelKind } from "./model/types";

interface ModelSuggestionsProps {
  kind: ModelKind;
  /** Every model the endpoint listed. */
  models: string[];
  /** What the model field holds; it narrows the suggestions and marks the chosen one. */
  typed: string;
  onSelect: (model: string) => void;
}

export function ModelSuggestions({ kind, models, typed, onSelect }: ModelSuggestionsProps) {
  if (models.length === 0) return null;
  const { shown, summary } = modelSuggestionList(kind, models, typed);
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">{summary}</p>
      <div role="group" aria-label="Suggested models" className="flex flex-wrap gap-1.5">
        {shown.map((model) => {
          const isSelected = model === typed.trim();
          return (
            <Button
              key={model}
              type="button"
              size="xs"
              variant={isSelected ? "secondary" : "outline"}
              aria-pressed={isSelected}
              onClick={() => onSelect(model)}
              className="h-auto max-w-full py-1 font-mono break-all whitespace-normal"
            >
              {model}
            </Button>
          );
        })}
      </div>
    </div>
  );
}
