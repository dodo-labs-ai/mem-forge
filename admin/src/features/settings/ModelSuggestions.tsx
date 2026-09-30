import { Button } from "@/ui/button";
import { suggestedModels, suggestionSummary } from "./model/modelSuggestions";
import type { EndpointKind } from "./model/types";

interface ModelSuggestionsProps {
  kind: EndpointKind;
  /** Every model the connection test returned. */
  models: string[];
  selected: string;
  onSelect: (model: string) => void;
}

export function ModelSuggestions({ kind, models, selected, onSelect }: ModelSuggestionsProps) {
  const suggestions = suggestedModels(kind, models);
  if (suggestions.length === 0) return null;
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">{suggestionSummary(kind, suggestions.length, models.length)}</p>
      <div role="group" aria-label="Suggested models" className="flex flex-wrap gap-1.5">
        {suggestions.map((model) => {
          const isSelected = model === selected.trim();
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
