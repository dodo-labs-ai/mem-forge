import { Button } from "@/ui/button";
import { modelSuggestionList } from "./model/modelSuggestions";
import type { ModelKind, ModelOption } from "./model/types";

interface ModelSuggestionsProps {
  kind: ModelKind;
  /** The models Load models found for the kind. */
  models: readonly ModelOption[];
  /** Whether the endpoint listed only models of the kind. */
  kindFiltered: boolean;
  /** What the model field holds; it narrows the suggestions and marks the chosen one. */
  typed: string;
  onSelect: (model: ModelOption) => void;
}

export function ModelSuggestions({ kind, models, kindFiltered, typed, onSelect }: ModelSuggestionsProps) {
  if (models.length === 0) return null;
  const { shown, summary } = modelSuggestionList(kind, models, typed, kindFiltered);
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">{summary}</p>
      {shown.length > 0 ? (
        <div role="group" aria-label="Suggested models" className="flex flex-wrap gap-1.5">
          {shown.map((model) => {
            const isSelected = model.id === typed.trim();
            return (
              <Button
                key={model.id}
                type="button"
                size="xs"
                variant={isSelected ? "secondary" : "outline"}
                aria-pressed={isSelected}
                onClick={() => onSelect(model)}
                className="h-auto max-w-full py-1 font-mono break-all whitespace-normal"
              >
                {model.id}
              </Button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
