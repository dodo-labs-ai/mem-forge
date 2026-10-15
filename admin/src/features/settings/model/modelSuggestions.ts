import { pluralize } from "@/lib/format";
import type { ModelKind, ModelOption } from "./types";

/** The most suggestions shown at once; a provider can list hundreds of models. */
export const MAX_SHOWN_SUGGESTIONS = 24;

/**
 * The models whose id contains what is typed, the one it names exactly first.
 * Nothing typed keeps every model.
 */
function matchingTyped(models: readonly ModelOption[], typed: string): ModelOption[] {
  const text = typed.trim().toLowerCase();
  if (text === "") return [...models];
  const matching = models.filter((model) => model.id.toLowerCase().includes(text));
  const exact = matching.filter((model) => model.id.toLowerCase() === text);
  return [...exact, ...matching.filter((model) => model.id.toLowerCase() !== text)];
}

export interface ModelSuggestionList {
  /** The models offered as buttons. */
  shown: ModelOption[];
  /** The sentence above the buttons. */
  summary: string;
}

/**
 * The listed models to offer, narrowed by what is typed in the model field and
 * cut to a readable number. `kindFiltered` says whether the endpoint listed
 * only models of the kind, or every model it has.
 */
export function modelSuggestionList(
  kind: ModelKind,
  models: readonly ModelOption[],
  typed: string,
  kindFiltered: boolean,
): ModelSuggestionList {
  const noun = kindFiltered ? `${kind} model` : "model";
  const matching = matchingTyped(models, typed);
  const shown = matching.slice(0, MAX_SHOWN_SUGGESTIONS);
  const text = typed.trim();
  const cut = shown.length < matching.length ? ` Showing the first ${shown.length}.` : "";
  if (text === "") {
    return {
      shown,
      summary: `${pluralize(models.length, `listed ${noun}`)}.${cut} Type in the Model field to filter them, or pick one.`,
    };
  }
  if (matching.length === 0) {
    return { shown, summary: `No listed ${noun} matches “${text}”. The model id is used as typed.` };
  }
  const all = matching.length < models.length ? " Clear the Model field to see them all." : "";
  return {
    shown,
    summary: `${matching.length} of ${pluralize(models.length, `listed ${noun}`)} ${matching.length === 1 ? "matches" : "match"} “${text}”.${cut}${all}`,
  };
}
