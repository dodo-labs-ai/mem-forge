import { createContext, useContext } from "react";
import type { Source } from "./model/types";
import type { SourceActions } from "./useSourceActions";

interface SourceTableContextValue {
  typeLabels: Readonly<Record<string, string>>;
  actions: SourceActions;
  onViewDetails: (source: Source) => void;
  onDelete: (source: Source) => void;
}

/** What the table cells need from the page. Columns stay module-level so rows never remount. */
export const SourceTableContext = createContext<SourceTableContextValue | null>(null);

export function useSourceTable(): SourceTableContextValue {
  const value = useContext(SourceTableContext);
  if (value === null) throw new Error("Source table cells must render inside SourceTableContext.");
  return value;
}
