import { createContext, useContext } from "react";
import type { Project } from "./model/types";

interface MemoryTableContextValue {
  projects: readonly Project[];
  typeLabels: Readonly<Record<string, string>>;
}

/** What the memory and relation cells need from the page. Columns stay module-level so rows never remount. */
export const MemoryTableContext = createContext<MemoryTableContextValue | null>(null);

export function useMemoryTable(): MemoryTableContextValue {
  const value = useContext(MemoryTableContext);
  if (value === null) throw new Error("Memory table cells must render inside MemoryTableContext.");
  return value;
}
