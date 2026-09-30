import { createContext, useContext } from "react";
import type { Project } from "./model/types";

interface ProjectTableContextValue {
  /** Sources per project key; `null` while the sources are loading or could not load. */
  sourceCounts: ReadonlyMap<string, number> | null;
  /** Whether the caller may rename and delete projects, as the project list reports it. */
  canManage: boolean;
  onEdit: (project: Project) => void;
  onDelete: (project: Project) => void;
}

/** What the table cells need from the page. Columns stay module-level so rows never remount. */
export const ProjectTableContext = createContext<ProjectTableContextValue | null>(null);

export function useProjectTable(): ProjectTableContextValue {
  const value = useContext(ProjectTableContext);
  if (value === null) throw new Error("Project table cells must render inside ProjectTableContext.");
  return value;
}
