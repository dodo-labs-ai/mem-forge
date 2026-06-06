/**
 * Shared form-state shape and reserved-key validation for the project
 * create form. Lives in its own module so the React component file can
 * stay component-only (react-refresh).
 */
import { isReservedProjectKey } from "@/api/projectKeys";

export interface ProjectCreateFormState {
  name: string;
  key: string;
  shared: boolean;
}

export const emptyProjectCreateForm: ProjectCreateFormState = {
  name: "",
  key: "",
  shared: false,
};

export function projectCreateKeyConflictsWithBuiltIn(
  form: ProjectCreateFormState,
): boolean {
  const trimmed = form.key.trim().toUpperCase();
  return trimmed.length > 0 && isReservedProjectKey(trimmed);
}
