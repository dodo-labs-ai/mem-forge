/**
 * Pure helpers for the source `project_binding` editor.
 *
 * The Source dialog drives a binding through three valid shapes:
 *   - unbound (null): falls into the Unmapped backlog
 *   - fixed: every extracted memory lands in `project_key`
 *   - by_field: the resolver looks up `field` per document; unmapped
 *     values fall through to `default`
 *
 * Validation is intentionally minimal so an admin can save partial state
 * mid-edit only when it is unambiguous. `map` is allowed to be empty.
 */
import type { ProjectBinding } from "@/api/types";

export function projectBindingIsComplete(binding: ProjectBinding | null): boolean {
  if (!binding) {
    return true;
  }
  if (binding.mode === "fixed") {
    return Boolean(binding.project_key && binding.project_key.trim().length > 0);
  }
  if (binding.mode === "by_field") {
    const hasField = Boolean(binding.field && binding.field.trim().length > 0);
    const hasDefault = Boolean(binding.default && binding.default.trim().length > 0);
    return hasField && hasDefault;
  }
  return false;
}
