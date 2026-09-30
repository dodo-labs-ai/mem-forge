import type { Source } from "./types";

/** How a source routes its memories to projects. */
export interface ProjectBinding {
  mode: "fixed" | "by_field";
  project_key?: string;
  field?: string;
  map?: Record<string, string>;
  default?: string;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/**
 * Reads the source's project binding. The server returns it as a free-form
 * object, so only a known mode and string fields are taken; anything else
 * counts as no binding.
 */
export function sourceProjectBinding(source: Pick<Source, "project_binding">): ProjectBinding | null {
  const binding = source.project_binding;
  if (!binding || (binding.mode !== "fixed" && binding.mode !== "by_field")) return null;
  const map = binding.map;
  return {
    mode: binding.mode,
    project_key: text(binding.project_key),
    field: text(binding.field),
    default: text(binding.default),
    map:
      map && typeof map === "object"
        ? Object.fromEntries(Object.entries(map).filter((entry): entry is [string, string] => typeof entry[1] === "string"))
        : undefined,
  };
}
