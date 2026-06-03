import { FileText, GitBranch, ListChecks, Ruler, type LucideIcon } from "lucide-react";

import type { Memory } from "@/api/types";
import { cn } from "@/lib/utils";
import { MEMORY_TYPE_ICON_COLOR, MEMORY_TYPE_LABEL, type MemoryType } from "@/views/memories/memoryTypeMeta";

// fact -> a recorded statement, decision -> a chosen branch, convention -> a
// standard/rule, procedure -> ordered steps. Keyed by the API's memory_type
// union via `satisfies`, so adding a backend type fails the build here instead
// of silently falling back to the neutral icon.
const TYPE_ICONS = {
  fact: FileText,
  decision: GitBranch,
  convention: Ruler,
  procedure: ListChecks,
} satisfies Record<Memory["memory_type"], LucideIcon>;

type MemoryTypeIconProps = {
  /** Memory type, e.g. "fact". Unknown values fall back to a neutral file icon. */
  type: string;
  /** Sizing classes for the glyph (e.g. "size-4"). */
  className?: string;
};

/**
 * Renders a memory's type as a colored line icon. The type's full name is
 * exposed as the accessible label so screen readers announce the kind.
 */
export function MemoryTypeIcon({ type, className }: MemoryTypeIconProps) {
  const key = type as MemoryType;
  const Icon = TYPE_ICONS[key] ?? FileText;
  const color = MEMORY_TYPE_ICON_COLOR[key] ?? "text-muted-foreground";
  const label = MEMORY_TYPE_LABEL[key] ?? type;

  return <Icon role="img" aria-label={label} className={cn("shrink-0", color, className)} />;
}
