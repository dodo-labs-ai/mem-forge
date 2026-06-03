// Visual metadata for memory types, shown as an icon on the Memory list and
// detail views. Kept as pure data (no React/lucide imports) so it can be unit
// tested directly and reused wherever a memory type needs a consistent color.

export const MEMORY_TYPES = ["fact", "decision", "convention", "procedure"] as const;

export type MemoryType = (typeof MEMORY_TYPES)[number];

// Icon tint per type, using the same hue family as the type badges in
// StatusBadge.tsx (fact = blue, decision = violet, convention = emerald,
// procedure = orange). The -600 shade is one step lighter than the badge's
// -700 text, tuned for a standalone glyph on a light background.
export const MEMORY_TYPE_ICON_COLOR: Record<MemoryType, string> = {
  fact: "text-blue-600",
  decision: "text-violet-600",
  convention: "text-emerald-600",
  procedure: "text-orange-600",
};

export const MEMORY_TYPE_LABEL: Record<MemoryType, string> = {
  fact: "Fact",
  decision: "Decision",
  convention: "Convention",
  procedure: "Procedure",
};
