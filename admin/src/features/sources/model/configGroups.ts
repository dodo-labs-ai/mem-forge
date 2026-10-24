import type { components } from "@/api";

type ConfigGroup = components["schemas"]["ConfigGroupResponse"];

/** Declared items by their `order`, keeping declaration order between equal ones. */
export function inDeclaredOrder<T extends { order: number }>(items: readonly T[]): T[] {
  return items
    .map((item, index) => ({ item, index }))
    .sort((a, b) => a.item.order - b.item.order || a.index - b.index)
    .map(({ item }) => item);
}

/**
 * The key of a source type's first group with this purpose, such as the
 * step "Configure" opens for `connection`, or "Configure file scope" for
 * `content`. Null when the type declares none.
 */
export function firstGroupWithPurpose(
  schema: { groups: readonly ConfigGroup[] },
  purpose: ConfigGroup["purpose"],
): string | null {
  return inDeclaredOrder(schema.groups).find((group) => group.purpose === purpose)?.key ?? null;
}
