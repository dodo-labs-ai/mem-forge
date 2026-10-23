import {
  comparisonString,
  displayValue,
  missingRequired,
  visibleChoices,
  visibleFields,
  type ConfigField,
  type ConfigFormState,
  type ConfigGroup,
  type ConfigSchema,
} from "./configForm";

/**
 * The steps of a source setup page: the product's own steps around the
 * type's groups.
 *
 * | Step | Shown | Writes |
 * |---|---|---|
 * | Basics | Always | `name` |
 * | Each group, by `order` | Always | `config` |
 * | Access | Create only, in a shared workspace | `access_policy` |
 * | Project | Always; the only step of a managed type | `project_binding` |
 * | Schedule | For configurable types | `sync_schedule` |
 *
 * A group step is keyed by the group's key, so `?step=connection` opens the
 * connection group and `?step=project` the Project step.
 */

export type SetupStepKind = "basics" | "group" | "access" | "project" | "schedule";

export interface SetupStep {
  key: string;
  label: string;
  kind: SetupStepKind;
  /** The type's group a `group` step draws. */
  group?: ConfigGroup;
}

export interface SetupStepOptions {
  mode: "create" | "edit";
  /** The workspace is shared, so a new Source chooses who can see it. */
  sharedWorkspace: boolean;
  /** The service creates Sources of this type; only their project is set here. */
  managed?: boolean;
}

function byOrder<T extends { order: number }>(items: readonly T[]): T[] {
  return items
    .map((item, index) => ({ item, index }))
    .sort((a, b) => a.item.order - b.item.order || a.index - b.index)
    .map(({ item }) => item);
}

/** The steps of the setup page, in order. */
export function setupSteps(schema: ConfigSchema, options: SetupStepOptions): SetupStep[] {
  const project: SetupStep = { key: "project", label: "Project", kind: "project" };
  if (options.managed) return [project];
  return [
    { key: "basics", label: "Basics", kind: "basics" },
    ...byOrder(schema.groups).map((group): SetupStep => ({ key: group.key, label: group.label, kind: "group", group })),
    ...(options.mode === "create" && options.sharedWorkspace ? [{ key: "access", label: "Access", kind: "access" as const }] : []),
    project,
    { key: "schedule", label: "Schedule", kind: "schedule" },
  ];
}

/** The declared fields of a group, in the order the step draws them. */
export function groupFields(schema: ConfigSchema, groupKey: string): ConfigField[] {
  return byOrder(schema.fields.filter((field) => field.group === groupKey));
}

/** The key of the first group with this purpose, such as the step "Configure" opens. */
export function firstGroupWithPurpose(schema: ConfigSchema, purpose: ConfigGroup["purpose"]): string | null {
  return byOrder(schema.groups).find((group) => group.purpose === purpose)?.key ?? null;
}

/** The key of the last content group, which "Check setup" follows. */
export function checkSetupGroup(schema: ConfigSchema): string | null {
  return byOrder(schema.groups).findLast((group) => group.purpose === "content")?.key ?? null;
}

/** Whether every visible required field of the group has a value. */
export function groupComplete(schema: ConfigSchema, groupKey: string, state: ConfigFormState): boolean {
  return !missingRequired(schema, state).some((field) => field.group === groupKey);
}

/** How many field issues fall in each group, keyed by group key. An issue for an undeclared key counts nowhere. */
export function issueCountsByGroup(schema: ConfigSchema, issueKeys: Iterable<string | null>): Record<string, number> {
  const groupOf = new Map(schema.fields.map((field) => [field.key, field.group]));
  const counts: Record<string, number> = {};
  for (const key of issueKeys) {
    const group = key === null ? undefined : groupOf.get(key);
    if (group !== undefined) counts[group] = (counts[group] ?? 0) + 1;
  }
  return counts;
}

function summaryPart(schema: ConfigSchema, field: ConfigField, state: ConfigFormState): string | null {
  const value = displayValue(field, state);
  if (Array.isArray(value)) return value.length > 0 ? `${field.label}: ${value.length}` : null;
  if (typeof value === "boolean" || comparisonString(value) === "") return null;
  if (field.field_type === "select") {
    return visibleChoices(schema, field, state).find((choice) => choice.value === value)?.label ?? value;
  }
  if (field.field_type === "url") {
    try {
      return new URL(value).host || value;
    } catch {
      return value;
    }
  }
  return value;
}

/**
 * One line naming what a completed group step set: its visible, non-secret,
 * non-advanced values, a URL as its host, a list as its count and a choice
 * as its label. Flags are left out.
 */
export function groupSummary(schema: ConfigSchema, groupKey: string, state: ConfigFormState): string {
  const visible = new Set(visibleFields(schema, state).map((field) => field.key));
  return groupFields(schema, groupKey)
    .filter((field) => visible.has(field.key) && field.field_type !== "secret" && !field.advanced)
    .map((field) => summaryPart(schema, field, state))
    .filter((part): part is string => part !== null)
    .join(" · ");
}
