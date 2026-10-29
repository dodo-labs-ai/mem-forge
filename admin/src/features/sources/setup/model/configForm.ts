import { z } from "zod";
import type { components } from "@/api/schema.gen";

/**
 * The source type form, read the way the server reads it.
 *
 * `GET /genes/{type}/config-schema` describes a type's form: groups, fields,
 * labelled choices and `visible_when` conditions. This module mirrors the
 * server's shared form check (`memforge.source_types.form`), so the setup
 * pages show, hide and require exactly what save checks:
 *
 * - **Comparison string.** A condition and a choice compare values as
 *   strings: `null`, `""` and an empty list are blank, a boolean is `"true"`
 *   or `"false"`, a number is its decimal string, a string is trimmed and
 *   lower-cased, and a list, or a comma string of a list field, is its
 *   non-blank items joined by `","`.
 * - **Effective value.** A field reads the value the person or the stored
 *   configuration gave it; if that is blank, the value config-check derived;
 *   if that is still blank, the declared `default`. The server reads a
 *   descriptor's scope default before the declared default, and a mode
 *   field's declared default always equals it, so the form needs only the
 *   declared one.
 * - **Required while visible.** A required field is checked only while its
 *   conditions hold, and a hidden field is otherwise ignored.
 *
 * Example: Jira declares `auth_mode` with the choices `pat`, visible only
 * when `sync_mode` is `cloud`, and `browser_cookie`, and the default `pat`.
 * Choosing local sync hides that choice, and `reconcileHiddenChoices` moves
 * the form off `pat`, chosen or default, to `browser_cookie`, which also
 * hides the `pat` field.
 */

export type ConfigSchema = components["schemas"]["GeneConfigSchemaResponse"];
export type ConfigField = components["schemas"]["ConfigFieldResponse"];
export type ConfigGroup = components["schemas"]["ConfigGroupResponse"];
export type ConfigOption = components["schemas"]["ConfigOptionResponse"];
export type ConfigCondition = components["schemas"]["ConfigConditionResponse"];

/** What one control holds: text for text, secret, integer and select fields, a flag, or a list. */
export type FieldValue = string | boolean | string[];

/** The controls' values by field key. A field nobody set, neither the person nor the stored configuration, is absent. */
export type ConfigValues = Partial<Record<string, FieldValue>>;

/** A configuration as the server sends it: a stored, redacted configuration, or config-check's `derived` values. */
export type ServerConfig = Readonly<Record<string, unknown>>;

export interface ConfigFormState {
  values: ConfigValues;
  /** config-check's `derived` values, read where nobody set the field. */
  derived?: ServerConfig;
  /** The edited Source's stored, redacted configuration; `<key>_configured` says a secret is kept. */
  stored?: ServerConfig;
}

const LIST_FIELD_TYPES = new Set(["tag_list", "multi_select"]);

function isListField(field: ConfigField): boolean {
  return LIST_FIELD_TYPES.has(field.field_type);
}

/** The string a condition or choice compares, `""` for a blank value. */
export function comparisonString(value: unknown, fieldType?: string): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => comparisonString(item))
      .filter(Boolean)
      .join(",");
  }
  const text = String(value);
  if (fieldType && LIST_FIELD_TYPES.has(fieldType)) return comparisonString(text.split(","));
  return text.trim().toLowerCase();
}

function isBlank(value: unknown, field: ConfigField): boolean {
  return comparisonString(value, field.field_type) === "";
}

function listItems(value: unknown): string[] {
  const items = Array.isArray(value) ? value.map(String) : typeof value === "string" ? value.split(",") : [];
  return items.map((item) => item.trim()).filter(Boolean);
}

/** A server value as the field's control holds it. */
export function controlValue(field: ConfigField, value: unknown): FieldValue {
  switch (field.field_type) {
    case "boolean":
      return value === true || comparisonString(value) === "true";
    case "tag_list":
    case "multi_select":
      return listItems(value);
    case "secret":
      return "";
    case "select": {
      const text = comparisonString(value);
      return field.choices.find((choice) => comparisonString(choice.value) === text)?.value ?? (value == null ? "" : String(value));
    }
    default:
      return value == null ? "" : String(value);
  }
}

/**
 * The values a form starts from: what the stored configuration gives each
 * declared field. A new Source starts empty, so every field shows its
 * declared default and is sent only once the person sets it. A secret never
 * starts with a value, because the stored configuration is redacted.
 */
export function defaultValues(schema: ConfigSchema, stored: ServerConfig = {}): ConfigValues {
  const values: ConfigValues = {};
  for (const field of schema.fields) {
    if (field.field_type === "secret" || isBlank(stored[field.key], field)) continue;
    values[field.key] = controlValue(field, stored[field.key]);
  }
  return values;
}

/** The value a field reads: what the person or the stored configuration set, else the derived value, else the declared default. */
export function effectiveValue(field: ConfigField, state: ConfigFormState): unknown {
  const value = state.values[field.key];
  if (value !== undefined && !isBlank(value, field)) return value;
  const derived = state.derived?.[field.key];
  if (!isBlank(derived, field)) return derived;
  return field.default;
}

/**
 * What the field's control shows: the value someone set, even a cleared one,
 * else the derived value, else the declared default.
 */
export function displayValue(field: ConfigField, state: ConfigFormState): FieldValue {
  const value = state.values[field.key];
  if (value !== undefined) return value;
  const derived = state.derived?.[field.key];
  return controlValue(field, isBlank(derived, field) ? field.default : derived);
}

function fieldsByKey(schema: ConfigSchema): Map<string, ConfigField> {
  return new Map(schema.fields.map((field) => [field.key, field]));
}

function conditionsHold(conditions: readonly ConfigCondition[], fields: Map<string, ConfigField>, state: ConfigFormState): boolean {
  return conditions.every((condition) => {
    const field = fields.get(condition.field);
    if (!field) return false;
    const value = comparisonString(effectiveValue(field, state), field.field_type);
    return condition.values.some((allowed) => comparisonString(allowed) === value);
  });
}

/** Whether every condition holds for the form's values. */
export function conditionsHoldFor(schema: ConfigSchema, conditions: readonly ConfigCondition[], state: ConfigFormState): boolean {
  return conditionsHold(conditions, fieldsByKey(schema), state);
}

/** The declared fields whose conditions hold, in declaration order. */
export function visibleFields(schema: ConfigSchema, state: ConfigFormState): ConfigField[] {
  const fields = fieldsByKey(schema);
  return schema.fields.filter((field) => conditionsHold(field.visible_when, fields, state));
}

/** The choices of a field whose conditions hold. */
export function visibleChoices(schema: ConfigSchema, field: ConfigField, state: ConfigFormState): ConfigOption[] {
  const fields = fieldsByKey(schema);
  return field.choices.filter((choice) => conditionsHold(choice.visible_when, fields, state));
}

/**
 * Values with every visible choice field moved off a choice that is no
 * longer offered: a select resets to its first visible choice, and a
 * multi-select drops the hidden items. A reset can hide further choices, so
 * it repeats until nothing changes.
 */
export function reconcileHiddenChoices(schema: ConfigSchema, state: ConfigFormState): ConfigValues {
  const values: ConfigValues = { ...state.values };
  for (let pass = 0; pass <= schema.fields.length; pass += 1) {
    const current = { ...state, values };
    let changed = false;
    for (const field of visibleFields(schema, current)) {
      if (field.choices.length === 0 || (field.field_type !== "select" && field.field_type !== "multi_select")) continue;
      const offered = visibleChoices(schema, field, current);
      const allowed = new Set(offered.map((choice) => comparisonString(choice.value)));
      if (field.field_type === "select") {
        const chosen = comparisonString(effectiveValue(field, current));
        if (chosen && !allowed.has(chosen) && offered[0]) {
          values[field.key] = offered[0].value;
          changed = true;
        }
      } else {
        const items = listItems(effectiveValue(field, current));
        const kept = items.filter((item) => allowed.has(comparisonString(item)));
        if (kept.length !== items.length) {
          values[field.key] = kept;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return values;
}

function secretConfigured(field: ConfigField, state: ConfigFormState): boolean {
  return state.stored?.[`${field.key}_configured`] === true;
}

function hasValue(field: ConfigField, state: ConfigFormState): boolean {
  if (field.field_type === "secret") {
    return comparisonString(state.values[field.key]) !== "" || secretConfigured(field, state);
  }
  return !isBlank(effectiveValue(field, state), field);
}

/** The visible required fields without a value. A declared default counts as a value, and so does a stored secret. */
export function missingRequired(schema: ConfigSchema, state: ConfigFormState): ConfigField[] {
  return visibleFields(schema, state).filter((field) => field.required && !hasValue(field, state));
}

const WHOLE_NUMBER = /^-?\d+$/;

/**
 * The configuration a save sends: every declared field someone set, hidden
 * ones included, so switching a mode back keeps what was typed. A field
 * nobody set is sent absent, so the server's inference or declared default
 * decides it. A blank secret is sent absent, which keeps the stored one, and
 * a blank integer is sent as `null`.
 */
export function serializeConfig(schema: ConfigSchema, values: ConfigValues): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  for (const field of schema.fields) {
    const value = values[field.key];
    if (value === undefined) continue;
    if (field.field_type === "secret") {
      if (typeof value === "string" && value !== "") config[field.key] = value;
    } else if (field.field_type === "integer") {
      const text = String(value).trim();
      config[field.key] = text === "" ? null : WHOLE_NUMBER.test(text) ? Number(text) : text;
    } else if (field.field_type === "boolean") {
      config[field.key] = value === true;
    } else if (isListField(field)) {
      config[field.key] = listItems(value);
    } else {
      config[field.key] = value;
    }
  }
  return config;
}

/**
 * A Zod schema for the controls' values that refuses what save would refuse
 * before it reaches the server: a visible required field without a value,
 * and a visible integer field that is not a whole number. Rebuild it when
 * the derived values or the stored configuration change.
 */
export function formSchema(schema: ConfigSchema, context: Omit<ConfigFormState, "values"> = {}) {
  const valueSchema = z.union([z.string(), z.boolean(), z.array(z.string())]).optional();
  return z.record(z.string(), valueSchema).superRefine((values, issues) => {
    const state: ConfigFormState = { ...context, values };
    const missing = new Set(missingRequired(schema, state).map((field) => field.key));
    for (const field of visibleFields(schema, state)) {
      if (missing.has(field.key)) {
        issues.addIssue({ code: "custom", path: [field.key], message: `${field.label} is required.` });
      } else if (field.field_type === "integer") {
        const text = String(values[field.key] ?? "").trim();
        if (text !== "" && !WHOLE_NUMBER.test(text)) {
          issues.addIssue({ code: "custom", path: [field.key], message: `${field.label} must be a whole number.` });
        }
      }
    }
  });
}
