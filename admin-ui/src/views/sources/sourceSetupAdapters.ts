import type { ConfigField, GeneConfigSchema } from "../../api/types.js";

import {
  applyConfluenceUrlInference,
  isConfluenceFieldRequired,
  isConfluenceFieldVisible,
} from "./confluenceConfig.js";
import type { SourceConnectionMode } from "./sourceConnectionPresentation.js";

export type ConfigValue = string | number | boolean | string[] | null;
export type ConfigForm = Record<string, ConfigValue>;
export type SchemaSourceType =
  | "confluence"
  | "github_pages"
  | "github_repo"
  | "jira"
  | "local_markdown";
export type SourceFieldSection = "connection" | "content" | "advanced" | "hidden";

export interface SourceSetupAdapter {
  sourceType: SchemaSourceType;
  displayName: string;
  connectionTitle: string;
  contentTitle: string;
  connection: { mode: SourceConnectionMode; label: string };
  sectionForField: (field: ConfigField, config: ConfigForm) => SourceFieldSection;
  isRequired: (field: ConfigField, config: ConfigForm) => boolean;
  normalizeFieldChange: (field: ConfigField, value: ConfigValue, config: ConfigForm) => ConfigForm;
  normalizeInitialConfig: (config: ConfigForm) => ConfigForm;
  optionLabel: (field: ConfigField, option: string) => string;
  connectionSummary: (config: ConfigForm) => string;
  contentSummary: (config: ConfigForm) => string;
}

const CONNECTION_FIELDS: Record<SchemaSourceType, ReadonlySet<string>> = {
  confluence: new Set(["base_url", "pat"]),
  github_pages: new Set(["auth_mode", "pat"]),
  github_repo: new Set(["connection_mode", "repo_url", "pat", "repo_path"]),
  jira: new Set(["base_url", "auth_mode", "sync_mode", "pat"]),
  local_markdown: new Set(["root"]),
};

const ADAPTERS: Record<SchemaSourceType, SourceSetupAdapter> = {
  confluence: createAdapter({
    sourceType: "confluence",
    displayName: "Confluence",
    connectionTitle: "Confluence site",
    contentTitle: "Pages to sync",
    connection: { mode: "direct", label: "Cloud sync" },
    advancedFields: new Set(["exclude_labels"]),
    visible: (field, config) => isConfluenceFieldVisible(field.key, config),
    required: (field, config) => field.required || isConfluenceFieldRequired(field.key, config),
    normalize: (field, _value, config) => field.key === "base_url"
      ? applyConfluenceUrlInference(config) as ConfigForm
      : config,
    initial: (config) => applyConfluenceUrlInference(config) as ConfigForm,
    connectionSummary: (config) => hostSummary(config.base_url, "Site not configured"),
    contentSummary: (config) => stringValue(config.sync_mode) === "space"
      ? listSummary(config.spaces, "space", "spaces")
      : stringValue(config.page_tree_root)
        ? `Page tree · root ${stringValue(config.page_tree_root)}`
        : "Page tree not selected",
  }),
  github_pages: createAdapter({
    sourceType: "github_pages",
    displayName: "GitHub Pages",
    connectionTitle: "Documentation site",
    contentTitle: "Pages to sync",
    connection: { mode: "direct", label: "Cloud sync" },
    visible: githubPagesFieldVisible,
    required: githubPagesFieldRequired,
    connectionSummary: (config) => optionLabelFor("auth_mode", stringValue(config.auth_mode) || "github_pat"),
    contentSummary: githubPagesContentSummary,
  }),
  github_repo: createAdapter({
    sourceType: "github_repo",
    displayName: "GitHub Repository",
    connectionTitle: "Repository access",
    contentTitle: "Files to sync",
    connection: { mode: "choice", label: "Cloud or local sync" },
    visible: githubRepoFieldVisible,
    required: githubRepoFieldRequired,
    connectionSummary: (config) => stringValue(config.connection_mode) === "local_push"
      ? "Local sync · repository folder selected"
      : hostSummary(config.repo_url, "Repository not configured"),
    contentSummary: (config) => listValue(config.include_paths).length > 0
      ? `${listValue(config.include_paths).length} selected path${listValue(config.include_paths).length === 1 ? "" : "s"}`
      : "Whole repository",
  }),
  jira: createAdapter({
    sourceType: "jira",
    displayName: "Jira",
    connectionTitle: "Jira access",
    contentTitle: "Issues to sync",
    connection: { mode: "choice", label: "Cloud or local sync" },
    visible: jiraFieldVisible,
    required: jiraFieldRequired,
    normalize: normalizeJiraChange,
    connectionSummary: (config) => hostSummary(config.base_url, "Jira site not configured"),
    contentSummary: (config) => stringValue(config.query_mode) === "advanced"
      ? "Advanced query"
      : listSummary(config.projects, "project", "projects"),
  }),
  local_markdown: createAdapter({
    sourceType: "local_markdown",
    displayName: "Local Repository",
    connectionTitle: "Folder",
    contentTitle: "Files to sync",
    connection: { mode: "device", label: "Local sync" },
    advancedFields: new Set(["include", "exclude"]),
    visible: () => true,
    required: (field) => field.required,
    connectionSummary: (config) => stringValue(config.root) ? "Folder selected" : "Choose a folder",
    contentSummary: () => "Markdown, text, JSON, and HTML",
  }),
};

export function sourceSetupAdapterFor(sourceType: string): SourceSetupAdapter {
  if (!isSchemaSourceType(sourceType)) {
    throw new Error(`No source setup adapter registered for ${sourceType}`);
  }
  return ADAPTERS[sourceType];
}

export function isSchemaSourceType(sourceType: string): sourceType is SchemaSourceType {
  return Object.hasOwn(ADAPTERS, sourceType);
}

export function buildDefaultConfig(schema: GeneConfigSchema): ConfigForm {
  return schema.fields.reduce<ConfigForm>((config, field) => {
    if (field.default !== "") config[field.key] = defaultValueForField(field);
    return config;
  }, {});
}

export function serializeConfig(fields: ConfigField[], config: ConfigForm): ConfigForm {
  return fields.reduce<ConfigForm>((result, field) => {
    const value = config[field.key];
    if (field.field_type === "tag_list" || field.field_type === "multi_select") {
      result[field.key] = listValue(value);
    } else if (field.field_type === "boolean") {
      result[field.key] = booleanValue(value);
    } else if (field.field_type === "integer") {
      result[field.key] = value === "" || value == null ? null : Number(value);
    } else if (field.field_type === "secret") {
      const text = stringValue(value);
      if (text.trim() || !config[`${field.key}_configured`]) result[field.key] = text;
    } else {
      result[field.key] = stringValue(value);
    }
    return result;
  }, {});
}

export function firstMissingRequiredField(
  adapter: SourceSetupAdapter,
  fields: ConfigField[],
  config: ConfigForm,
): ConfigField | null {
  for (const field of fields) {
    if (adapter.sectionForField(field, config) === "hidden" || !adapter.isRequired(field, config)) continue;
    const value = config[field.key];
    if (field.field_type === "tag_list" || field.field_type === "multi_select") {
      if (listValue(value).length === 0) return field;
    } else if (field.field_type === "secret" && config[`${field.key}_configured`]) {
      continue;
    } else if (!stringValue(value).trim()) {
      return field;
    }
  }
  return null;
}

export function optionLabel(adapter: SourceSetupAdapter, field: ConfigField, option: string): string {
  return adapter.optionLabel(field, option);
}

export function listValue(value: ConfigValue | undefined): string[] {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return value.split(",").map((item) => item.trim()).filter(Boolean);
  return [];
}

export function stringValue(value: ConfigValue | undefined): string {
  if (Array.isArray(value)) return value.join(", ");
  return value == null ? "" : String(value);
}

export function booleanValue(value: ConfigValue | undefined): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value === "true";
  return Boolean(value);
}

function createAdapter(definition: {
  sourceType: SchemaSourceType;
  displayName: string;
  connectionTitle: string;
  contentTitle: string;
  connection: { mode: SourceConnectionMode; label: string };
  advancedFields?: ReadonlySet<string>;
  visible: (field: ConfigField, config: ConfigForm) => boolean;
  required: (field: ConfigField, config: ConfigForm) => boolean;
  normalize?: (field: ConfigField, value: ConfigValue, config: ConfigForm) => ConfigForm;
  initial?: (config: ConfigForm) => ConfigForm;
  connectionSummary: (config: ConfigForm) => string;
  contentSummary: (config: ConfigForm) => string;
}): SourceSetupAdapter {
  return {
    sourceType: definition.sourceType,
    displayName: definition.displayName,
    connectionTitle: definition.connectionTitle,
    contentTitle: definition.contentTitle,
    connection: definition.connection,
    sectionForField(field, config) {
      if (!definition.visible(field, config)) return "hidden";
      if (field.advanced || definition.advancedFields?.has(field.key)) return "advanced";
      return CONNECTION_FIELDS[definition.sourceType].has(field.key) ? "connection" : "content";
    },
    isRequired: definition.required,
    normalizeFieldChange(field, value, config) {
      const next = { ...config, [field.key]: value };
      return definition.normalize?.(field, value, next) ?? next;
    },
    normalizeInitialConfig: definition.initial ?? ((config) => config),
    optionLabel: (field, option) => optionLabelFor(field.key, option),
    connectionSummary: definition.connectionSummary,
    contentSummary: definition.contentSummary,
  };
}

function defaultValueForField(field: ConfigField): ConfigValue {
  if (field.field_type === "boolean") return field.default === "true";
  if (field.field_type === "integer") return Number(field.default);
  if (field.field_type === "tag_list" || field.field_type === "multi_select") return listValue(field.default);
  return field.default;
}

function githubPagesFieldVisible(field: ConfigField, config: ConfigForm): boolean {
  const authMode = stringValue(config.auth_mode) || "github_pat";
  const syncMode = stringValue(config.sync_mode) || "single_page";
  if (field.field_type === "secret") return authMode !== "none";
  const modes: Record<string, ReadonlySet<string>> = {
    single_page: new Set(["page_url"]),
    subtree: new Set(["root_url", "max_depth", "max_pages", "exclude_url_patterns"]),
    explicit_list: new Set(["pages", "max_pages", "exclude_url_patterns"]),
  };
  const modeFields = new Set(["page_url", "root_url", "max_depth", "max_pages", "exclude_url_patterns", "pages"]);
  return !modeFields.has(field.key) || Boolean(modes[syncMode]?.has(field.key));
}

function githubPagesFieldRequired(field: ConfigField, config: ConfigForm): boolean {
  const syncMode = stringValue(config.sync_mode) || "single_page";
  if (field.key === "pat") return (stringValue(config.auth_mode) || "github_pat") === "github_pat";
  if (field.key === "page_url") return syncMode === "single_page";
  if (field.key === "root_url") return syncMode === "subtree";
  if (field.key === "pages") return syncMode === "explicit_list";
  return field.required;
}

function githubPagesContentSummary(config: ConfigForm): string {
  const mode = stringValue(config.sync_mode) || "single_page";
  if (mode === "subtree") return hostSummary(config.root_url, "Subtree not selected");
  if (mode === "explicit_list") return listSummary(config.pages, "page", "pages");
  return hostSummary(config.page_url, "Page not selected");
}

function githubRepoFieldVisible(field: ConfigField, config: ConfigForm): boolean {
  const mode = stringValue(config.connection_mode) || "cloud_pull";
  if (field.key === "pat") return mode === "cloud_pull";
  if (field.key === "repo_path") return mode === "local_push";
  return field.key !== "include_paths";
}

function githubRepoFieldRequired(field: ConfigField, config: ConfigForm): boolean {
  if (field.key === "pat" || field.key === "include_paths") return false;
  if (field.key === "repo_path") return stringValue(config.connection_mode) === "local_push";
  return field.required;
}

function jiraFieldVisible(field: ConfigField, config: ConfigForm): boolean {
  const authMode = stringValue(config.auth_mode) || "browser_cookie";
  if (field.key === "jira_cookie") return false;
  if (field.key === "pat") return authMode === "pat";
  const queryMode = stringValue(config.query_mode) || "simple";
  if (field.key === "jql") return queryMode === "advanced";
  if (["projects", "issue_types", "jql_filter"].includes(field.key)) return queryMode === "simple";
  return true;
}

function jiraFieldRequired(field: ConfigField, config: ConfigForm): boolean {
  if (field.key === "jira_cookie") return false;
  if (field.key === "pat") return (stringValue(config.auth_mode) || "browser_cookie") === "pat";
  const mode = stringValue(config.query_mode) || "simple";
  if (field.key === "projects") return mode === "simple";
  if (field.key === "jql") return mode === "advanced";
  return field.required;
}

function normalizeJiraChange(field: ConfigField, value: ConfigValue, config: ConfigForm): ConfigForm {
  if (field.key === "auth_mode" && stringValue(value) === "pat") return { ...config, sync_mode: "cloud" };
  if (field.key === "sync_mode" && stringValue(value) === "local_agent") return { ...config, auth_mode: "browser_cookie" };
  return config;
}

function optionLabelFor(fieldKey: string, option: string): string {
  const labels: Record<string, Record<string, string>> = {
    query_mode: { simple: "Simple (projects and issue types)", advanced: "Advanced (JQL)" },
    auth_mode: { browser_cookie: "Browser session", pat: "Personal access token", github_pat: "Personal access token", none: "No authentication" },
    sync_mode: { cloud: "Cloud", local_agent: "Local sync", page_tree: "This page tree", space: "Whole space", single_page: "Single page", subtree: "Subtree", explicit_list: "Selected pages" },
    connection_mode: { cloud_pull: "Cloud access", local_push: "Local sync" },
  };
  return labels[fieldKey]?.[option] ?? option;
}

function hostSummary(value: ConfigValue | undefined, empty: string): string {
  const text = stringValue(value).trim();
  if (!text) return empty;
  try {
    return new URL(text).host;
  } catch {
    return text;
  }
}

function listSummary(value: ConfigValue | undefined, singular: string, plural: string): string {
  const count = listValue(value).length;
  return count > 0 ? `${count} ${count === 1 ? singular : plural}` : `No ${plural} selected`;
}
