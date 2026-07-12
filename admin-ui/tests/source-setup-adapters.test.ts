import assert from "node:assert/strict";

import type { ConfigField, GeneConfigSchema } from "../src/api/types.js";
import {
  buildDefaultConfig,
  firstMissingRequiredField,
  isSchemaSourceType,
  serializeConfig,
  sourceSetupAdapterFor,
} from "../src/views/sources/sourceSetupAdapters.js";

const field = (input: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField => ({
  key: input.key,
  label: input.label ?? input.key,
  field_type: input.field_type ?? "string",
  required: input.required ?? false,
  placeholder: input.placeholder ?? "",
  help_text: input.help_text ?? "",
  group: input.group ?? "scope",
  order: input.order ?? 0,
  default: input.default ?? "",
  options: input.options ?? [],
  advanced: input.advanced ?? false,
});

assert.equal(isSchemaSourceType("confluence"), true);
assert.equal(isSchemaSourceType("teams"), false);
assert.throws(
  () => sourceSetupAdapterFor("future_source"),
  /No source setup adapter registered/,
  "new source types must add an explicit adapter instead of falling into a generic form",
);

const local = sourceSetupAdapterFor("local_markdown");
assert.equal(local.sectionForField(field({ key: "root", required: true }), {}), "connection");
assert.equal(local.sectionForField(field({ key: "include" }), {}), "advanced");
assert.equal(local.sectionForField(field({ key: "exclude" }), {}), "advanced");

const confluence = sourceSetupAdapterFor("confluence");
assert.equal(confluence.sectionForField(field({ key: "exclude_labels" }), {}), "advanced");

const github = sourceSetupAdapterFor("github_repo");
assert.equal(
  github.sectionForField(field({ key: "repo_path" }), { connection_mode: "cloud_pull" }),
  "hidden",
);
assert.equal(
  github.sectionForField(field({ key: "repo_path" }), { connection_mode: "local_push" }),
  "connection",
);
assert.equal(
  github.sectionForField(field({ key: "include_paths" }), { connection_mode: "cloud_pull" }),
  "hidden",
  "repository paths are owned by the tree picker instead of a second text field",
);

const pages = sourceSetupAdapterFor("github_pages");
assert.equal(
  pages.sectionForField(field({ key: "page_url" }), { sync_mode: "subtree" }),
  "hidden",
);
assert.equal(
  pages.sectionForField(field({ key: "root_url" }), { sync_mode: "subtree" }),
  "content",
);

const jira = sourceSetupAdapterFor("jira");
const switched = jira.normalizeFieldChange(
  field({ key: "sync_mode" }),
  "local_agent",
  { auth_mode: "pat", sync_mode: "cloud" },
);
assert.equal(switched.auth_mode, "browser_cookie");

const schema: GeneConfigSchema = {
  groups: [],
  fields: [
    field({ key: "base_url", required: true }),
    field({ key: "include_comments", field_type: "boolean", default: "true" }),
    field({ key: "projects", field_type: "tag_list", default: "PAY, ARCH" }),
  ],
};
const defaults = buildDefaultConfig(schema);
assert.deepEqual(defaults, { include_comments: true, projects: ["PAY", "ARCH"] });
assert.equal(firstMissingRequiredField(jira, schema.fields, defaults)?.key, "base_url");
assert.deepEqual(serializeConfig(schema.fields, { ...defaults, base_url: "https://jira.example" }), {
  base_url: "https://jira.example",
  include_comments: true,
  projects: ["PAY", "ARCH"],
});
