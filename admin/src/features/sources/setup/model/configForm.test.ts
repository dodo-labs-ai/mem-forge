import { describe, expect, test } from "vitest";
import { SOURCE_TYPE_FORMS, sourceTypeSchema } from "@/test/sourceTypeForms";
import {
  comparisonString,
  defaultValues,
  displayValue,
  effectiveValue,
  formSchema,
  missingRequired,
  reconcileHiddenChoices,
  serializeConfig,
  visibleChoices,
  visibleFields,
  type ConfigFormState,
  type ConfigSchema,
  type ConfigValues,
} from "./configForm";
import { groupFields, setupSteps } from "./setupSteps";

function field(schema: ConfigSchema, key: string) {
  const found = schema.fields.find((candidate) => candidate.key === key);
  if (!found) throw new Error(`No field ${key}`);
  return found;
}

function visibleKeys(schema: ConfigSchema, state: ConfigFormState): string[] {
  return visibleFields(schema, state).map((candidate) => candidate.key);
}

/** The values after a person sets one field, as the setup page applies a change. */
function change(schema: ConfigSchema, state: ConfigFormState, key: string, value: ConfigValues[string]): ConfigFormState {
  const next = { ...state, values: { ...state.values, [key]: value } };
  return { ...next, values: reconcileHiddenChoices(schema, next) };
}

describe("comparison string", () => {
  test("reads values as the server's comparison string does", () => {
    expect(comparisonString(" PAT ")).toBe("pat");
    expect(comparisonString(365)).toBe("365");
    expect(comparisonString(true)).toBe("true");
    expect(comparisonString(["a", " B", ""])).toBe("a,b");
    expect(comparisonString("A, B", "tag_list")).toBe("a,b");
    expect(comparisonString("A, B")).toBe("a, b");
    expect(comparisonString(null)).toBe("");
    expect(comparisonString([])).toBe("");
  });
});

describe("config form", () => {
  test("renders only declared fields for every registered type", () => {
    for (const gene of SOURCE_TYPE_FORMS.genes) {
      const schema = sourceTypeSchema(gene.name);
      const declared = schema.fields.map((candidate) => candidate.key).sort();
      const drawn = setupSteps(schema, { mode: "create", sharedWorkspace: true })
        .flatMap((step) => (step.group ? groupFields(schema, step.group.key) : []))
        .map((candidate) => candidate.key)
        .sort();
      expect(drawn, gene.name).toEqual(declared);

      const stored = { ...Object.fromEntries(schema.fields.map((candidate) => [candidate.key, candidate.default || "x"])), vault_id: "v-1" };
      const sent = Object.keys(serializeConfig(schema, defaultValues(schema, stored)));
      expect(sent.every((key) => declared.includes(key)), gene.name).toBe(true);
    }
  });

  test("a new Jira source defaults to a personal access token", () => {
    const schema = sourceTypeSchema("jira");
    const state: ConfigFormState = { values: defaultValues(schema) };

    expect(effectiveValue(field(schema, "auth_mode"), state)).toBe("pat");
    expect(visibleKeys(schema, state)).toContain("pat");
    expect(missingRequired(schema, state).map((candidate) => candidate.key)).toContain("pat");
  });

  test("Jira hides the PAT option for local sync and resets auth to browser session", () => {
    const schema = sourceTypeSchema("jira");
    let state: ConfigFormState = { values: { sync_mode: "cloud", auth_mode: "pat" } };
    expect(visibleChoices(schema, field(schema, "auth_mode"), state).map((choice) => choice.value)).toEqual(["pat", "browser_cookie"]);
    expect(visibleKeys(schema, state)).toContain("pat");

    state = change(schema, state, "sync_mode", "local_agent");

    expect(visibleChoices(schema, field(schema, "auth_mode"), state).map((choice) => choice.value)).toEqual(["browser_cookie"]);
    expect(state.values.auth_mode).toBe("browser_cookie");
    expect(visibleKeys(schema, state)).not.toContain("pat");
  });

  test("Jira local sync moves the default PAT to browser session", () => {
    const schema = sourceTypeSchema("jira");
    const state = change(schema, { values: defaultValues(schema) }, "sync_mode", "local_agent");

    expect(state.values.auth_mode).toBe("browser_cookie");
    expect(visibleKeys(schema, state)).not.toContain("pat");
  });

  test("a stored Jira browser session keeps its method", () => {
    const schema = sourceTypeSchema("jira");
    const stored = { base_url: "https://jira.example.com", sync_mode: "cloud", auth_mode: "browser_cookie", projects: ["PAY"] };
    const state: ConfigFormState = { values: defaultValues(schema, stored), stored };

    expect(effectiveValue(field(schema, "auth_mode"), state)).toBe("browser_cookie");
    expect(serializeConfig(schema, state.values).auth_mode).toBe("browser_cookie");
    expect(visibleKeys(schema, state)).not.toContain("pat");
  });

  test("GitHub Pages shows page_url only for a single page", () => {
    const schema = sourceTypeSchema("github_pages");
    const shown = (sync_mode?: string) => visibleKeys(schema, { values: sync_mode ? { sync_mode } : {} });

    expect(shown()).toContain("page_url");
    expect(shown("single_page")).toContain("page_url");
    expect(shown("subtree")).not.toContain("page_url");
    expect(shown("subtree")).toContain("root_url");
    expect(shown("explicit_list")).not.toContain("page_url");
    expect(shown("explicit_list")).toContain("pages");
  });

  test("required applies only while visible", () => {
    const schema = sourceTypeSchema("confluence");
    const values = { base_url: "https://wiki.example.com" };
    const missing = (state: ConfigFormState) => missingRequired(schema, state).map((candidate) => candidate.key);

    expect(missing({ values })).toEqual(["spaces", "pat"]);
    expect(missing({ values: { ...values, sync_mode: "page_tree" } })).toEqual(["page_tree_root", "pat"]);

    const result = formSchema(schema).safeParse({ ...values, sync_mode: "page_tree" });
    expect(result.success).toBe(false);
    expect(result.error?.issues.map((issue) => [issue.path.join("."), issue.message])).toEqual([
      ["page_tree_root", "Page Tree Root is required."],
      ["pat", "Personal Access Token is required."],
    ]);
  });

  test("a configured secret left blank is kept", () => {
    const schema = sourceTypeSchema("confluence");
    const stored = { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["ENG"], pat_configured: true };
    const values = defaultValues(schema, stored);
    expect(values.pat).toBeUndefined();

    const state = { values: { ...values, pat: "" }, stored };

    expect(missingRequired(schema, state)).toEqual([]);
    expect(serializeConfig(schema, state.values)).not.toHaveProperty("pat");
    expect(serializeConfig(schema, { ...state.values, pat: "new-token" })).toHaveProperty("pat", "new-token");
    expect(missingRequired(schema, { values: state.values }).map((candidate) => candidate.key)).toEqual(["pat"]);
  });

  test("a blank integer serialises to null", () => {
    const schema = sourceTypeSchema("github_pages");

    expect(serializeConfig(schema, { max_pages: "" })).toEqual({ max_pages: null });
    expect(serializeConfig(schema, { max_pages: " 250 " })).toEqual({ max_pages: 250 });
    expect(formSchema(schema).safeParse({ sync_mode: "subtree", max_pages: "lots" }).error?.issues).toContainEqual(
      expect.objectContaining({ path: ["max_pages"], message: "Max Pages must be a whole number." }),
    );
  });

  test("a hidden field keeps its value when the mode switches back", () => {
    const schema = sourceTypeSchema("github_pages");
    let state: ConfigFormState = { values: { sync_mode: "single_page", page_url: "https://docs.example.com/guide/" } };

    state = change(schema, state, "sync_mode", "subtree");
    expect(visibleKeys(schema, state)).not.toContain("page_url");
    expect(serializeConfig(schema, state.values)).toMatchObject({ sync_mode: "subtree", page_url: "https://docs.example.com/guide/" });

    state = change(schema, state, "sync_mode", "single_page");
    expect(visibleKeys(schema, state)).toContain("page_url");
    expect(displayValue(field(schema, "page_url"), state)).toBe("https://docs.example.com/guide/");
  });

  test("an untouched field shows its declared default and is sent absent", () => {
    const schema = sourceTypeSchema("confluence");
    const state: ConfigFormState = { values: { base_url: "https://wiki.example.com" } };

    expect(displayValue(field(schema, "sync_mode"), state)).toBe("space");
    expect(displayValue(field(schema, "include_children"), state)).toBe(true);
    expect(serializeConfig(schema, state.values)).toEqual({ base_url: "https://wiki.example.com" });
  });

  test("a derived value is shown and read where nobody set the field, never over a value someone set", () => {
    const schema = sourceTypeSchema("confluence");
    const derived = { sync_mode: "page_tree", page_tree_root: "12345" };

    expect(displayValue(field(schema, "sync_mode"), { values: {}, derived })).toBe("page_tree");
    expect(visibleKeys(schema, { values: {}, derived })).toContain("page_tree_root");
    expect(effectiveValue(field(schema, "sync_mode"), { values: { sync_mode: "space" }, derived })).toBe("space");
    expect(displayValue(field(schema, "sync_mode"), { values: { sync_mode: "space" }, derived })).toBe("space");
  });

  test("a stored value starts the form in the control's shape", () => {
    const schema = sourceTypeSchema("teams");

    expect(defaultValues(schema, { rolling_retention_days: 365, conversation_ids: "19:a, 19:b", initial_history_days: 30 })).toEqual({
      rolling_retention_days: "365",
      conversation_ids: ["19:a", "19:b"],
      initial_history_days: "30",
    });
    const forever = defaultValues(schema, { rolling_retention_days: null });
    expect(forever).toEqual({});
    expect(displayValue(field(schema, "rolling_retention_days"), { values: forever })).toBe("forever");
  });

  test("a multi-select keeps only the visible choices", () => {
    const schema: ConfigSchema = {
      groups: [{ key: "scope", label: "Scope", order: 0, description: "", purpose: "content", panels: [] }],
      fields: [
        { ...field(sourceTypeSchema("jira"), "sync_mode"), group: "scope" },
        {
          ...field(sourceTypeSchema("jira"), "issue_types"),
          visible_when: [],
          group: "scope",
          choices: [
            { value: "Epic", label: "Epic", help_text: "", visible_when: [] },
            { value: "Bug", label: "Bug", help_text: "", visible_when: [{ field: "sync_mode", values: ["cloud"] }] },
          ],
        },
      ],
    };

    const state = change(schema, { values: { issue_types: ["Epic", "Bug"] } }, "sync_mode", "local_agent");

    expect(state.values.issue_types).toEqual(["Epic"]);
  });
});
