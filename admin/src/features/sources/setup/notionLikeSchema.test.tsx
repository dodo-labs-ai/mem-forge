import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test } from "vitest";
import { ConfigGroupFields } from "./ConfigGroupFields";
import { reconcileHiddenChoices, serializeConfig, type ConfigField, type ConfigFormState, type ConfigSchema } from "./model/configForm";
import { setupSteps } from "./model/setupSteps";

function textField(overrides: Partial<ConfigField> & Pick<ConfigField, "key" | "label" | "field_type" | "group">): ConfigField {
  return {
    advanced: false,
    choices: [],
    default: "",
    fixed_after_create: false,
    help_text: "",
    options: [],
    order: 0,
    placeholder: "",
    required: false,
    scope: false,
    visible_when: [],
    widget: null,
    ...overrides,
  };
}

/** A new document type the admin UI has never seen: no widgets, no panels it knows, no per-type code. */
const NOTION_LIKE: ConfigSchema = {
  project_field: null,
  groups: [
    { key: "connection", label: "Notion access", order: 0, description: "Connect an internal integration.", purpose: "connection", panels: [] },
    {
      key: "content",
      label: "Pages to sync",
      order: 1,
      description: "",
      purpose: "content",
      panels: [{ name: "notion_session", visible_when: [] }],
    },
  ],
  fields: [
    textField({ key: "token", label: "Internal integration token", field_type: "secret", group: "connection", required: true }),
    textField({
      key: "select_by",
      label: "Select pages by",
      field_type: "select",
      group: "content",
      default: "pages",
      choices: [
        { value: "pages", label: "Root pages", help_text: "", visible_when: [] },
        { value: "databases", label: "Databases", help_text: "", visible_when: [] },
      ],
      order: 0,
    }),
    textField({
      key: "root_pages",
      label: "Root pages",
      field_type: "tag_list",
      group: "content",
      order: 1,
      help_text: "Page URLs or ids",
      visible_when: [{ field: "select_by", values: ["pages"] }],
    }),
    textField({
      key: "databases",
      label: "Databases",
      field_type: "tag_list",
      group: "content",
      order: 2,
      visible_when: [{ field: "select_by", values: ["databases"] }],
    }),
    textField({ key: "max_pages", label: "Max pages", field_type: "integer", group: "content", order: 3, default: "1000", advanced: true }),
  ],
};

function NotionLikeForm({ issues }: { issues?: Record<string, string> }) {
  const [state, setState] = useState<ConfigFormState>({ values: {} });
  const steps = setupSteps(NOTION_LIKE, { mode: "create", sharedWorkspace: false });
  return (
    <>
      {steps.map((step) =>
        step.group ? (
          <section key={step.key} aria-label={step.label}>
            <ConfigGroupFields
              schema={NOTION_LIKE}
              group={step.group}
              state={state}
              sourceType="notion"
              issues={issues}
              onChange={(key, value) =>
                setState((current) => {
                  const next = { ...current, values: { ...current.values, [key]: value } };
                  return { ...next, values: reconcileHiddenChoices(NOTION_LIKE, next) };
                })
              }
            />
          </section>
        ) : null,
      )}
      <output aria-label="Config">{JSON.stringify(serializeConfig(NOTION_LIKE, state.values))}</output>
    </>
  );
}

test("a type with no widgets renders entirely from the schema", async () => {
  render(<NotionLikeForm />);

  const access = screen.getByRole("region", { name: "Notion access" });
  expect(within(access).getByText("Connect an internal integration.")).toBeInTheDocument();
  expect(within(access).getByLabelText("Internal integration token")).toHaveAttribute("type", "password");

  const content = screen.getByRole("region", { name: "Pages to sync" });
  expect(within(content).getByRole("radio", { name: "Root pages" })).toBeChecked();
  expect(within(content).getByRole("textbox", { name: "Root pages" })).toHaveAccessibleDescription("Page URLs or ids");
  expect(within(content).queryByRole("textbox", { name: "Databases" })).not.toBeInTheDocument();
  expect(within(content).getByText("Advanced settings")).toBeInTheDocument();
  expect(within(content).getByRole("textbox", { name: "Max pages", hidden: true })).not.toBeVisible();

  await userEvent.type(within(content).getByRole("textbox", { name: "Root pages" }), "https://notion.so/Team-0123{Enter}");
  await userEvent.click(within(content).getByRole("radio", { name: "Databases" }));

  expect(within(content).queryByRole("textbox", { name: "Root pages" })).not.toBeInTheDocument();
  expect(within(content).getByRole("textbox", { name: "Databases" })).toBeInTheDocument();
  expect(JSON.parse(screen.getByRole("status", { name: "Config" }).textContent ?? "")).toEqual({
    root_pages: ["https://notion.so/Team-0123"],
    select_by: "databases",
  });
});

test("Advanced settings opens by itself when one of its fields has an issue", () => {
  render(<NotionLikeForm issues={{ max_pages: "Max pages must be a whole number." }} />);

  expect(screen.getByRole("textbox", { name: "Max pages" })).toBeVisible();
  expect(screen.getByRole("textbox", { name: "Max pages" })).toHaveAccessibleDescription("Max pages must be a whole number.");
});
