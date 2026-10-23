import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, test, vi } from "vitest";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import { ConfigFieldControl } from "./ConfigFieldControl";
import type { ConfigField, FieldValue } from "./model/configForm";

function field(sourceType: string, key: string): ConfigField {
  const found = sourceTypeSchema(sourceType).fields.find((candidate) => candidate.key === key);
  if (!found) throw new Error(`No field ${key}`);
  return found;
}

function Controlled({
  config,
  initial,
  onChange = () => {},
  ...rest
}: {
  config: ConfigField;
  initial: FieldValue;
  onChange?: (value: FieldValue) => void;
  error?: string;
  readOnly?: boolean;
  secretConfigured?: boolean;
}) {
  const [value, setValue] = useState(initial);
  return (
    <ConfigFieldControl
      field={config}
      choices={config.choices}
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
      sourceType="test"
      {...rest}
    />
  );
}

describe("controls by field type", () => {
  test("a string or URL field is a text input named by its label and described by its help text and issue", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("confluence", "base_url")} initial="" onChange={onChange} error="Wiki URL is required." />);

    const input = screen.getByRole("textbox", { name: "Wiki URL" });
    expect(input).toHaveAttribute("type", "url");
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toHaveAccessibleDescription(expect.stringContaining("Wiki URL is required."));
    expect(input).toHaveAccessibleDescription(expect.stringContaining(field("confluence", "base_url").help_text));

    await userEvent.type(input, "https://wiki.example.com");
    expect(onChange).toHaveBeenLastCalledWith("https://wiki.example.com");
  });

  test("a secret is masked and says when a stored value is kept", () => {
    render(<Controlled config={field("confluence", "pat")} initial="" secretConfigured />);

    const input = screen.getByLabelText("Personal Access Token");
    expect(input).toHaveAttribute("type", "password");
    expect(input).toHaveAccessibleDescription(expect.stringContaining("A value is stored. Leave this blank to keep it."));
  });

  test("an integer is a numeric text input", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("teams", "initial_history_days")} initial="14" onChange={onChange} />);

    const input = screen.getByRole("textbox", { name: "Initial History (days)" });
    expect(input).toHaveAttribute("inputmode", "numeric");
    await userEvent.clear(input);
    await userEvent.type(input, "30");
    expect(onChange).toHaveBeenLastCalledWith("30");
  });

  test("a boolean is a switch", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("confluence", "include_children")} initial={true} onChange={onChange} />);

    const toggle = screen.getByRole("switch", { name: "Include Child Pages" });
    expect(toggle).toBeChecked();
    await userEvent.click(toggle);
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  test("a select with a few choices is radio cards labelled by the choices", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("confluence", "sync_mode")} initial="space" onChange={onChange} />);

    expect(screen.getByRole("radiogroup", { name: "Sync Scope" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Whole space" })).toBeChecked();
    await userEvent.click(screen.getByRole("radio", { name: "This page tree" }));
    expect(onChange).toHaveBeenLastCalledWith("page_tree");
  });

  test("a multi-select is a checkbox list", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("jira", "issue_types")} initial={["Epic", "Story"]} onChange={onChange} />);

    expect(screen.getByRole("group", { name: "Issue Types" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Epic" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Bug" })).not.toBeChecked();
    await userEvent.click(screen.getByRole("checkbox", { name: "Bug" }));
    expect(onChange).toHaveBeenLastCalledWith(["Epic", "Story", "Bug"]);
  });

  test("a tag list is a tag input", async () => {
    const onChange = vi.fn();
    render(<Controlled config={field("confluence", "spaces")} initial={["ENG"]} onChange={onChange} />);

    await userEvent.type(screen.getByRole("textbox", { name: "Spaces to Sync" }), "OPS{Enter}");
    expect(onChange).toHaveBeenLastCalledWith(["ENG", "OPS"]);
  });
});

test("a field fixed after create is read-only", () => {
  render(<Controlled config={field("github_repo", "repo_url")} initial="https://github.com/org/repo" readOnly />);

  expect(screen.getByRole("textbox", { name: "Repository URL" })).toHaveAttribute("readonly");
});

test("a widget this admin UI does not have falls back to the control for the field type", () => {
  render(<Controlled config={{ ...field("confluence", "spaces"), widget: "space_picker" }} initial={[]} />);

  expect(screen.getByRole("textbox", { name: "Spaces to Sync" })).toBeInTheDocument();
});
