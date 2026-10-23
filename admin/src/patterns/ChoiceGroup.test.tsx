import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ChoiceGroup, type Choice } from "./ChoiceGroup";

const FEW: Choice[] = [
  { value: "page_tree", label: "This page tree", description: "A page and the pages under it" },
  { value: "space", label: "Whole space" },
];
const MANY: Choice[] = ["a", "b", "c", "d", "e"].map((value) => ({ value, label: `Choice ${value.toUpperCase()}` }));

function renderGroup(choices: Choice[], value: string, onValueChange = vi.fn()) {
  render(
    <>
      <span id="group-label">Sync scope</span>
      <ChoiceGroup id="scope" aria-labelledby="group-label" choices={choices} value={value} onValueChange={onValueChange} />
    </>,
  );
  return onValueChange;
}

test("up to four choices are radio cards with their descriptions", async () => {
  const onValueChange = renderGroup(FEW, "space");

  expect(screen.getByRole("radiogroup", { name: "Sync scope" })).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: "Whole space" })).toBeChecked();
  expect(screen.getByRole("radio", { name: "This page tree" })).toHaveAccessibleDescription("A page and the pages under it");

  await userEvent.click(screen.getByRole("radio", { name: "This page tree" }));

  expect(onValueChange).toHaveBeenCalledWith("page_tree");
});

test("more than four choices are a select", async () => {
  const onValueChange = renderGroup(MANY, "b");

  const select = screen.getByRole("combobox", { name: "Sync scope" });
  expect(select).toHaveTextContent("Choice B");

  await userEvent.click(select);
  await userEvent.click(await screen.findByRole("option", { name: "Choice E" }));

  expect(onValueChange).toHaveBeenCalledWith("e");
});
