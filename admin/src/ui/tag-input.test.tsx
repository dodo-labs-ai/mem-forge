import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test, vi } from "vitest";
import { TagInput } from "./tag-input";

function Controlled({ initial = [], onChange }: { initial?: string[]; onChange?: (value: string[]) => void }) {
  const [value, setValue] = useState(initial);
  return (
    <TagInput
      aria-label="Spaces"
      value={value}
      onValueChange={(next) => {
        setValue(next);
        onChange?.(next);
      }}
    />
  );
}

function chips(): string[] {
  return screen.queryAllByRole("button", { name: /^Remove / }).map((button) => button.getAttribute("aria-label")!.slice(7));
}

test("Enter and a comma add what was typed as a chip", async () => {
  render(<Controlled />);
  const input = screen.getByRole("textbox", { name: "Spaces" });

  await userEvent.type(input, "ENG{Enter}OPS,");

  expect(chips()).toEqual(["ENG", "OPS"]);
  expect(input).toHaveValue("");
});

test("a pasted comma list adds one chip per item, skipping blanks and repeats", async () => {
  render(<Controlled initial={["ENG"]} />);
  const input = screen.getByRole("textbox", { name: "Spaces" });

  fireEvent.paste(input, { clipboardData: { getData: () => "OPS, ENG,, HR\nFIN" } });

  expect(chips()).toEqual(["ENG", "OPS", "HR", "FIN"]);
});

test("Backspace in the empty input removes the last chip, and each chip has a remove button", async () => {
  const onChange = vi.fn();
  render(<Controlled initial={["ENG", "OPS", "HR"]} onChange={onChange} />);

  await userEvent.type(screen.getByRole("textbox", { name: "Spaces" }), "{Backspace}");
  expect(chips()).toEqual(["ENG", "OPS"]);

  await userEvent.click(screen.getByRole("button", { name: "Remove ENG" }));
  expect(chips()).toEqual(["OPS"]);
  expect(onChange).toHaveBeenLastCalledWith(["OPS"]);
});

test("leaving the input adds what is left in it", async () => {
  render(<Controlled />);

  await userEvent.type(screen.getByRole("textbox", { name: "Spaces" }), "ENG");
  await userEvent.tab();

  expect(chips()).toEqual(["ENG"]);
});
