import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { StepRail, type RailStep } from "./StepRail";

const STEPS: RailStep[] = [
  { key: "basics", label: "Basics", complete: true, summary: "Payroll wiki" },
  { key: "connection", label: "Confluence site", errorCount: 2 },
  { key: "scope", label: "Pages to sync" },
];

function renderRail(onSelect = vi.fn(), current = "connection") {
  render(<StepRail aria-label="Setup steps" steps={STEPS} current={current} onSelect={onSelect} />);
  return onSelect;
}

test("marks the current step, completed steps and each step's issue count", () => {
  renderRail();

  const rail = screen.getByRole("navigation", { name: "Setup steps" });
  expect(rail).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confluence site, 2 issues" })).toHaveAttribute("aria-current", "step");
  expect(screen.getByRole("button", { name: "Basics, complete" })).toHaveTextContent("Payroll wiki");
  expect(screen.getByRole("button", { name: "Pages to sync" })).not.toHaveAttribute("aria-current");
});

test("only the current step is in the tab order", async () => {
  renderRail();

  await userEvent.tab();

  expect(screen.getByRole("button", { name: /Confluence site/ })).toHaveFocus();
});

test("the arrow keys, Home and End move between steps, and Enter opens one", async () => {
  const onSelect = renderRail();
  screen.getByRole("button", { name: /Confluence site/ }).focus();

  await userEvent.keyboard("{ArrowDown}");
  expect(screen.getByRole("button", { name: /Pages to sync/ })).toHaveFocus();
  await userEvent.keyboard("{ArrowDown}");
  expect(screen.getByRole("button", { name: /Pages to sync/ })).toHaveFocus();
  await userEvent.keyboard("{Home}");
  expect(screen.getByRole("button", { name: /Basics/ })).toHaveFocus();
  await userEvent.keyboard("{End}");
  expect(screen.getByRole("button", { name: /Pages to sync/ })).toHaveFocus();
  await userEvent.keyboard("{ArrowUp}{Enter}");

  expect(onSelect).toHaveBeenCalledWith("connection");
});

test("clicking a step opens it", async () => {
  const onSelect = renderRail();

  await userEvent.click(screen.getByRole("button", { name: /Pages to sync/ }));

  expect(onSelect).toHaveBeenCalledWith("scope");
});

test("a long summary stays on one line within the rail and is whole in its tooltip", () => {
  const jql = "Advanced (JQL) · project = SFPAY AND type not in (Epic, Sub-task, Test Execution) ORDER BY updated DESC";
  render(<StepRail aria-label="Setup steps" steps={[{ key: "scope", label: "Issues to sync", complete: true, summary: jql }]} current="scope" onSelect={vi.fn()} />);

  const summary = screen.getByText(jql);
  expect(summary).toHaveAttribute("title", jql);
  expect(summary).toHaveClass("truncate");
  expect(screen.getByRole("navigation", { name: "Setup steps" })).toHaveClass("min-w-0");
});
