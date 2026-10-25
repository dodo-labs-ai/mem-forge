import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ConfirmDialog } from "./ConfirmDialog";

function renderDialog(onConfirm: () => Promise<unknown>, onOpenChange = vi.fn()) {
  render(
    <ConfirmDialog
      open
      onOpenChange={onOpenChange}
      title="Archive Payroll wiki?"
      description="This cannot be undone."
      confirmLabel="Archive"
      tone="danger"
      confirmationText="Payroll wiki"
      onConfirm={onConfirm}
    />,
  );
  return onOpenChange;
}

test("confirms only after the exact text is typed", async () => {
  const onConfirm = vi.fn().mockResolvedValue(undefined);
  const onOpenChange = renderDialog(onConfirm);
  const confirm = screen.getByRole("button", { name: "Archive" });

  expect(confirm).toBeDisabled();
  await userEvent.type(screen.getByLabelText(/to confirm/), "Payroll wiki");
  await userEvent.click(confirm);

  expect(onConfirm).toHaveBeenCalledTimes(1);
  expect(onOpenChange).toHaveBeenCalledWith(false);
});

test("stays open and shows the reason when the action fails", async () => {
  const onOpenChange = renderDialog(() => Promise.reject(new Error("Source is syncing")));
  await userEvent.type(screen.getByLabelText(/to confirm/), "Payroll wiki");
  await userEvent.click(screen.getByRole("button", { name: "Archive" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Source is syncing");
  expect(onOpenChange).not.toHaveBeenCalledWith(false);
});

function renderTwoStepDialog(onConfirm: () => Promise<unknown>, ready = true) {
  render(
    <ConfirmDialog
      open
      onOpenChange={vi.fn()}
      title="Delete Payroll?"
      description="708 memories move to the Unsorted project."
      continueLabel="Continue"
      confirmationText="PAY"
      confirmLabel="Delete project"
      tone="danger"
      ready={ready}
      onConfirm={onConfirm}
    />,
  );
}

test("with a continue step, the consequences come first and the typed confirmation second", async () => {
  const onConfirm = vi.fn().mockResolvedValue(undefined);
  renderTwoStepDialog(onConfirm);

  expect(screen.getByText("708 memories move to the Unsorted project.")).toBeInTheDocument();
  expect(screen.queryByLabelText(/to confirm/)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Delete project" })).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByLabelText(/to confirm/)).toHaveFocus();
  expect(screen.getByRole("button", { name: "Delete project" })).toBeDisabled();

  await userEvent.type(screen.getByLabelText(/to confirm/), "PAY");
  await userEvent.click(screen.getByRole("button", { name: "Delete project" }));
  expect(onConfirm).toHaveBeenCalledTimes(1);
});

test("a dialog that is not ready cannot go ahead", () => {
  renderTwoStepDialog(vi.fn(), false);
  expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
});

test("the confirmation text can be selected and copied", async () => {
  const user = userEvent.setup();
  renderDialog(vi.fn().mockResolvedValue(undefined));
  const label = screen.getByText("Payroll wiki").closest("label");

  expect(label).toHaveClass("select-text");
  expect(label).not.toHaveClass("select-none");
  await user.click(screen.getByRole("button", { name: "Copy Payroll wiki" }));

  expect(await navigator.clipboard.readText()).toBe("Payroll wiki");
});

test("a dialog that asks for the text at once opens on its input, not on the Copy button", async () => {
  renderDialog(vi.fn().mockResolvedValue(undefined));

  await vi.waitFor(() => expect(screen.getByLabelText(/to confirm/)).toHaveFocus());
});
