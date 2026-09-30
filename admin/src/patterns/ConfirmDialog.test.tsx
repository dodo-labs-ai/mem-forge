import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ConfirmDialog } from "./ConfirmDialog";

function renderDialog(onConfirm: () => Promise<unknown>, onOpenChange = vi.fn()) {
  render(
    <ConfirmDialog
      open
      onOpenChange={onOpenChange}
      title="Delete Payroll wiki?"
      description="This cannot be undone."
      confirmLabel="Delete source"
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
  const confirm = screen.getByRole("button", { name: "Delete source" });

  expect(confirm).toBeDisabled();
  await userEvent.type(screen.getByLabelText(/to confirm/), "Payroll wiki");
  await userEvent.click(confirm);

  expect(onConfirm).toHaveBeenCalledTimes(1);
  expect(onOpenChange).toHaveBeenCalledWith(false);
});

test("stays open and shows the reason when the action fails", async () => {
  const onOpenChange = renderDialog(() => Promise.reject(new Error("Source is syncing")));
  await userEvent.type(screen.getByLabelText(/to confirm/), "Payroll wiki");
  await userEvent.click(screen.getByRole("button", { name: "Delete source" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Source is syncing");
  expect(onOpenChange).not.toHaveBeenCalledWith(false);
});
