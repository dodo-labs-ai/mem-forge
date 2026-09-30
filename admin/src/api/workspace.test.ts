import { expect, test, vi } from "vitest";
import { createWorkspaceController, type WorkspaceTarget } from "./workspace";

const WORKSPACE_A: WorkspaceTarget = {
  resourceBaseUrl: "/api/cloud/workspaces/ws-a/v1/",
  localAgentBaseUrl: "/api/cloud/workspaces/ws-a/local-agent",
  workspaceId: "ws-a",
  label: "Payroll",
};

test("a new workspace drops cached data and notifies subscribers", () => {
  const onChange = vi.fn();
  const listener = vi.fn();
  const controller = createWorkspaceController(onChange);
  controller.subscribe(listener);

  controller.setTarget(WORKSPACE_A);

  expect(onChange).toHaveBeenCalledTimes(1);
  expect(listener).toHaveBeenCalledTimes(1);
  expect(controller.current()?.resourceBaseUrl).toBe("/api/cloud/workspaces/ws-a/v1");
});

test("a renamed workspace keeps cached data and notifies subscribers", () => {
  const onChange = vi.fn();
  const listener = vi.fn();
  const controller = createWorkspaceController(onChange, WORKSPACE_A);
  controller.subscribe(listener);

  controller.setTarget({ ...WORKSPACE_A, label: "Payroll EU" });

  expect(onChange).not.toHaveBeenCalled();
  expect(listener).toHaveBeenCalledTimes(1);
  expect(controller.current()?.label).toBe("Payroll EU");
});

test("setting the same target changes nothing", () => {
  const onChange = vi.fn();
  const listener = vi.fn();
  const controller = createWorkspaceController(onChange, WORKSPACE_A);
  controller.subscribe(listener);

  controller.setTarget({ ...WORKSPACE_A });

  expect(onChange).not.toHaveBeenCalled();
  expect(listener).not.toHaveBeenCalled();
});
