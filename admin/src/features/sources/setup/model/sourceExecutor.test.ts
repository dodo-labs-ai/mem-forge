import { expect, test } from "vitest";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import { chooserDevices, deviceChoiceStep, deviceFormSchema, executorChoices } from "./sourceExecutor";

function status(devices: { status: "online" | "offline"; contract: number }[]) {
  return {
    status: devices.some((device) => device.status === "online") ? ("online" as const) : ("offline" as const),
    last_seen_at: null,
    checked_at: "2026-11-27T10:00:00Z",
    stale_after_seconds: 90,
    sources: [],
    server: { version: "0.2.0b7", acquisition_contracts: { confluence: 2 } },
    devices: devices.map((device, index) => ({
      device_id: `dev-${index}`,
      name: `Laptop ${index}`,
      version: "0.2.0b7",
      status: device.status,
      last_seen_at: "2026-11-27T10:00:00Z",
      acquisition_contracts: { confluence: device.contract },
    })),
  };
}

test("a type offers a choice only when it declares both executors", () => {
  expect(executorChoices(["server", "device"])).toEqual(["server", "device"]);
  expect(executorChoices(["device"])).toEqual([]);
  expect(executorChoices(null)).toEqual([]);
});

test("the device form holds no secret and no connection panel", () => {
  const jira = deviceFormSchema(sourceTypeSchema("jira"));
  expect(jira.fields.some((field) => field.field_type === "secret")).toBe(false);
  expect(jira.fields.map((field) => field.key)).toContain("auth_mode");
  expect(jira.groups.find((group) => group.key === "connection")?.panels).toEqual([]);
});

test.each([
  [[], "set_up"],
  [[{ status: "offline", contract: 2 }], "start"],
  [[{ status: "online", contract: 1 }], "update"],
  [[{ status: "online", contract: 3 }, { status: "offline", contract: 2 }], "update"],
  [[{ status: "online", contract: 1 }, { status: "online", contract: 2 }], null],
] as const)("devices %o need the step %s", (devices, step) => {
  expect(deviceChoiceStep(chooserDevices(status([...devices]), "confluence"))).toBe(step);
});

test("a device whose contract differs either way needs an update", () => {
  const devices = chooserDevices(status([{ status: "online", contract: 1 }, { status: "online", contract: 3 }, { status: "offline", contract: 1 }]), "confluence");
  expect(devices.map((device) => device.state)).toEqual(["update_required", "update_required", "offline"]);
});
