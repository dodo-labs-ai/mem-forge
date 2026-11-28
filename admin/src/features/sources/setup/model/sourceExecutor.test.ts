import { expect, test } from "vitest";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import { chooserDevices, deviceChoiceStep, deviceFormSchema, executorChoices, initialChosenDevice } from "./sourceExecutor";

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
  [[], null, "set_up"],
  [[{ status: "online", contract: 2 }, { status: "online", contract: 2 }], null, "choose"],
  [[{ status: "offline", contract: 2 }], "dev-0", "start"],
  [[{ status: "online", contract: 1 }], "dev-0", "update"],
  [[{ status: "online", contract: 2 }, { status: "offline", contract: 2 }], "dev-1", "start"],
  [[{ status: "online", contract: 1 }, { status: "online", contract: 2 }], "dev-1", null],
] as const)("devices %o with %s chosen need the step %s", (devices, chosen, step) => {
  expect(deviceChoiceStep(chooserDevices(status([...devices]), "confluence"), chosen)).toBe(step);
});

test("Where it runs starts with the execution device, else the only device, else none", () => {
  const two = chooserDevices(status([{ status: "online", contract: 2 }, { status: "offline", contract: 2 }]), "confluence");
  const one = chooserDevices(status([{ status: "offline", contract: 2 }]), "confluence");
  expect(initialChosenDevice(two, "dev-1")).toBe("dev-1");
  expect(initialChosenDevice(two, "dev-of-another-member")).toBeNull();
  expect(initialChosenDevice(two, null)).toBeNull();
  expect(initialChosenDevice(one, null)).toBe("dev-0");
});

test("a device whose contract differs either way needs an update", () => {
  const devices = chooserDevices(status([{ status: "online", contract: 1 }, { status: "online", contract: 3 }, { status: "offline", contract: 1 }]), "confluence");
  expect(devices.map((device) => device.state)).toEqual(["update_required", "update_required", "offline"]);
});
