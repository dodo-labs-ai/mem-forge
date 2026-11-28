import { expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { deviceUpdateProgress, viewerDeviceUpdates, waitsForDeviceUpdate } from "./deviceUpdates";

type State = "online" | "offline" | "update_required";

function onDevice(
  id: string,
  state: State,
  {
    device = "Alex's MacBook",
    deviceId,
    viewerOwns = true,
    status = "active" as "active" | "paused",
  }: { device?: string; deviceId?: string; viewerOwns?: boolean; status?: "active" | "paused" } = {},
) {
  return makeSource({
    id,
    status,
    ownership: { ...makeSource().ownership, execution_owner_user_id: "local", viewer_is_execution_owner: viewerOwns },
    execution: {
      kind: "local_agent",
      operation: "document_acquisition",
      device: { id: deviceId ?? `dev-${device}`, name: viewerOwns ? device : null, state, not_ready: [], installed_version: "0.2.0b8", required_version: "0.2.0b7", credential: "secret" },
    },
  });
}

test("the viewer's waiting sources are grouped by device; paused, other members' and ready ones are left out", () => {
  const updates = viewerDeviceUpdates([
    onDevice("a", "update_required"),
    onDevice("b", "update_required", { device: "Build box" }),
    onDevice("c", "update_required"),
    onDevice("d", "update_required", { viewerOwns: false }),
    onDevice("e", "update_required", { status: "paused" }),
    onDevice("f", "online"),
    makeSource({ id: "g" }),
  ]);
  expect(updates.map((update) => [update.deviceName, update.sources.map((source) => source.id)])).toEqual([
    ["Alex's MacBook", ["a", "c"]],
    ["Build box", ["b"]],
  ]);
  expect(updates[0]).toMatchObject({ installedVersion: "0.2.0b8", requiredVersion: "0.2.0b7" });
});

test("two computers with the same name are two devices", () => {
  const updates = viewerDeviceUpdates([
    onDevice("a", "update_required", { deviceId: "dev-old" }),
    onDevice("b", "update_required", { deviceId: "dev-new" }),
  ]);
  expect(updates.map((update) => [update.deviceId, update.deviceName])).toEqual([
    ["dev-old", "Alex's MacBook"],
    ["dev-new", "Alex's MacBook"],
  ]);
});

test("a paused source waits for nothing", () => {
  expect(waitsForDeviceUpdate(onDevice("a", "update_required", { status: "paused" }))).toBe(false);
});

test.each([
  [["update_required", "update_required"], "waiting"],
  [["online", "update_required"], "waiting"],
  [["online", "online"], "updated"],
  [["offline", "update_required"], "offline"],
] as const)("an update whose devices now report %o is %s", (states, progress) => {
  const [update] = viewerDeviceUpdates([onDevice("a", "update_required"), onDevice("b", "update_required")]);
  expect(deviceUpdateProgress(update!, [onDevice("a", states[0]), onDevice("b", states[1])])).toBe(progress);
});
