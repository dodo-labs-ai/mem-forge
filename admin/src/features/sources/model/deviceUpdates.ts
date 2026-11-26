import type { Source } from "./types";

type SourceDevice = NonNullable<NonNullable<Source["execution"]>["device"]>;

/**
 * One device of the viewer's that needs an update before the Sources it
 * collects can sync: its name and versions, from the source API's
 * `execution.device`, and those Sources.
 */
export interface DeviceUpdate {
  deviceName: string;
  installedVersion: string | null;
  requiredVersion: string;
  sources: Source[];
}

/** Where an open update stands, read again from the current source list. */
export type DeviceUpdateProgress = "waiting" | "offline" | "updated";

/**
 * Whether an active Source waits for its device to be updated: the device
 * is online, but its acquisition contract for the type differs from the
 * server's. A paused Source waits for nothing.
 */
export function waitsForDeviceUpdate(source: Source): boolean {
  return source.status === "active" && source.execution?.device?.state === "update_required";
}

/** Whether the viewer is the one who can update the device a waiting Source runs on. */
export function viewerUpdatesDevice(source: Source): boolean {
  return waitsForDeviceUpdate(source) && source.ownership.viewer_is_execution_owner;
}

/**
 * The viewer's devices that need an update, one entry per device with every
 * Source it holds back, in the order the list names them. Only the
 * execution owner can update their device, so another member's waiting
 * Sources are not here.
 */
export function viewerDeviceUpdates(sources: readonly Source[]): DeviceUpdate[] {
  const byDevice = new Map<string, DeviceUpdate>();
  for (const source of sources) {
    const device = source.execution?.device;
    if (!device?.name || !viewerUpdatesDevice(source)) continue;
    const update = byDevice.get(device.name);
    if (update) update.sources.push(source);
    else byDevice.set(device.name, deviceUpdate(device, [source]));
  }
  return [...byDevice.values()];
}

/** The update of the device a waiting Source runs on, with every Source of the viewer's that the device holds back. */
export function deviceUpdateOf(source: Source, sources: readonly Source[]): DeviceUpdate | null {
  const name = source.execution?.device?.name;
  return viewerDeviceUpdates(sources).find((update) => update.deviceName === name) ?? null;
}

function deviceUpdate(device: SourceDevice, sources: Source[]): DeviceUpdate {
  return {
    deviceName: device.name ?? "",
    installedVersion: device.installed_version,
    requiredVersion: device.required_version,
    sources,
  };
}

/**
 * Where an update stands, from the device each of its Sources now reports:
 * updated once every one of them is online again, offline while any is
 * offline, and waiting otherwise.
 */
export function deviceUpdateProgress(update: DeviceUpdate, current: readonly Source[]): DeviceUpdateProgress {
  const ids = new Set(update.sources.map((source) => source.id));
  const states = current.filter((source) => ids.has(source.id)).map((source) => source.execution?.device?.state);
  if (states.some((state) => state === "offline")) return "offline";
  if (states.length > 0 && states.every((state) => state === "online")) return "updated";
  return "waiting";
}

/**
 * The update as the current source list shows it: the versions the device
 * reports now, for the Sources the update was opened for.
 */
export function currentDeviceUpdate(update: DeviceUpdate, current: readonly Source[]): DeviceUpdate {
  const ids = new Set(update.sources.map((source) => source.id));
  const device = current.find((source) => ids.has(source.id))?.execution?.device;
  return device ? { ...update, installedVersion: device.installed_version, requiredVersion: device.required_version } : update;
}
