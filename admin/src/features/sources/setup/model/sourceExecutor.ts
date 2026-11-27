import type { components } from "@/api/schema.gen";
import type { SourceExecutor } from "../../model/localAgentSources";
import type { ConfigSchema } from "./configForm";

type DaemonStatus = components["schemas"]["LocalAgentDaemonStatusResponse"];

/**
 * Where a Source runs, as Configure's Where it runs offers it.
 *
 * - *Server*: the server reaches the provider with the secrets the form
 *   holds, which save requires.
 * - *My computer*: the execution owner's device reaches it with the
 *   credential stored there, so the form holds no secret and no panel of
 *   its connection group, which describe a credential the server holds.
 */

/** The executors a type offers as a choice; empty when it runs in one place only. */
export function executorChoices(executors: readonly SourceExecutor[] | null | undefined): SourceExecutor[] {
  const declared = executors ?? [];
  return declared.length > 1 ? [...declared] : [];
}

/** The form of a Source that runs on a device: the type's form without its secrets and connection panels. */
export function deviceFormSchema(schema: ConfigSchema): ConfigSchema {
  return {
    ...schema,
    fields: schema.fields.filter((field) => field.field_type !== "secret"),
    groups: schema.groups.map((group) => (group.purpose === "connection" ? { ...group, panels: [] } : group)),
  };
}

/** One of the chooser's own devices, with whether it can collect the type being configured. */
export interface ChooserDevice {
  id: string;
  name: string;
  version: string;
  state: "online" | "offline" | "update_required";
}

/**
 * The chooser's own devices from their latest reports. An online device
 * whose acquisition contract for the type differs from the server's needs
 * an update before it can collect it.
 */
export function chooserDevices(status: DaemonStatus, sourceType: string): ChooserDevice[] {
  const required = status.server.acquisition_contracts[sourceType];
  return status.devices.map((device) => ({
    id: device.device_id,
    name: device.name,
    version: device.version,
    state:
      device.status === "offline"
        ? "offline"
        : device.acquisition_contracts[sourceType] === required
          ? "online"
          : "update_required",
  }));
}

/**
 * The device Where it runs starts with: the Source's execution device when
 * it is one of the chooser's, else their only device, else none until they
 * choose.
 */
export function initialChosenDevice(devices: readonly ChooserDevice[], executionDeviceId: string | null | undefined): string | null {
  if (executionDeviceId && devices.some((device) => device.id === executionDeviceId)) return executionDeviceId;
  return devices.length === 1 ? devices[0]!.id : null;
}

/**
 * What the chooser does before the Source can run on the chosen device, or
 * null once it can: set up a device when they have none, choose one, start
 * the chosen one when it is offline, or update it when its acquisition
 * contract for the type differs from the server's.
 */
export type DeviceChoiceStep = "set_up" | "choose" | "start" | "update" | null;

export function deviceChoiceStep(devices: readonly ChooserDevice[], chosenId: string | null): DeviceChoiceStep {
  if (devices.length === 0) return "set_up";
  const chosen = devices.find((device) => device.id === chosenId);
  if (!chosen) return "choose";
  if (chosen.state === "offline") return "start";
  if (chosen.state === "update_required") return "update";
  return null;
}
