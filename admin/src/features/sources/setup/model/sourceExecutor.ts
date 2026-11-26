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
 * What the chooser does before a Source can run on one of their devices,
 * or null once one can: set up a device, update one that is online, or
 * start one.
 */
export type DeviceChoiceStep = "set_up" | "update" | "start" | null;

export function deviceChoiceStep(devices: readonly ChooserDevice[]): DeviceChoiceStep {
  if (devices.some((device) => device.state === "online")) return null;
  if (devices.length === 0) return "set_up";
  if (devices.some((device) => device.state === "update_required")) return "update";
  return "start";
}
