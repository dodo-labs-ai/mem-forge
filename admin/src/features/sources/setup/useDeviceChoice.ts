import { useQuery } from "@tanstack/react-query";
import { useApi } from "@/api";
import { localSyncStatusQuery } from "@/features/local-sync";
import { chooserDevices, deviceChoiceStep } from "./model/sourceExecutor";

/** How often the chooser's devices are read while a move to My computer waits for one. */
const DEVICE_CHOICE_REFETCH_MS = 10_000;

/**
 * Whether the chooser can save a Source on My computer: one of their own
 * devices is online with the server's acquisition contract for the type.
 * Their devices are read only while `enabled`, and pending until then.
 */
export function useDeviceChoice(sourceType: string, enabled: boolean) {
  const api = useApi();
  const status = useQuery({ ...localSyncStatusQuery(api), enabled, refetchInterval: DEVICE_CHOICE_REFETCH_MS });
  const devices = status.data ? chooserDevices(status.data, sourceType) : [];
  const step = status.data ? deviceChoiceStep(devices) : null;
  return { devices, step, pending: status.isPending, error: status.error, ready: Boolean(status.data) && step === null };
}

export type DeviceChoice = ReturnType<typeof useDeviceChoice>;
