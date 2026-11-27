import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { unwrap, useApi } from "@/api";
import { localSyncKeys, localSyncStatusQuery } from "@/features/local-sync";
import { sourceKeys } from "../api";
import { chooserDevices, deviceChoiceStep, initialChosenDevice } from "./model/sourceExecutor";

/** How often the chooser's devices are read while Where it runs waits for the chosen one. */
const DEVICE_CHOICE_REFETCH_MS = 10_000;

/**
 * The chooser's device in Where it runs: their own devices from their
 * reports, the one chosen, and the step that makes it ready. It starts with
 * the Source's execution device when that is one of theirs, else their only
 * device. Their devices are read only while `enabled`, and pending until
 * then; `ready` once the chosen device is online with the server's
 * acquisition contract for the type.
 */
export function useDeviceChoice(sourceType: string, enabled: boolean, executionDeviceId: string | null) {
  const api = useApi();
  const status = useQuery({ ...localSyncStatusQuery(api), enabled, refetchInterval: DEVICE_CHOICE_REFETCH_MS });
  const [picked, setPicked] = useState<string | null>(null);
  const devices = status.data ? chooserDevices(status.data, sourceType) : [];
  const chosenId = picked ?? initialChosenDevice(devices, executionDeviceId);
  const chosen = devices.find((device) => device.id === chosenId) ?? null;
  const step = status.data ? deviceChoiceStep(devices, chosenId) : null;
  return {
    devices,
    chosen,
    choose: setPicked,
    step,
    pending: status.isPending,
    error: status.error,
    ready: Boolean(status.data) && step === null,
  };
}

export type DeviceChoice = ReturnType<typeof useDeviceChoice>;

/**
 * Forgets one of the person's devices, such as a computer they no longer
 * use. Each Source it ran loses its execution device and shows Finish setup
 * until its owner chooses another; a daemon still running there appears
 * again with its next report.
 */
export function useRemoveDevice() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (deviceId: string) =>
      unwrap(api.DELETE("/api/cloud/local-agent/devices/{device_id}", { params: { path: { device_id: deviceId } } })),
    onSettled: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: localSyncKeys.status }),
        queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
      ]),
  });
}
