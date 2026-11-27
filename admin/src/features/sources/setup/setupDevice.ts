import { createContext, useContext } from "react";

/**
 * The device chosen in Where it runs while Configure shows the device
 * choice. A device-bound setup job of the form, such as the folder picker,
 * runs on it; without one, a saved device Source's jobs run on its
 * execution device.
 */
export const SetupDeviceContext = createContext<string | undefined>(undefined);

export function useSetupDevice(): string | undefined {
  return useContext(SetupDeviceContext);
}
