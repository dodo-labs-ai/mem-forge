import { createContext, useContext, type ReactNode } from "react";
import type { AdminExtension } from "./contract";

const ExtensionContext = createContext<AdminExtension | undefined>(undefined);

export function ExtensionProvider({ extension, children }: { extension?: AdminExtension; children: ReactNode }) {
  return <ExtensionContext.Provider value={extension}>{children}</ExtensionContext.Provider>;
}

/** The mounted extension, or `undefined` in the standalone build. */
export function useExtension(): AdminExtension | undefined {
  return useContext(ExtensionContext);
}
