import { Laptop, Server, type LucideIcon } from "lucide-react";
import { isLocalAgentBackedSource } from "./model/localAgentSources";
import type { SourceRow } from "./model/sourceRows";

/**
 * Where the source's sync runs: the server, or local sync on a computer. A
 * device Source names its device once that device's report names the
 * Source, which is when its connection is ready; the name is shown only to
 * the execution owner.
 */
export function sourceRunsOn(row: SourceRow): { label: string; icon: LucideIcon } {
  const device = row.source.execution?.device;
  if (device?.name && row.source.connection_status?.state === "ready") return { label: device.name, icon: Laptop };
  if (isLocalAgentBackedSource(row.source)) return { label: "Local sync", icon: Laptop };
  return { label: "Server", icon: Server };
}
