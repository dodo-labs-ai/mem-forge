import { Laptop, Plug, Server, type LucideIcon } from "lucide-react";
import { isLocalAgentBackedSource } from "./model/localAgentSources";
import { isManagedSourceType } from "./model/managedSources";
import type { SourceRow } from "./model/sourceRows";

/** Where the source's sync runs: the server, local sync on a computer, or a coding-agent plugin. */
export function sourceRunsOn(row: SourceRow): { label: string; icon: LucideIcon } {
  if (isManagedSourceType(row.source.type)) return { label: "Plugin", icon: Plug };
  if (isLocalAgentBackedSource(row.source)) return { label: "Local sync", icon: Laptop };
  return { label: "Server", icon: Server };
}
