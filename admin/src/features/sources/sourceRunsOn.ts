import { Laptop, Server, type LucideIcon } from "lucide-react";
import { isLocalAgentBackedSource } from "./model/localAgentSources";
import type { SourceRow } from "./model/sourceRows";

/** Where the source's sync runs: the server or local sync on a computer. */
export function sourceRunsOn(row: SourceRow): { label: string; icon: LucideIcon } {
  if (isLocalAgentBackedSource(row.source)) return { label: "Local sync", icon: Laptop };
  return { label: "Server", icon: Server };
}
