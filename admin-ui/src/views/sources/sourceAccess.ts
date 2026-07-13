import type { SourceAccessPolicy } from "@/api/types";

export function sourceAccessSummary(policy: SourceAccessPolicy | null): string {
  if (policy === "private") return "Only me";
  if (policy === "workspace") return "Everyone in this workspace";
  return "Choose who can use this source";
}
