import type { SourceAccessPolicy } from "@/api/types";

export function sourceAccessSummary(policy: SourceAccessPolicy | null): string {
  if (policy === "private") return "Only me";
  if (policy === "workspace") return "Everyone in this workspace";
  return "Choose who can use this source";
}

/** A workspace of one keeps every Source to its one user, who never chooses access. */
export const PERSONAL_WORKSPACE_ACCESS_POLICY: SourceAccessPolicy = "private";
