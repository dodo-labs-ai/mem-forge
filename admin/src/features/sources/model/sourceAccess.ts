import type { components } from "@/api";
import { formatCount } from "@/lib/format";
import type { Source } from "./types";

export type AccessPolicy = components["schemas"]["CreateSourceRequest"]["access_policy"];
export type SourceAccessTransition = NonNullable<Source["access_transition"]>;

/** Who can use a Source in a shared workspace, as setup and an access change offer it. */
export const ACCESS_CHOICES: readonly { value: AccessPolicy; label: string; description: string }[] = [
  { value: "private", label: "Only me", description: "Only you can find this source or query memories from it." },
  {
    value: "workspace",
    label: "Everyone in this workspace",
    description: "Workspace members can find the source and query its memories.",
  },
];

/** The full name of a policy, such as "Everyone in this workspace". */
export function accessChoiceLabel(policy: AccessPolicy): string {
  return ACCESS_CHOICES.find((choice) => choice.value === policy)?.label ?? policy;
}

/** The short name of a policy, such as "Only me". */
export function accessPolicyLabel(policy: AccessPolicy): string {
  return policy === "private" ? "Only me" : "Workspace";
}

/** The other policy, which an access change offers first. */
export function oppositeAccessPolicy(policy: AccessPolicy): AccessPolicy {
  return policy === "private" ? "workspace" : "private";
}

/**
 * Whether the viewer can start an access change from here: only in a shared
 * workspace, for someone who manages the source, while no change is in
 * progress or failed.
 */
export function canChangeSourceAccess(source: Source, sharedWorkspace: boolean): boolean {
  return sharedWorkspace && Boolean(source.capabilities?.can_change_access) && source.access_state === "active";
}

/** How a source's access reads on its row and in its drawer. */
export type SourceAccessView =
  | { kind: "policy"; label: string; personal: boolean }
  | { kind: "changing"; label: string }
  | { kind: "failed"; label: string };

/**
 * The server lists a source only to someone who can use it, so a changing
 * source reaches its owner alone and an `orphaned_private` one reaches no one.
 */
export function presentSourceAccess(source: Source): SourceAccessView {
  if (source.access_state === "changing") {
    return source.access_transition?.status === "failed"
      ? { kind: "failed", label: "Access change failed" }
      : { kind: "changing", label: "Changing access" };
  }
  return { kind: "policy", label: accessPolicyLabel(source.access_policy), personal: source.access_policy === "private" };
}

/** How far an access change has re-shared the source's memories, as text and a percentage when it is known. */
export function accessTransitionProgress(transition: SourceAccessTransition): { text: string; percent: number | null } {
  const total = transition.total_memories;
  if (total <= 0) return { text: "Preparing existing memories", percent: null };
  const processed = Math.min(transition.processed_memories, total);
  return {
    text: `${formatCount(processed)} of ${formatCount(total)} memories updated`,
    percent: Math.round((processed / total) * 100),
  };
}

/** What the server's refusal of an access change, a retry or a revert means, keyed by the start of its message. */
const REFUSALS: readonly [prefix: string, message: string][] = [
  ["source sync run already active", "A sync of this source is running. Stop it or wait for it to end, then change access."],
  ["source_access_policy_unchanged", "The source already has this access."],
  ["source_access_transition_already_active", "An access change for this source is already in progress."],
  ["source_access_transition_not_retryable", "This access change is no longer waiting for a retry."],
  ["source_access_transition_not_revertible", "This access change is no longer waiting for a revert."],
];

export function accessChangeRefusal(message: string): string {
  return REFUSALS.find(([prefix]) => message.startsWith(prefix))?.[1] ?? message;
}
