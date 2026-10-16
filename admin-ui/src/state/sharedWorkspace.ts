import { useSyncExternalStore } from "react";

import { currentWorkspaceIsShared, subscribeWorkspaceTarget } from "@/api/client";

/**
 * Whether other people use the selected workspace. Only a shared workspace
 * offers who can use a Source and marks the Memories only their owner can
 * use; in a workspace of one, everything belongs to its one user.
 */
export function useSharedWorkspace(): boolean {
  return useSyncExternalStore(subscribeWorkspaceTarget, currentWorkspaceIsShared, currentWorkspaceIsShared);
}
