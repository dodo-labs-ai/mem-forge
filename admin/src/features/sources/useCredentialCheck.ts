import { useMutation, useQueryClient } from "@tanstack/react-query";
import { localSyncKeys } from "@/features/local-sync";
import { sourceKeys } from "./api";
import { useLocalSetupJob, type LocalSetupJobOptions } from "./useLocalSetupJob";

/**
 * The Source whose credential is checked, by its id and type, and for a
 * device Source its execution device, so that an offline device fails fast.
 */
export interface CredentialCheckTarget {
  id: string;
  type: string;
  deviceId?: string;
}

const SIGN_IN: LocalSetupJobOptions = {
  messages: {
    offline: "Start local sync on your computer, then try signing in again.",
    timeout: "Still waiting for sign-in. Check the browser on your computer, then try again.",
    forbidden: "You do not have permission to renew this connection. Ask a workspace admin for help.",
    busy: "Check the previous source's sign-in before starting another connection.",
    failed: "The local sign-in did not complete. Try signing in again.",
  },
  describeFailure: (detail) =>
    detail === "principal_changed"
      ? "Signed in with a different account. Use the account this source was set up with, then try again."
      : undefined,
};

/**
 * Checks a Source's credential on the person's computer with a
 * `credential_check` job, which the server builds from the Source. A
 * browser session is captured from the computer's browser: a server
 * Source's is stored on the server, and a device Source's stays in the
 * device's keyring. A device Source's stored token is checked against the
 * provider. Every list that shows the Source's readiness reads again after.
 */
export function useCredentialCheck() {
  const queryClient = useQueryClient();
  const check = useLocalSetupJob("credential_check", SIGN_IN);
  return useMutation({
    mutationFn: (source: CredentialCheckTarget) =>
      check({ sourceId: source.id, sourceType: source.type, deviceId: source.deviceId, payload: {} }),
    onSettled: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
        queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
      ]),
  });
}
