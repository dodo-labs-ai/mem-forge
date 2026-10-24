import { useMutation, useQueryClient } from "@tanstack/react-query";
import { localSyncKeys } from "@/features/local-sync";
import { sourceKeys } from "./api";
import { useLocalSetupJob, type LocalSetupJobOptions } from "./useLocalSetupJob";

/** The Teams region a configuration names, as the daemon reads it: EMEA when none is set. */
export function teamsRegion(config: Readonly<Record<string, unknown>>): string {
  return typeof config.region === "string" && config.region.trim() ? config.region.trim().toLowerCase() : "emea";
}

/** Where the Teams session is renewed: the Source whose sign-in expired, or none while one is set up. */
export interface TeamsSignInTarget {
  id?: string;
  config: Readonly<Record<string, unknown>>;
}

/** Every cached Teams session check and conversation list, keyed further by region. */
export const TEAMS_SESSION_KEY = ["teams-session"] as const;

const TEAMS_SIGN_IN: LocalSetupJobOptions = {
  messages: {
    offline: "Start local sync on your computer, then try signing in again.",
    timeout: "Still waiting for Teams sign-in. Check the browser on your computer, then try again.",
    forbidden: "You do not have permission to renew this connection. Ask a workspace admin for help.",
    busy: "A Teams sign-in is still waiting. Finish it in the browser on your computer first.",
    failed: "The Teams sign-in did not complete. Try signing in again.",
  },
};

/**
 * Renews the Teams session on the person's computer through the `teams_auth`
 * setup job, which opens a browser window there for the Microsoft sign-in
 * (ADR 0002). The session belongs to the device, so it serves every Teams
 * Source that syncs there.
 */
export function useTeamsSignIn() {
  const queryClient = useQueryClient();
  const signIn = useLocalSetupJob("teams_auth", TEAMS_SIGN_IN);
  return useMutation({
    mutationFn: (target: TeamsSignInTarget) => signIn({ sourceId: target.id, payload: { region: teamsRegion(target.config) } }),
    onSettled: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
        queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
        queryClient.invalidateQueries({ queryKey: TEAMS_SESSION_KEY }),
      ]),
  });
}
