import { useQuery } from "@tanstack/react-query";
import { TEAMS_SESSION_KEY } from "../useTeamsSignIn";
import { LocalSetupJobError, useLocalSetupJob } from "../useLocalSetupJob";

/** Whether the person's computer holds a usable Teams session for a region. */
export interface TeamsSessionStatus {
  authenticated: boolean;
  /** Why there is no usable session, in words the panel can show; null when there is one. */
  problem: string | null;
}

const NO_SESSION = "No active Teams session was found. Sign in to Teams in Chrome on your computer, then connect.";

/** The daemon's words for a missing or expired session, reworded for the panel. */
function sessionProblem(detail: string | null | undefined): string {
  const text = detail?.trim();
  if (!text) return NO_SESSION;
  const lower = text.toLowerCase();
  return lower.includes("token") || lower.includes("no teams session") || lower.includes("session expired") ? NO_SESSION : text;
}

/**
 * The Teams session check for a region, through the `teams_auth_check` setup
 * job, which reads the session on the person's computer without opening a
 * browser. The session panel and the conversation picker share it.
 */
export function useTeamsSessionCheck(region: string) {
  const check = useLocalSetupJob("teams_auth_check");
  return useQuery({
    queryKey: [...TEAMS_SESSION_KEY, region, "check"] as const,
    queryFn: async (): Promise<TeamsSessionStatus> => {
      try {
        const result = await check({ payload: { region } });
        return { authenticated: result.authenticated, problem: result.authenticated ? null : sessionProblem(result.error) };
      } catch (error) {
        // A missing session completes the check as failed, with `authenticated: false` in its result.
        if (error instanceof LocalSetupJobError && error.result.authenticated === false) {
          return { authenticated: false, problem: sessionProblem(error.message) };
        }
        throw error;
      }
    },
    retry: false,
    staleTime: Infinity,
  });
}

/** The conversations the Teams session can read, through the `teams_browse` setup job, once the session is known to be usable. */
export function useTeamsConversations(region: string, enabled: boolean) {
  const browse = useLocalSetupJob("teams_browse");
  return useQuery({
    queryKey: [...TEAMS_SESSION_KEY, region, "conversations"] as const,
    queryFn: () => browse({ payload: { region } }),
    enabled,
    retry: false,
    staleTime: 60_000,
  });
}
