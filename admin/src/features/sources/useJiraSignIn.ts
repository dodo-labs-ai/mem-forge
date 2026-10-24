import { useMutation, useQueryClient } from "@tanstack/react-query";
import { localSyncKeys } from "@/features/local-sync";
import { sourceKeys } from "./api";
import { useLocalSetupJob, type LocalSetupJobOptions } from "./useLocalSetupJob";

/** The Jira Source whose browser session is renewed, with the configuration whose `base_url` names the site. */
export interface JiraSignInTarget {
  id: string;
  config: Readonly<Record<string, unknown>>;
}

const JIRA_SIGN_IN: LocalSetupJobOptions = {
  messages: {
    offline: "Start local sync on your computer, then try signing in again.",
    timeout: "Still waiting for Jira sign-in. Check the browser on your computer, then try again.",
    forbidden: "You do not have permission to renew this connection. Ask a workspace admin for help.",
    busy: "Check the previous source's sign-in before starting another connection.",
    failed: "The local sign-in did not complete. Try signing in again.",
  },
  describeFailure: (detail) =>
    detail === "principal_changed"
      ? "Signed in with a different Jira account. Use the account this source was set up with, then try again."
      : undefined,
};

/** Renews a Jira browser session through the user's existing local daemon. */
export function useJiraSignIn() {
  const queryClient = useQueryClient();
  const renew = useLocalSetupJob("jira_auth", JIRA_SIGN_IN);
  return useMutation({
    mutationFn: async (source: JiraSignInTarget) => {
      const baseUrl = typeof source.config.base_url === "string" ? source.config.base_url.trim() : "";
      if (!baseUrl) throw new Error("The Jira URL is missing. Configure this source before signing in.");
      return renew({ sourceId: source.id, payload: { base_url: baseUrl, auth_mode: "browser_cookie" } });
    },
    onSettled: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
        queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
      ]),
  });
}
