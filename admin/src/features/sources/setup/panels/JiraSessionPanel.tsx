import { useQuery } from "@tanstack/react-query";
import { Copy, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { unwrap, useApi } from "@/api";
import { errorMessage } from "@/lib/errors";
import { StatusBadge, type Tone } from "@/patterns";
import { Button } from "@/ui/button";
import { useJiraSignIn } from "../../useJiraSignIn";
import type { ConfigPanelProps } from "../panels";

const STATUS: Readonly<Record<string, { label: string; tone: Tone }>> = {
  active: { label: "Active", tone: "ok" },
  expired: { label: "Expired", tone: "warn" },
  failed: { label: "Failed", tone: "danger" },
};
const MISSING = { label: "Not signed in", tone: "idle" as Tone };

/** The local command that captures a Jira browser session on the person's computer. */
function jiraRefreshCommand(baseUrl: string): string {
  return `memforge adapter auth jira refresh --base-url ${baseUrl || "<jira-base-url>"}`;
}

/**
 * The Jira browser session a Source reads through: whether one is stored for
 * the site and whose it is (`GET /auth/jira-session`). An existing Source can
 * renew it in place through local sync (`jira_auth`); either way the copyable
 * `memforge adapter auth jira refresh` command captures it from the
 * computer where the person is signed in to Jira.
 */
export default function JiraSessionPanel({ sourceId, config }: ConfigPanelProps) {
  const api = useApi();
  const baseUrl = typeof config.base_url === "string" ? config.base_url.trim() : "";
  const session = useQuery({
    queryKey: ["jira-session", baseUrl] as const,
    queryFn: () => unwrap(api.GET("/api/v1/auth/jira-session", { params: { query: { base_url: baseUrl } } })),
    enabled: baseUrl.startsWith("https://"),
    retry: false,
  });
  const signIn = useJiraSignIn();
  const command = jiraRefreshCommand(baseUrl);
  const status = (session.data && STATUS[session.data.status]) ?? MISSING;
  const principal = session.data?.principal_name || session.data?.principal_email || session.data?.principal_id;

  function renew() {
    if (!sourceId) return;
    signIn.mutate({ id: sourceId, config: { base_url: baseUrl } }, { onSettled: () => void session.refetch() });
  }

  function copy() {
    navigator.clipboard.writeText(command).then(
      () => toast.success("Command copied"),
      () => toast.error("Could not copy"),
    );
  }

  const error = signIn.error ?? session.error ?? (session.data?.last_error ? new Error(session.data.last_error) : null);

  return (
    <section aria-label="Jira browser session" className="space-y-3 rounded-lg border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium">Browser session</h3>
          {baseUrl.startsWith("https://") ? <StatusBadge tone={status.tone}>{session.isPending ? "Checking…" : status.label}</StatusBadge> : null}
        </div>
        <div className="flex gap-2">
          {sourceId ? (
            <Button type="button" size="sm" variant="outline" disabled={!baseUrl || signIn.isPending} onClick={renew}>
              {signIn.isPending ? <Loader2 aria-hidden className="animate-spin" /> : null}
              {signIn.isPending ? "Waiting for sign-in…" : "Sign in again"}
            </Button>
          ) : null}
          <Button type="button" size="sm" variant="ghost" disabled={!baseUrl.startsWith("https://") || session.isFetching} onClick={() => void session.refetch()}>
            <RefreshCw aria-hidden />
            Check again
          </Button>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        {session.data?.status === "active"
          ? `Signed in as ${principal || "an unknown user"}${session.data.browser ? ` through ${session.data.browser}` : ""}.`
          : "MemForge never reads your browser. Capture the session on the computer where you are signed in to Jira, with local sync or this command."}
      </p>
      <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-2">
        <code className="min-w-0 flex-1 font-mono text-xs break-all">{command}</code>
        <Button type="button" size="sm" variant="outline" onClick={copy}>
          <Copy aria-hidden />
          Copy
        </Button>
      </div>
      {error ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          {errorMessage(error)}
        </p>
      ) : null}
    </section>
  );
}
