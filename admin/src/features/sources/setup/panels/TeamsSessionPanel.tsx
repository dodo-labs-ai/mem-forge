import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { TEAMS_SESSION_KEY, teamsRegion, useTeamsSignIn } from "../../useTeamsSignIn";
import type { ConfigPanelProps } from "../panels";
import { useTeamsSessionCheck } from "../teamsSession";

function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn(className, "animate-spin")} />;
}

/**
 * The Teams session on the person's computer, for the region the form names.
 * It checks the session without opening a browser (`teams_auth_check`); when
 * there is none, Connect signs in through a browser window on the computer
 * (`teams_auth`, ADR 0002), then checks again.
 */
export default function TeamsSessionPanel({ sourceId, config }: ConfigPanelProps) {
  const region = teamsRegion(config);
  const queryClient = useQueryClient();
  const session = useTeamsSessionCheck(region);
  const signIn = useTeamsSignIn();
  const busy = session.isFetching || signIn.isPending;
  const authenticated = session.data?.authenticated === true;

  function connect() {
    signIn.mutate({ id: sourceId, config: { region } });
  }

  function checkAgain() {
    signIn.reset();
    void queryClient.invalidateQueries({ queryKey: [...TEAMS_SESSION_KEY, region] });
  }

  const error = signIn.error ?? session.error;
  const title = signIn.isPending
    ? "Waiting for Teams sign-in…"
    : session.isPending
      ? "Checking the Teams session…"
      : authenticated
        ? "Teams session active"
        : "Connect Microsoft Teams";
  const message = signIn.isPending
    ? "Finish signing in to Microsoft Teams in the browser window on your computer."
    : authenticated
      ? "Local sync can list and sync Teams conversations from your computer."
      : (session.data?.problem ?? "Local sync reads Teams through a browser session on your computer.");

  return (
    <section aria-label="Teams session" className="space-y-3">
      <Notice
        tone={authenticated ? "ok" : busy ? "live" : "warn"}
        icon={busy ? Spinner : authenticated ? CheckCircle2 : AlertTriangle}
        title={title}
        action={
          authenticated || session.isError ? (
            <Button type="button" size="sm" variant="outline" disabled={busy} onClick={checkAgain}>
              Check again
            </Button>
          ) : (
            <Button type="button" size="sm" variant="outline" disabled={busy} onClick={connect}>
              Connect
            </Button>
          )
        }
      >
        {message}
      </Notice>
      {error ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          {errorMessage(error)}
        </p>
      ) : null}
    </section>
  );
}
