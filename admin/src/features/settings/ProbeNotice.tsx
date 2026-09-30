import { AlertCircle, Check, Info } from "lucide-react";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import type { ProbeOutcome, ProbeTone } from "./model/probeOutcome";

const PROBE_ICONS = { ok: Check, warn: Info, danger: AlertCircle } satisfies Record<ProbeTone, unknown>;

interface ProbeNoticeProps {
  outcome: ProbeOutcome;
  onUseBaseUrl: (baseUrl: string) => void;
}

/** The result of Test connection, with the base URL the server suggests when it has one. */
export function ProbeNotice({ outcome, onUseBaseUrl }: ProbeNoticeProps) {
  const suggestion = outcome.suggestedBaseUrl;
  return (
    <Notice
      tone={outcome.tone}
      icon={PROBE_ICONS[outcome.tone]}
      action={
        suggestion ? (
          <Button type="button" variant="outline" size="sm" onClick={() => onUseBaseUrl(suggestion)}>
            Use {new URL(suggestion).hostname}
          </Button>
        ) : null
      }
    >
      {outcome.message}
    </Notice>
  );
}
