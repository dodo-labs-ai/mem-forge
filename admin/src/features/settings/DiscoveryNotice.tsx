import { AlertCircle, Check, Info } from "lucide-react";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import type { DiscoveryOutcome, DiscoveryTone } from "./model/discoveryOutcome";

const DISCOVERY_ICONS = { ok: Check, warn: Info, danger: AlertCircle } satisfies Record<DiscoveryTone, unknown>;

interface DiscoveryNoticeProps {
  outcome: DiscoveryOutcome;
  onUseEndpoint: (endpoint: string) => void;
}

/** What Load models found, with the base URL the server suggests when it has one. */
export function DiscoveryNotice({ outcome, onUseEndpoint }: DiscoveryNoticeProps) {
  const suggestion = outcome.suggestedEndpoint;
  return (
    <Notice
      tone={outcome.tone}
      icon={DISCOVERY_ICONS[outcome.tone]}
      action={
        suggestion ? (
          <Button type="button" variant="outline" size="sm" onClick={() => onUseEndpoint(suggestion)}>
            Use {new URL(suggestion).hostname}
          </Button>
        ) : null
      }
    >
      {outcome.message}
    </Notice>
  );
}
