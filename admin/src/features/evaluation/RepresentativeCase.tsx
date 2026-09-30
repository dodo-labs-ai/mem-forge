import { Check, Clipboard, Copy } from "lucide-react";
import { Button } from "@/ui/button";
import { CaseIdentifiers } from "./CaseIdentifiers";
import type { EvaluationCase } from "./model/types";

export type CopyState = "copied" | "failed";

interface RepresentativeCaseProps {
  evaluationCase: EvaluationCase;
  sourceName: string;
  prompt: string;
  copyState: CopyState | undefined;
  onCopy: () => void;
}

/** One example of an issue group, with the prompt that hands it to a coding agent. */
export function RepresentativeCase({ evaluationCase, sourceName, prompt, copyState, onCopy }: RepresentativeCaseProps) {
  const copied = copyState === "copied";
  return (
    <div className="space-y-3 rounded-lg border bg-surface p-3.5">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-sm font-medium text-foreground">{sourceName}</p>
          <CaseIdentifiers evaluationCase={evaluationCase} />
        </div>
        <Button size="sm" variant="outline" onClick={onCopy} aria-label={`Investigate with agent: case from ${sourceName}`}>
          {copied ? <Check /> : <Clipboard />}
          {copied ? "Prompt copied" : "Investigate with agent"}
        </Button>
      </div>
      {copyState === "failed" ? (
        <p role="alert" className="text-xs text-tone-danger-foreground">
          Clipboard access failed. Try again from a secure browser window.
        </p>
      ) : null}
      {copied ? <CopiedPrompt prompt={prompt} onCopyAgain={onCopy} /> : null}
    </div>
  );
}

function CopiedPrompt({ prompt, onCopyAgain }: { prompt: string; onCopyAgain: () => void }) {
  return (
    <section aria-label="Copied prompt" className="space-y-2.5 rounded-lg border bg-surface-subtle p-3">
      <div className="flex items-center gap-2">
        <p className="text-sm font-semibold text-foreground">Prompt copied to the clipboard</p>
        <Button size="sm" variant="outline" className="ml-auto" onClick={onCopyAgain}>
          <Copy />
          Copy again
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Paste it into Codex or Claude Code. It carries IDs and versions only, no source text.
      </p>
      <pre className="max-h-72 overflow-auto rounded-md border bg-surface px-3 py-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap text-subtle-foreground">
        {prompt}
      </pre>
    </section>
  );
}
