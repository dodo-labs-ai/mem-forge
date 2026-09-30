import { ChevronDown } from "lucide-react";
import { useState, type ComponentType } from "react";
import { cn } from "@/lib/cn";
import { pluralize } from "@/lib/format";
import { EmptyState, StatusBadge } from "@/patterns";
import { buildInvestigationPrompt } from "./model/investigationPrompt";
import { checkName, issueGroupSummary, reasonDescription, RESULT_LABELS } from "./model/presentation";
import type { EvaluationCase, IssueGroup } from "./model/types";
import { RepresentativeCase, type CopyState } from "./RepresentativeCase";

interface IssueGroupListProps {
  groups: IssueGroup[];
  sourceNames: ReadonlyMap<string, string>;
  typeLabels: Readonly<Record<string, string>>;
  empty: { icon: ComponentType<{ className?: string }>; title: string; description: string };
}

export function IssueGroupList({ groups, sourceNames, typeLabels, empty }: IssueGroupListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [copy, setCopy] = useState<{ eventId: string; state: CopyState } | null>(null);

  if (groups.length === 0) return <EmptyState icon={empty.icon} title={empty.title} description={empty.description} />;

  async function copyPrompt(eventId: string, prompt: string) {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopy({ eventId, state: "copied" });
    } catch {
      setCopy({ eventId, state: "failed" });
    }
  }

  function renderCase(group: IssueGroup, evaluationCase: EvaluationCase) {
    const sourceName = sourceNames.get(evaluationCase.source_id) ?? evaluationCase.source_id;
    const prompt = buildInvestigationPrompt(sourceName, group, evaluationCase);
    return (
      <RepresentativeCase
        key={evaluationCase.event_id}
        evaluationCase={evaluationCase}
        sourceName={sourceName}
        prompt={prompt}
        copyState={copy?.eventId === evaluationCase.event_id ? copy.state : undefined}
        onCopy={() => void copyPrompt(evaluationCase.event_id, prompt)}
      />
    );
  }

  return (
    <ul className="divide-y">
      {groups.map((group) => {
        const expanded = expandedId === group.group_id;
        const result = RESULT_LABELS[group.label];
        return (
          <li key={group.group_id} className={cn(expanded && "bg-surface-subtle")}>
            <button
              type="button"
              aria-expanded={expanded}
              onClick={() => setExpandedId(expanded ? null : group.group_id)}
              className="flex w-full items-start gap-3.5 px-4 py-3.5 text-left hover:bg-surface-subtle"
            >
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-foreground">{checkName(group.criterion)}</span>
                  <StatusBadge tone={result.tone}>{result.text}</StatusBadge>
                </div>
                <p className="text-sm text-subtle-foreground">{reasonDescription(group.reason_code)}</p>
                <p className="text-xs text-muted-foreground">{issueGroupSummary(group, typeLabels)}</p>
              </div>
              <span className="flex shrink-0 items-center gap-1.5 pt-0.5 text-xs whitespace-nowrap text-muted-foreground">
                {pluralize(group.representative_cases.length, "example")}
                <ChevronDown aria-hidden className={cn("size-4 transition-transform", expanded && "rotate-180")} />
              </span>
            </button>
            {expanded ? (
              <div className="space-y-2 px-4 pb-4">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Representative cases</h3>
                  <p className="text-xs text-muted-foreground sm:ml-auto">
                    Investigate copies a bounded, read-only diagnosis prompt with lineage and evaluator versions.
                  </p>
                </div>
                {group.representative_cases.map((evaluationCase) => renderCase(group, evaluationCase))}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
