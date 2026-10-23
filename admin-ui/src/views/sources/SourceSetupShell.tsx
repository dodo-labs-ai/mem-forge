import type { ReactNode } from "react";
import { AlertCircle, Check, ChevronDown, Info, Loader2, RefreshCw } from "lucide-react";

import { SourceIcon } from "@/components/sources/SourceIcon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import type { ScopeChangeWarning } from "./scopeChange";
import type { SourceConnectionMode } from "./sourceConnectionPresentation";

export type SourceSetupSectionId =
  | "basics"
  | "connection"
  | "content"
  | "access"
  | "project"
  | "schedule";

export interface SourceSetupSection {
  id: SourceSetupSectionId;
  title: string;
  summary: string;
  state: "complete" | "incomplete" | "attention";
  content: ReactNode;
}

/**
 * Saving while a sync of the Source runs. The server refuses the change until
 * the sync has ended, so a save asks first: stop the sync and save, or wait.
 */
export interface RunningSyncSave {
  /** A sync of the Source is running; the dialog says that saving stops it. */
  running: boolean;
  /** A save is waiting for the person's choice. */
  confirming: boolean;
  stopping: boolean;
  error: string | null;
  onStopAndSave: () => void;
  /** Keeps the dialog open with its edits and saves nothing, also while the sync is stopping. */
  onWait: () => void;
}

export function SourceSetupShell({
  sourceType,
  sourceLabel,
  sourceName,
  connection,
  isEdit,
  sections,
  openSection,
  onOpenSectionChange,
  error,
  runningSync,
  scopeChange,
  validationMessage,
  saving,
  saveDisabled,
  onCancel,
  onSave,
}: {
  sourceType: string;
  sourceLabel: string;
  sourceName: string;
  connection: { mode: SourceConnectionMode; label: string };
  isEdit: boolean;
  sections: SourceSetupSection[];
  openSection: SourceSetupSectionId;
  onOpenSectionChange: (section: SourceSetupSectionId) => void;
  error?: string | null;
  runningSync?: RunningSyncSave | null;
  /** What saving would do to what the source syncs, when the form should say so before saving. */
  scopeChange?: ScopeChangeWarning | null;
  validationMessage?: string | null;
  saving: boolean;
  saveDisabled?: boolean;
  onCancel: () => void;
  onSave: () => void;
}) {
  const completed = sections.filter((section) => section.state === "complete").length;
  const progress = Math.round((completed / sections.length) * 100);

  return (
    <>
      <DialogHeader className="shrink-0 border-b px-5 py-4">
        <div className="flex items-start gap-3 pr-8">
          <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-muted">
            <SourceIcon type={sourceType} className="size-6" />
          </div>
          <div className="min-w-0">
            <DialogTitle>{isEdit ? `Configure ${sourceLabel}` : `Add ${sourceLabel} source`}</DialogTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              Connection, content, access, destination, and sync frequency
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <ConnectionBadge mode={connection.mode}>{connection.label}</ConnectionBadge>
              <Badge variant="outline" className="text-[11px]">
                {isEdit ? "Editing existing source" : "New source"}
              </Badge>
            </div>
          </div>
        </div>
      </DialogHeader>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        <div className="mb-3 flex items-center justify-between gap-5 rounded-xl border bg-muted/20 p-4">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {sourceLabel}
            </div>
            <div className="mt-1 truncate text-base font-semibold">
              {sourceName.trim() || "New source"}
            </div>
          </div>
          <div className="w-44 shrink-0">
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-foreground transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="mt-1.5 text-right text-[11px] text-muted-foreground">
              {completed === sections.length
                ? "All settings complete"
                : `${completed} of ${sections.length} sections complete`}
            </div>
          </div>
        </div>

        {runningSync?.running && (
          <div role="status" className="mb-3 flex items-start gap-2 rounded-lg bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-900/20 dark:text-amber-100">
            <RefreshCw className="mt-0.5 size-4 shrink-0" />
            <span>A sync is running. Saving will stop it first.</span>
          </div>
        )}

        <div className="space-y-2">
          {sections.map((section, index) => {
            const open = section.id === openSection;
            return (
              <section key={section.id} className="overflow-hidden rounded-xl border bg-background">
                <button
                  type="button"
                  className="flex w-full items-center gap-3 p-3.5 text-left hover:bg-muted/30"
                  aria-expanded={open}
                  onClick={() => onOpenSectionChange(section.id)}
                >
                  <SectionState state={section.state} index={index + 1} />
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold">{index + 1}. {section.title}</span>
                    <span className="mt-0.5 block truncate text-xs text-muted-foreground">{section.summary}</span>
                  </span>
                  <ChevronDown className={`size-4 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} />
                </button>
                {open && <div className="space-y-4 border-t bg-muted/10 p-4">{section.content}</div>}
              </section>
            );
          })}
        </div>

        {scopeChange && (
          <div
            role={scopeChange.tone === "warning" ? "alert" : "status"}
            className={`mt-3 flex items-start gap-2 rounded-lg p-3 text-sm ${
              scopeChange.tone === "warning"
                ? "bg-amber-50 text-amber-900 dark:bg-amber-900/20 dark:text-amber-100"
                : "bg-sky-50 text-sky-900 dark:bg-sky-900/20 dark:text-sky-100"
            }`}
          >
            {scopeChange.tone === "warning"
              ? <AlertCircle className="mt-0.5 size-4 shrink-0" />
              : <Info className="mt-0.5 size-4 shrink-0" />}
            <span>
              <span className="block font-medium">{scopeChange.title}</span>
              <span className="mt-0.5 block">{scopeChange.message}</span>
            </span>
          </div>
        )}

        {(error || validationMessage) && (
          <div className={`mt-3 flex items-start gap-2 rounded-lg p-3 text-sm ${error ? "bg-destructive/10 text-destructive" : "bg-muted/50 text-muted-foreground"}`}>
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>{error || validationMessage}</span>
          </div>
        )}
      </div>

      <DialogFooter className="mx-0 mb-0 shrink-0 flex-row justify-between rounded-none rounded-b-xl border-t bg-background px-5 py-4 sm:justify-between">
        <Button type="button" variant="outline" onClick={onCancel}>Cancel</Button>
        <Button type="button" onClick={onSave} disabled={saving || saveDisabled || runningSync?.stopping}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
          {isEdit ? "Save Changes" : "Create Source"}
        </Button>
      </DialogFooter>

      {runningSync && <RunningSyncConfirmation runningSync={runningSync} />}
    </>
  );
}

function RunningSyncConfirmation({ runningSync }: { runningSync: RunningSyncSave }) {
  return (
    <Dialog
      open={runningSync.confirming}
      onOpenChange={(open) => {
        if (!open) runningSync.onWait();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Stop the sync and save?</DialogTitle>
          <DialogDescription>
            A sync of this source is running, and a change can be saved only once it has ended. Stopping the sync
            keeps what it has saved, and the next sync uses your new settings.
          </DialogDescription>
        </DialogHeader>
        {runningSync.stopping && (
          <p className="text-sm text-muted-foreground">
            The sync usually stops within a minute. Waiting instead keeps your edits unsaved, and the sync still
            stops.
          </p>
        )}
        {runningSync.error && (
          <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>{runningSync.error}</span>
          </div>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={runningSync.onWait}>
            Wait for the sync to finish
          </Button>
          <Button type="button" disabled={runningSync.stopping} onClick={runningSync.onStopAndSave}>
            {runningSync.stopping && <Loader2 className="size-4 animate-spin" />}
            {runningSync.stopping ? "Stopping the sync…" : "Stop sync and save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function SectionState({ state, index }: { state: SourceSetupSection["state"]; index: number }) {
  if (state === "complete") {
    return <span className="grid size-7 shrink-0 place-items-center rounded-full bg-emerald-50 text-emerald-700"><Check className="size-3.5" /></span>;
  }
  if (state === "attention") {
    return <span className="grid size-7 shrink-0 place-items-center rounded-full bg-amber-50 text-xs font-semibold text-amber-700">!</span>;
  }
  return <span className="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">{index}</span>;
}

function ConnectionBadge({ mode, children }: { mode: SourceConnectionMode; children: ReactNode }) {
  const tone = mode === "device"
    ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200"
    : mode === "choice"
      ? "border-violet-200 bg-violet-50 text-violet-700 dark:border-violet-800 dark:bg-violet-950/50 dark:text-violet-200"
      : "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-800 dark:bg-sky-950/50 dark:text-sky-200";
  return <Badge variant="outline" className={`text-[11px] ${tone}`}>{children}</Badge>;
}
