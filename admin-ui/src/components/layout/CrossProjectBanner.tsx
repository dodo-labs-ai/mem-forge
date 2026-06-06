import { AlertTriangle } from "lucide-react";

import { useActiveProject } from "@/state/activeProject";

const BANNER_MESSAGE =
  "Cross-project view active. Ranking ignores your project preference.";
const EXIT_LABEL_RESTORE = "Exit";
const EXIT_LABEL_PICK = "Pick a project";

/**
 * Surfaces the cross-project (admin) state on every page. Picks the right Exit
 * affordance based on whether a previously active project is remembered: if
 * one is, Exit restores it; otherwise it leaves the banner cleared and the
 * user is funneled to the topbar chip to pick a project. Renders nothing when
 * the user is not in cross-project mode, so callers can mount it
 * unconditionally.
 */
export function CrossProjectBanner() {
  const { crossProjectMode, lastActiveProjectKey, setActiveProjectKey } =
    useActiveProject();

  if (!crossProjectMode) return null;

  const exitLabel = lastActiveProjectKey
    ? EXIT_LABEL_RESTORE
    : EXIT_LABEL_PICK;

  return (
    <div
      role="status"
      className="flex items-center justify-between gap-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-100"
    >
      <span className="flex items-center gap-2">
        <AlertTriangle className="size-4 shrink-0" aria-hidden />
        <span>{BANNER_MESSAGE}</span>
      </span>
      <button
        type="button"
        className="font-medium underline underline-offset-2 hover:opacity-80"
        onClick={() => setActiveProjectKey(lastActiveProjectKey)}
      >
        {exitLabel}
      </button>
    </div>
  );
}
