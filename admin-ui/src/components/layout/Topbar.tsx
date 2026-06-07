import { Bell, ChevronsUpDown, Circle, Menu, Sun } from "lucide-react";
import { BRAND_INITIALS, BRAND_NAME } from "@/brand";
import { Button } from "@/components/ui/button";
import { ACTIVE_WORKSPACE_NAME } from "@/lib/workspace";
import { getExtensionAccountSurface, getExtensionTopbarSlots } from "@/extension";
import { ActiveProjectChip } from "./ActiveProjectChip";
import { CommandSearch } from "./CommandSearch";

const WORKSPACE_SWITCH_HINT = "Workspace switching arrives with team support.";

function ExtensionSlots({ placement }: { placement: "before-account" }) {
  const slots = getExtensionTopbarSlots().filter(
    (slot) => (slot.placement ?? "before-account") === placement,
  );
  if (slots.length === 0) return null;
  return (
    <>
      {slots.map((slot) => (
        <span key={slot.id} className="contents">
          {slot.render()}
        </span>
      ))}
    </>
  );
}

/**
 * Default standalone account affordance. Decorative only: an extension that
 * owns identity will replace this with an interactive control via
 * `accountSurface.topbar`. Marking it `aria-hidden` and giving it a title
 * keeps the visual layout stable without advertising a menu the standalone
 * shell cannot open.
 */
function DefaultAccountBadge() {
  return (
    <div
      className="grid size-8 place-items-center rounded-md bg-muted text-xs font-medium"
      aria-hidden="true"
      title={BRAND_NAME}
    >
      {BRAND_INITIALS}
    </div>
  );
}

export function Topbar({ onOpenNavigation }: { onOpenNavigation: () => void }) {
  const accountSurface = getExtensionAccountSurface();
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background px-3 lg:px-4">
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        className="lg:hidden"
        onClick={onOpenNavigation}
        aria-label="Open navigation"
      >
        <Menu className="size-4" />
      </Button>

      <div className="ml-auto min-w-0 flex-1 md:max-w-sm">
        <CommandSearch />
      </div>

      <div className="flex items-center gap-1">
        <ActiveProjectChip />
        <span
          className="hidden items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground sm:inline-flex"
          title={WORKSPACE_SWITCH_HINT}
          aria-label={`Workspace: ${ACTIVE_WORKSPACE_NAME}`}
        >
          <span className="truncate">{ACTIVE_WORKSPACE_NAME}</span>
          <ChevronsUpDown className="size-3 opacity-70" aria-hidden="true" />
        </span>
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Theme">
          <Sun className="size-4" />
        </Button>
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Notifications">
          <Bell className="size-4" />
        </Button>
        <div className="hidden items-center gap-2 rounded-md px-2 py-1 text-xs text-muted-foreground sm:flex">
          <Circle className="size-2 fill-emerald-500 text-emerald-500" />
          <span>API</span>
        </div>
        <ExtensionSlots placement="before-account" />
        {accountSurface?.topbar ? accountSurface.topbar() : <DefaultAccountBadge />}
      </div>
    </header>
  );
}
