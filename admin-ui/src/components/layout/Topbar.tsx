import { Bell, ChevronsUpDown, Circle, Menu, Sun } from "lucide-react";
import { BRAND_INITIALS } from "@/brand";
import { Button } from "@/components/ui/button";
import { ACTIVE_WORKSPACE_NAME } from "@/lib/workspace";
import { ActiveProjectChip } from "./ActiveProjectChip";
import { CommandSearch } from "./CommandSearch";

const WORKSPACE_SWITCH_HINT = "Workspace switching arrives with team support.";

export function Topbar({ onOpenNavigation }: { onOpenNavigation: () => void }) {
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
        <div className="grid size-8 place-items-center rounded-md bg-muted text-xs font-medium">
          {BRAND_INITIALS}
        </div>
      </div>
    </header>
  );
}
