import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Brain,
  ChevronsUpDown,
  Database,
  Files,
  FolderKanban,
  Settings,
  ShieldCheck,
  X,
} from "lucide-react";
import client from "@/api/client";
import type { MemoryReviewListResponse } from "@/api/types";
import { BRAND_INITIALS, BRAND_NAME, BRAND_SUBTITLE } from "@/brand";
import { TaiSealLogo } from "@/components/brand/TaiSealLogo";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ACTIVE_WORKSPACE_NAME } from "@/lib/workspace";

const navGroups = [
  {
    label: "General",
    items: [
      { to: "/memories", label: "Memories", icon: Brain },
      { to: "/review", label: "Review", icon: ShieldCheck, badgeKey: "pending-reviews" as const },
      { to: "/entities", label: "Entities", icon: Database },
      { to: "/sources", label: "Sources", icon: Files },
      { to: "/projects", label: "Project setup", icon: FolderKanban },
    ],
  },
  {
    label: "Other",
    items: [{ to: "/settings", label: "Settings", icon: Settings }],
  },
];

const PENDING_REVIEW_POLL_MS = 30_000;
const PENDING_REVIEW_BADGE_CAP = 99;

function formatBadgeCount(count: number): string {
  if (count <= 0) return "";
  return count > PENDING_REVIEW_BADGE_CAP ? `${PENDING_REVIEW_BADGE_CAP}+` : String(count);
}

function NavBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <span className="ml-auto inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-full bg-muted px-1.5 text-[11px] font-medium text-muted-foreground">
      {formatBadgeCount(count)}
    </span>
  );
}

function usePendingReviewCount() {
  return useQuery<MemoryReviewListResponse>({
    queryKey: ["pending-reviews"],
    queryFn: () =>
      client
        .get("/api/memory-reviews", { params: { status: "open", limit: 1 } })
        .then((res) => res.data),
    refetchInterval: PENDING_REVIEW_POLL_MS,
    staleTime: PENDING_REVIEW_POLL_MS,
  });
}

function Brand({ onClose }: { onClose?: () => void }) {
  return (
    <div className="flex h-16 items-center gap-3 px-3">
      <div className="grid size-9 place-items-center rounded-lg bg-background ring-1 ring-sidebar-border">
        <TaiSealLogo className="size-8" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-semibold leading-none">{BRAND_NAME}</div>
        <div className="text-xs text-muted-foreground">{BRAND_SUBTITLE}</div>
      </div>
      {onClose && (
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="lg:hidden"
          onClick={onClose}
          aria-label="Close navigation"
        >
          <X className="size-4" />
        </Button>
      )}
    </div>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const pendingReviews = usePendingReviewCount();
  const pendingCount = pendingReviews.data?.total ?? 0;

  return (
    <nav className="flex-1 space-y-5 overflow-y-auto px-2 py-2">
      {navGroups.map((group) => (
        <div key={group.label} className="space-y-1">
          <div className="px-2 pb-1 text-[11px] font-medium text-muted-foreground">
            {group.label}
          </div>
          {group.items.map((item) => {
            const Icon = item.icon;
            const badge = item.badgeKey === "pending-reviews" ? pendingCount : 0;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    "flex h-8 items-center gap-2 rounded-md px-2 text-sm transition-colors",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground/75 hover:bg-sidebar-accent/70 hover:text-sidebar-foreground"
                  )
                }
              >
                <Icon className="size-4 text-current opacity-80" />
                <span className="truncate">{item.label}</span>
                <NavBadge count={badge} />
              </NavLink>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

function AccountFooter() {
  return (
    <div className="p-2">
      <button
        type="button"
        className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors hover:bg-sidebar-accent"
      >
        <span className="grid size-8 shrink-0 place-items-center rounded-md bg-background text-xs font-medium ring-1 ring-sidebar-border">
          {BRAND_INITIALS}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium leading-none">{BRAND_NAME}</span>
          <span className="mt-1 block truncate text-xs text-muted-foreground">{ACTIVE_WORKSPACE_NAME}</span>
        </span>
        <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground" />
      </button>
    </div>
  );
}

export function Sidebar({
  mobileOpen = false,
  onMobileOpenChange,
}: {
  mobileOpen?: boolean;
  onMobileOpenChange?: (open: boolean) => void;
}) {
  return (
    <>
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 border-r border-sidebar-border bg-sidebar text-sidebar-foreground lg:flex lg:flex-col">
        <Brand />
        <NavItems />
        <AccountFooter />
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            aria-label="Close navigation overlay"
            onClick={() => onMobileOpenChange?.(false)}
          />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label="Navigation"
            className="relative flex h-full w-72 max-w-[85vw] flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground shadow-lg"
          >
            <Brand onClose={() => onMobileOpenChange?.(false)} />
            <NavItems onNavigate={() => onMobileOpenChange?.(false)} />
            <AccountFooter />
          </aside>
        </div>
      )}
    </>
  );
}
