import { Activity, Brain, Files, KanbanSquare, Settings, ShieldCheck, type LucideIcon } from "lucide-react";
import type { ReservedRouteSegment } from "./extension/contract";

export interface NavItem {
  segment: ReservedRouteSegment;
  label: string;
  icon: LucideIcon;
  /**
   * Set while the page is not built in this app yet: the item opens the V1
   * page at this path instead. Remove it when the page lands (ADR 0044).
   */
  v1Path?: string;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  { title: "Knowledge", items: [{ segment: "memories", label: "Memories", icon: Brain, v1Path: "/memories" }] },
  {
    title: "Curate",
    items: [
      { segment: "review", label: "Review", icon: ShieldCheck, v1Path: "/review" },
      { segment: "evaluation", label: "Evaluation", icon: Activity, v1Path: "/evaluation" },
    ],
  },
  {
    title: "Connect",
    items: [
      { segment: "sources", label: "Sources", icon: Files },
      { segment: "projects", label: "Projects", icon: KanbanSquare, v1Path: "/projects" },
    ],
  },
];

export const SETTINGS_NAV_ITEM: NavItem = { segment: "settings", label: "Settings", icon: Settings, v1Path: "/settings" };

/** The page the app opens on. */
export const HOME_SEGMENT: ReservedRouteSegment = "sources";
