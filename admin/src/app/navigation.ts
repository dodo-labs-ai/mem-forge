import { Activity, Brain, Files, KanbanSquare, Settings, ShieldCheck, type LucideIcon } from "lucide-react";
import type { ReservedRouteSegment } from "./extension/contract";

export interface NavItem {
  segment: ReservedRouteSegment;
  label: string;
  icon: LucideIcon;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  { title: "Knowledge", items: [{ segment: "memories", label: "Memories", icon: Brain }] },
  {
    title: "Curate",
    items: [
      { segment: "review", label: "Review", icon: ShieldCheck },
      { segment: "evaluation", label: "Evaluation", icon: Activity },
    ],
  },
  {
    title: "Connect",
    items: [
      { segment: "sources", label: "Sources", icon: Files },
      { segment: "projects", label: "Projects", icon: KanbanSquare },
    ],
  },
];

export const SETTINGS_NAV_ITEM: NavItem = { segment: "settings", label: "Settings", icon: Settings };

/** The page the app opens on. */
export const HOME_SEGMENT: ReservedRouteSegment = "sources";
