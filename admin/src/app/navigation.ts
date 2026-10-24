import { Activity, Brain, Files, KanbanSquare, Settings, ShieldCheck, Tags, type LucideIcon } from "lucide-react";
import { useOpenReviewCount } from "@/api";
import type { ReservedRouteSegment } from "./extension/contract";

export interface NavItem {
  segment: ReservedRouteSegment;
  label: string;
  icon: LucideIcon;
  /** A hook that reads the count the item shows beside its label; nothing shows while it is null or zero. */
  useBadge?: () => number | null;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    title: "Knowledge",
    items: [
      { segment: "memories", label: "Memories", icon: Brain },
      { segment: "entities", label: "Entities", icon: Tags },
    ],
  },
  {
    title: "Curate",
    items: [
      { segment: "review", label: "Review", icon: ShieldCheck, useBadge: useOpenReviewCount },
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
