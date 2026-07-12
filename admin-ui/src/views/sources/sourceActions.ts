export type SourceActionTone = "neutral" | "primary" | "destructive";

export interface SourceAction {
  id: "configure" | "sync" | "toggle-status" | "force-resync" | "delete";
  label: string;
  description?: string;
  tone: SourceActionTone;
  disabled?: boolean;
  requiresConfirmation?: boolean;
}

export const sourceActionLayout = {
  primary: {
    configure: { id: "configure", label: "Configure", tone: "neutral" },
    sync: { id: "sync", label: "Sync", tone: "primary" },
  },
  menu: [
    {
      id: "toggle-status",
      label: "Pause source",
      description:
        "Pause or resume source discovery without deleting configuration or extracted memories.",
      tone: "neutral",
    },
    {
      id: "force-resync",
      label: "Refresh source",
      description:
        "Look for new, changed, or removed documents. Existing memories are not rebuilt unless source content changed.",
      tone: "neutral",
    },
    {
      id: "delete",
      label: "Delete source",
      description: "Remove the source and retire memories left without support.",
      tone: "destructive",
      requiresConfirmation: true,
    },
  ],
} as const satisfies {
  primary: Readonly<Record<"configure" | "sync", SourceAction>>;
  menu: readonly SourceAction[];
};

export function getSourceActionEndpoint(sourceId: string, actionId: "force-resync" | "delete"): string {
  if (actionId === "force-resync") return `/sources/${sourceId}/force-resync`;
  return `/sources/${sourceId}`;
}

export function getSourceMenuPlacement({
  triggerTop,
  triggerBottom,
  viewportHeight,
  menuHeight,
  gap = 8,
}: {
  triggerTop: number;
  triggerBottom: number;
  viewportHeight: number;
  menuHeight: number;
  gap?: number;
}): { direction: "up" | "down"; top: number } {
  const spaceBelow = viewportHeight - triggerBottom;
  if (spaceBelow < menuHeight + gap && triggerTop > spaceBelow) {
    return { direction: "up", top: Math.max(gap, triggerTop - menuHeight - gap) };
  }
  return { direction: "down", top: Math.min(triggerBottom + gap, viewportHeight - menuHeight - gap) };
}

export function getSourceMenuStyle({
  triggerRight,
  triggerTop,
  triggerBottom,
  viewportWidth,
  viewportHeight,
  menuHeight,
  menuWidth = 288,
  gap = 8,
}: {
  triggerRight: number;
  triggerTop: number;
  triggerBottom: number;
  viewportWidth: number;
  viewportHeight: number;
  menuHeight: number;
  menuWidth?: number;
  gap?: number;
}): { position: "fixed"; top: number; left: number; width: number } {
  const placement = getSourceMenuPlacement({
    triggerTop,
    triggerBottom,
    viewportHeight,
    menuHeight,
    gap,
  });
  const maxLeft = viewportWidth - menuWidth - gap;
  const left = Math.min(Math.max(gap, triggerRight - menuWidth), Math.max(gap, maxLeft));
  return {
    position: "fixed",
    top: placement.top,
    left,
    width: menuWidth,
  };
}
