import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { toneDot, type Tone } from "./tone";

interface ActionCardProps {
  tone: Tone;
  title: string;
  description: ReactNode;
  action?: ReactNode;
  className?: string;
}

/** Something that needs the user, with the one action that resolves it. */
export function ActionCard({ tone, title, description, action, className }: ActionCardProps) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-2 rounded-lg border bg-surface p-3", className)}>
      <div className="flex items-center gap-2">
        <span aria-hidden className={cn("size-2 shrink-0 rounded-full", toneDot[tone])} />
        <p className="truncate font-medium text-foreground">{title}</p>
      </div>
      <p className="text-sm text-muted-foreground">{description}</p>
      {action ? <div className="mt-auto pt-1">{action}</div> : null}
    </div>
  );
}
