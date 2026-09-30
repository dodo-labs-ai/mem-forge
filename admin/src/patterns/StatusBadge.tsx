import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { toneDot, toneSurface, type Tone } from "./tone";

interface StatusBadgeProps {
  tone: Tone;
  children: ReactNode;
  /** Shows a colored dot before the label instead of a tinted background. */
  variant?: "soft" | "dot";
  className?: string;
}

export function StatusBadge({ tone, children, variant = "soft", className }: StatusBadgeProps) {
  if (variant === "dot") {
    return (
      <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium text-subtle-foreground", className)}>
        <span aria-hidden className={cn("size-1.5 rounded-full", toneDot[tone])} />
        {children}
      </span>
    );
  }
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center gap-1 rounded-full px-2 text-xs font-medium whitespace-nowrap [&_svg]:size-3",
        toneSurface[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
