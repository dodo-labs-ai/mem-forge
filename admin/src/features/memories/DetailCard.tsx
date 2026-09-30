import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

interface DetailCardProps {
  title: string;
  /** Right-aligned summary, such as a count. */
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** One titled section of the memory detail page. */
export function DetailCard({ title, meta, children, className }: DetailCardProps) {
  return (
    <section aria-label={title} className={cn("flex flex-col gap-3 rounded-lg border bg-surface p-4", className)}>
      <div className="flex items-baseline gap-2">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        {meta ? <span className="ml-auto text-xs text-muted-foreground tabular-nums">{meta}</span> : null}
      </div>
      {children}
    </section>
  );
}
