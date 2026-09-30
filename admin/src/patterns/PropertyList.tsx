import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface Property {
  label: string;
  value: ReactNode;
  /** Secondary line under the value. */
  hint?: ReactNode;
}

interface PropertyListProps {
  items: Property[];
  /** `row` lays properties side by side, `stack` puts one per line. */
  layout?: "row" | "stack";
  className?: string;
}

/** Labeled properties. Use this instead of joining values into one string. */
export function PropertyList({ items, layout = "stack", className }: PropertyListProps) {
  return (
    <dl
      className={cn(
        layout === "row" ? "flex flex-wrap gap-x-8 gap-y-3" : "grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2.5",
        className,
      )}
    >
      {items.map((item) =>
        layout === "row" ? (
          <div key={item.label} className="min-w-0 space-y-0.5">
            <dt className="text-xs text-muted-foreground">{item.label}</dt>
            <dd className="text-sm text-foreground">{item.value}</dd>
            {item.hint ? <dd className="text-xs text-muted-foreground">{item.hint}</dd> : null}
          </div>
        ) : (
          <div key={item.label} className="contents">
            <dt className="text-sm text-muted-foreground">{item.label}</dt>
            <dd className="min-w-0 text-sm text-foreground">
              {item.value}
              {item.hint ? <div className="text-xs text-muted-foreground">{item.hint}</div> : null}
            </dd>
          </div>
        ),
      )}
    </dl>
  );
}
