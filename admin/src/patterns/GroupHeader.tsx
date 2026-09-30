import type { ReactNode } from "react";

interface GroupHeaderProps {
  title: string;
  description?: ReactNode;
  /** Right-aligned summary, such as a count. */
  meta?: ReactNode;
}

/** The label of one group of rows in a DataTable. */
export function GroupHeader({ title, description, meta }: GroupHeaderProps) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="font-semibold text-foreground">{title}</span>
      {description ? <span className="truncate text-xs text-muted-foreground">{description}</span> : null}
      {meta ? <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">{meta}</span> : null}
    </div>
  );
}
