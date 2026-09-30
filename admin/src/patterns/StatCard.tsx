import { useRender } from "@base-ui/react/use-render";
import { ChevronRight } from "lucide-react";
import type { ReactElement, ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Skeleton } from "@/ui/skeleton";
import { toneText, type Tone } from "./tone";

interface StatCardLink {
  label: string;
  /** The link element, such as the router's `<Link to="/review" />`; the card fills in its content. */
  render: ReactElement;
}

interface StatCardProps {
  label: string;
  /** The headline number, already formatted. Undefined while it loads. */
  value: ReactNode | undefined;
  /** What the number counts, or what else is worth knowing about it. */
  detail?: ReactNode;
  /** Colors the value when it asks for attention. */
  tone?: Tone;
  /** Where the number leads, such as the list it counts. The whole card follows the link. */
  link?: StatCardLink;
  className?: string;
}

function CardLink({ label, render }: StatCardLink) {
  return useRender({
    defaultTagName: "a",
    render,
    props: {
      className:
        "inline-flex shrink-0 items-center gap-1 font-medium text-subtle-foreground outline-none after:absolute after:inset-0 after:rounded-lg focus-visible:after:ring-2 focus-visible:after:ring-ring",
      children: (
        <>
          {label}
          <ChevronRight aria-hidden className="size-3" />
        </>
      ),
    },
  });
}

/** One headline number with what it counts. */
export function StatCard({ label, value, detail, tone, link, className }: StatCardProps) {
  return (
    <div
      className={cn(
        "relative flex min-w-0 flex-col gap-1.5 rounded-lg border bg-surface px-4 py-3.5",
        link && "transition-colors hover:bg-surface-subtle",
        className,
      )}
    >
      <span className="text-sm text-muted-foreground">{label}</span>
      {value === undefined ? (
        <Skeleton className="h-8 w-20" />
      ) : (
        <span
          className={cn("font-mono text-2xl font-medium tracking-tight tabular-nums", tone ? toneText[tone] : "text-foreground")}
        >
          {value}
        </span>
      )}
      {detail || link ? (
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="min-w-0 flex-1">{detail}</span>
          {link ? <CardLink {...link} /> : null}
        </span>
      ) : null}
    </div>
  );
}
