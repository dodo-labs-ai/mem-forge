import type { ComponentType, ReactNode } from "react";
import { cn } from "@/lib/cn";
import { toneSurface, type Tone } from "./tone";

interface NoticeProps {
  tone: Tone;
  icon: ComponentType<{ className?: string }>;
  /** A short first line in bold; the body then reads as its explanation. */
  title?: ReactNode;
  children?: ReactNode;
  /** One control at the end, such as a link to what the notice is about. */
  action?: ReactNode;
  className?: string;
}

/** A tinted line inside the page that says what state something is in and what it means. */
export function Notice({ tone, icon: Icon, title, children, action, className }: NoticeProps) {
  return (
    <div role="status" className={cn("flex items-start gap-2.5 rounded-lg px-3.5 py-3 text-sm", toneSurface[tone], className)}>
      <Icon aria-hidden className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0 flex-1 space-y-1">
        {title ? <p className="font-semibold">{title}</p> : null}
        {children ? <div className={cn("space-y-1.5", title ? "text-subtle-foreground" : undefined)}>{children}</div> : null}
      </div>
      {action ? <div className="-my-1 shrink-0">{action}</div> : null}
    </div>
  );
}
