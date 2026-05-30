import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function DataSurface({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("overflow-hidden rounded-xl border bg-card shadow-xs", className)}>
      {children}
    </div>
  );
}
