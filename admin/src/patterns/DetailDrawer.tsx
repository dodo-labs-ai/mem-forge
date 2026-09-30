import type { ReactNode } from "react";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/ui/sheet";

interface DetailDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  /** Short line under the title, such as the item's type. */
  subtitle?: ReactNode;
  /** Leading visual next to the title, such as an icon. */
  leading?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}

/** The side panel that shows one item of a list without leaving the page. */
export function DetailDrawer({ open, onOpenChange, title, subtitle, leading, actions, children }: DetailDrawerProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full gap-0 p-0 sm:max-w-xl">
        <SheetHeader className="gap-3 border-b p-5 pr-12">
          <div className="flex items-start gap-3">
            {leading}
            <div className="min-w-0 space-y-0.5">
              <SheetTitle className="truncate text-base">{title}</SheetTitle>
              {subtitle ? <SheetDescription>{subtitle}</SheetDescription> : null}
            </div>
          </div>
          {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
        </SheetHeader>
        <div className="flex-1 space-y-6 overflow-y-auto p-5">{children}</div>
      </SheetContent>
    </Sheet>
  );
}

interface DrawerSectionProps {
  title: string;
  children: ReactNode;
}

export function DrawerSection({ title, children }: DrawerSectionProps) {
  return (
    <section className="space-y-3">
      <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">{title}</h3>
      {children}
    </section>
  );
}
