import { Check, Lock, Users } from "lucide-react";

import type { SourceAccessPolicy } from "@/api/types";
import { cn } from "@/lib/utils";

export function SourceAccessSelection({
  value,
  onChange,
}: {
  value: SourceAccessPolicy | null;
  onChange: (value: SourceAccessPolicy) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2" role="radiogroup" aria-label="Who can use this source">
      <AccessChoice
        value="private"
        selected={value === "private"}
        icon={<Lock className="size-5" />}
        title="Only me"
        description="Only you can find this source or query memories from it."
        onSelect={onChange}
      />
      <AccessChoice
        value="workspace"
        selected={value === "workspace"}
        icon={<Users className="size-5" />}
        title="Everyone in this workspace"
        description="Workspace members can find the source and query its memories."
        onSelect={onChange}
      />
    </div>
  );
}

export function SourceAccessSummary({ policy }: { policy: SourceAccessPolicy }) {
  const isPrivate = policy === "private";
  return (
    <div className="flex items-start gap-3 rounded-lg border bg-background p-3">
      <span className={cn(
        "grid size-9 shrink-0 place-items-center rounded-full",
        isPrivate ? "bg-muted text-foreground" : "bg-sky-50 text-sky-700 dark:bg-sky-950/50 dark:text-sky-200",
      )}>
        {isPrivate ? <Lock className="size-4" /> : <Users className="size-4" />}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold">
          {isPrivate ? "Only me" : "Everyone in this workspace"}
        </span>
        <span className="mt-1 block text-xs text-muted-foreground">
          {isPrivate
            ? "Only the source owner can find it or query its memories."
            : "Workspace members can find the source and query its memories."}
        </span>
      </span>
    </div>
  );
}

function AccessChoice({
  value,
  selected,
  icon,
  title,
  description,
  onSelect,
}: {
  value: SourceAccessPolicy;
  selected: boolean;
  icon: React.ReactNode;
  title: string;
  description: string;
  onSelect: (value: SourceAccessPolicy) => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      className={cn(
        "relative flex min-h-28 items-start gap-3 rounded-xl border p-4 text-left transition-colors",
        "hover:border-foreground/30 hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected && "border-foreground bg-muted/40 ring-1 ring-foreground",
      )}
      onClick={() => onSelect(value)}
    >
      <span className="mt-0.5 text-muted-foreground">{icon}</span>
      <span className="min-w-0 pr-5">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="mt-1.5 block text-xs leading-relaxed text-muted-foreground">{description}</span>
      </span>
      {selected && (
        <span className="absolute right-3 top-3 grid size-5 place-items-center rounded-full bg-foreground text-background">
          <Check className="size-3" />
        </span>
      )}
    </button>
  );
}
