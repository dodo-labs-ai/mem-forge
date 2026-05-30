import { Search } from "lucide-react";

export function CommandSearch() {
  return (
    <button
      type="button"
      className="flex h-8 w-full items-center gap-2 rounded-md border bg-background px-3 text-sm text-muted-foreground shadow-xs transition-colors hover:bg-accent hover:text-accent-foreground"
    >
      <Search className="size-4" />
      <span className="truncate">Search</span>
      <kbd className="ml-auto hidden rounded border bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground sm:inline-flex">
        ⌘K
      </kbd>
    </button>
  );
}
