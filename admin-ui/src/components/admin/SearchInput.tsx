import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export function SearchInput({
  value,
  onChange,
  placeholder,
  ariaLabel,
  size = "default",
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  ariaLabel?: string;
  size?: "default" | "sm";
  className?: string;
}) {
  const compact = size === "sm";

  return (
    <div className={cn("relative min-w-0 flex-1", className)}>
      <Search
        className={cn(
          "pointer-events-none absolute top-1/2 -translate-y-1/2 text-muted-foreground",
          compact ? "left-2.5 size-3.5" : "left-3 size-4",
        )}
      />
      <Input
        type="search"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={cn(
          compact
            ? "h-7 rounded-[min(var(--radius-md),12px)] pl-8 pr-2.5 text-[0.8rem]"
            : "h-8 pl-9",
        )}
        placeholder={placeholder}
        aria-label={ariaLabel ?? placeholder}
      />
    </div>
  );
}
