import { Search } from "lucide-react";
import { useEffect, useEffectEvent, useState } from "react";
import { cn } from "@/lib/cn";
import { Input } from "@/ui/input";

/** Delay before typed search text is committed, so a list is not read on every keystroke. */
const SEARCH_DEBOUNCE_MS = 300;

interface SearchBoxProps {
  /** The committed search text, such as the page's URL parameter. */
  value: string;
  /** Receives the typed text once the user pauses typing. */
  onCommit: (value: string) => void;
  /** The accessible name and placeholder, such as "Search memories". */
  label: string;
  className?: string;
}

/**
 * A search field that commits its text once the user pauses typing. A new
 * `value` from outside, such as a link that opens the page with a search,
 * replaces the typed text.
 */
export function SearchBox({ value, onCommit, label, className }: SearchBoxProps) {
  const [text, setText] = useState(value);
  const [committed, setCommitted] = useState(value);
  if (value !== committed) {
    setCommitted(value);
    setText(value);
  }
  const commit = useEffectEvent((next: string) => onCommit(next));
  useEffect(() => {
    if (text === committed) return;
    const timer = setTimeout(() => commit(text), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [text, committed]);

  return (
    <div className={cn("relative w-full max-w-sm", className)}>
      <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        type="search"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder={label}
        aria-label={label}
        className="pl-8"
      />
    </div>
  );
}
