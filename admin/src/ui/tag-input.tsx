import { XIcon } from "lucide-react";
import { useState, type ClipboardEvent, type FocusEvent, type KeyboardEvent } from "react";
import { cn } from "@/lib/cn";

type TagInputProps = {
  id?: string;
  value: readonly string[];
  onValueChange: (value: string[]) => void;
  onBlur?: (event: FocusEvent<HTMLInputElement>) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  "aria-describedby"?: string;
  "aria-invalid"?: boolean;
} & ({ "aria-label"?: string } | { "aria-labelledby"?: string });

/** The non-blank, trimmed items of a comma or newline separated string. */
function splitItems(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/**
 * A list of short strings shown as chips. Enter or a comma adds what was typed,
 * a pasted comma or newline list adds one chip per item, and Backspace in the
 * empty input removes the last chip. Leaving the input adds what is left in it.
 */
export function TagInput({
  id,
  value,
  onValueChange,
  onBlur,
  placeholder,
  disabled,
  className,
  ...aria
}: TagInputProps) {
  const [draft, setDraft] = useState("");

  function add(items: string[]) {
    const next = [...value];
    for (const item of items) if (!next.includes(item)) next.push(item);
    if (next.length !== value.length) onValueChange(next);
    setDraft("");
  }

  function remove(index: number) {
    onValueChange(value.filter((_, at) => at !== index));
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" || event.key === ",") {
      if (!draft.trim()) {
        if (event.key === ",") event.preventDefault();
        return;
      }
      event.preventDefault();
      add(splitItems(draft));
    } else if (event.key === "Backspace" && draft === "" && value.length > 0) {
      event.preventDefault();
      remove(value.length - 1);
    }
  }

  function onPaste(event: ClipboardEvent<HTMLInputElement>) {
    const text = event.clipboardData.getData("text");
    if (!/[,\n]/.test(text)) return;
    event.preventDefault();
    add(splitItems(draft + text));
  }

  return (
    <div
      data-slot="tag-input"
      className={cn(
        "flex min-h-8 w-full flex-wrap items-center gap-1 rounded-lg border border-input bg-transparent px-1.5 py-1 transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50 has-aria-invalid:border-destructive has-aria-invalid:ring-3 has-aria-invalid:ring-destructive/20 has-disabled:cursor-not-allowed has-disabled:opacity-50 dark:bg-input/30",
        className,
      )}
    >
      {value.map((item, index) => (
        <span
          key={item}
          className="inline-flex h-6 max-w-full items-center gap-1 rounded-md bg-muted pr-0.5 pl-2 text-sm text-foreground"
        >
          <span className="truncate">{item}</span>
          <button
            type="button"
            disabled={disabled}
            aria-label={`Remove ${item}`}
            onClick={() => remove(index)}
            className="inline-flex size-5 items-center justify-center rounded-sm text-muted-foreground outline-none hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <XIcon className="size-3.5" aria-hidden />
          </button>
        </span>
      ))}
      <input
        id={id}
        {...aria}
        value={draft}
        disabled={disabled}
        placeholder={value.length === 0 ? placeholder : undefined}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        onPaste={onPaste}
        onBlur={(event) => {
          if (draft.trim()) add(splitItems(draft));
          onBlur?.(event);
        }}
        className="h-6 min-w-24 flex-1 bg-transparent px-1 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed md:text-sm"
      />
    </div>
  );
}
