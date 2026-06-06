/**
 * Shared form for creating a project. Used by the dedicated Projects page
 * dialog and by the inline-create flow inside the Source dialog's project
 * step. Owns layout (Name + Advanced disclosure with Code + team-wide
 * checkbox) and reserved-key validation; the parent owns submission.
 */
import type { ChangeEvent } from "react";
import { Input } from "@/components/ui/input";
import {
  projectCreateKeyConflictsWithBuiltIn,
} from "./projectCreateForm";
import type { ProjectCreateFormState } from "./projectCreateForm";

export function ProjectCreateFields({
  form,
  onChange,
  advancedOpen,
  onAdvancedToggle,
  autoFocus = true,
  namePlaceholder = "Payments platform",
  keyPlaceholder = "PAY",
}: {
  form: ProjectCreateFormState;
  onChange: (next: ProjectCreateFormState) => void;
  advancedOpen: boolean;
  onAdvancedToggle: (open: boolean) => void;
  autoFocus?: boolean;
  namePlaceholder?: string;
  keyPlaceholder?: string;
}) {
  const keyConflictsWithBuiltIn = projectCreateKeyConflictsWithBuiltIn(form);

  const handleNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange({ ...form, name: event.target.value });
  };
  const handleKeyChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange({ ...form, key: event.target.value });
  };
  const handleSharedChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange({ ...form, shared: event.target.checked });
  };

  return (
    <div className="space-y-4">
      <label className="block space-y-1 text-sm">
        <span className="font-medium">Name</span>
        <Input
          value={form.name}
          onChange={handleNameChange}
          placeholder={namePlaceholder}
          autoFocus={autoFocus}
        />
      </label>

      <details
        className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-sm"
        open={advancedOpen}
        onToggle={(event) => {
          onAdvancedToggle((event.currentTarget as HTMLDetailsElement).open);
        }}
      >
        <summary className="cursor-pointer list-none text-sm font-medium text-muted-foreground hover:text-foreground">
          Advanced
        </summary>
        <div className="mt-3 space-y-4">
          <label className="block space-y-1 text-sm">
            <span className="font-medium">Code (optional)</span>
            <Input
              value={form.key}
              onChange={handleKeyChange}
              placeholder={keyPlaceholder}
            />
            <span className="block text-xs text-muted-foreground">
              Short code shown in URLs and tags (e.g., PAY).
            </span>
            {keyConflictsWithBuiltIn && (
              <span className="block text-xs text-destructive" role="alert">
                That code is reserved for system use. Pick a different one.
              </span>
            )}
          </label>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-0.5 size-4 rounded border-border accent-primary"
              checked={form.shared}
              onChange={handleSharedChange}
            />
            <span className="space-y-1">
              <span className="block font-medium">
                Make this a team-wide project
              </span>
              <span className="block text-xs text-muted-foreground">
                Team-wide projects surface for everyone, no matter what
                they're working on.
              </span>
            </span>
          </label>
        </div>
      </details>
    </div>
  );
}
