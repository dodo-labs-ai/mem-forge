import { ExternalLink } from "lucide-react";
import { errorMessage } from "@/lib/errors";
import { pluralize } from "@/lib/format";
import { Button } from "@/ui/button";
import type { SetupPreview } from "./api";

interface CheckSetupPanelProps {
  /** Every required field has a value, so the server can be asked. */
  ready: boolean;
  pending: boolean;
  preview: SetupPreview | undefined;
  /** Why the check failed; a refused configuration is shown on its fields instead. */
  error: unknown;
  onCheck: () => void;
}

/**
 * "Check setup" after the type's content step: the first items the
 * configuration would sync, read from the provider without saving anything.
 */
export function CheckSetupPanel({ ready, pending, preview, error, onCheck }: CheckSetupPanelProps) {
  return (
    <section aria-labelledby="check-setup-heading" className="space-y-3 rounded-lg border p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-0.5">
          <h3 id="check-setup-heading" className="text-sm font-medium">
            Check setup
          </h3>
          <p className="text-xs text-muted-foreground">
            {ready ? "List the first items this source would sync. Nothing is saved." : "Complete the required fields first."}
          </p>
        </div>
        <Button type="button" size="sm" variant="outline" disabled={!ready || pending} onClick={onCheck}>
          {pending ? "Checking…" : "Check setup"}
        </Button>
      </div>
      {error ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          {errorMessage(error)}
        </p>
      ) : null}
      {preview && !error ? (
        preview.items.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nothing to sync was found. Check the scope settings.</p>
        ) : (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              {preview.truncated
                ? `The first ${pluralize(preview.items.length, "item")} found`
                : `${pluralize(preview.count, "item")} found`}
            </p>
            <ul aria-label="First items found" className="space-y-1">
              {preview.items.map((item) => (
                <li key={item.id} className="text-sm">
                  {item.href ? (
                    <a href={item.href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-foreground hover:underline">
                      {item.title}
                      <ExternalLink aria-hidden className="size-3 text-muted-foreground" />
                    </a>
                  ) : (
                    <span className="font-mono text-xs">{item.title}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )
      ) : null}
    </section>
  );
}
