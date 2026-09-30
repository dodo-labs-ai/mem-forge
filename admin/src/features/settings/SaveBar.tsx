import { Check, Loader2 } from "lucide-react";
import { errorMessage } from "@/lib/errors";
import { ErrorNotice } from "@/patterns";
import { Button } from "@/ui/button";
import { endpointList } from "./model/settingsForm";
import type { EndpointKind } from "./model/types";

interface SaveBarProps {
  changed: EndpointKind[];
  pending: boolean;
  saved: boolean;
  error: unknown;
}

/** Saves every card at once and says which cards still have unsaved changes. */
export function SaveBar({ changed, pending, saved, error }: SaveBarProps) {
  const status = changed.length > 0 ? `Unsaved changes in ${endpointList(changed)}` : saved ? "Saved" : "";
  return (
    <div className="space-y-3">
      {error ? <ErrorNotice title="Couldn't save" message={errorMessage(error)} /> : null}
      <div className="flex flex-wrap items-center gap-3.5">
        <Button type="submit" size="lg" disabled={changed.length === 0 || pending}>
          {pending ? <Loader2 className="animate-spin" /> : <Check />}
          {pending ? "Saving…" : "Save changes"}
        </Button>
        <p role="status" className={changed.length > 0 ? "text-sm text-tone-warn-foreground" : "text-sm text-tone-ok-foreground"}>
          {status}
        </p>
      </div>
    </div>
  );
}
