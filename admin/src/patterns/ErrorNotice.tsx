import { AlertCircle } from "lucide-react";
import { Button } from "@/ui/button";

interface ErrorNoticeProps {
  title: string;
  message: string;
  onRetry?: () => void;
}

/** An inline failure with the server's reason and an optional retry. */
export function ErrorNotice({ title, message, onRetry }: ErrorNoticeProps) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-tone-danger/30 bg-tone-danger-soft p-3">
      <AlertCircle className="mt-0.5 size-4 shrink-0 text-tone-danger" />
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="font-medium text-tone-danger-foreground">{title}</p>
        <p className="text-sm text-tone-danger-foreground/90">{message}</p>
      </div>
      {onRetry ? (
        <Button size="sm" variant="outline" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  );
}
