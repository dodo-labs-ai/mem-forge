import type { ReactNode } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export function AsyncBoundary({
  isLoading,
  isError,
  isEmpty,
  error,
  empty,
  onRetry,
  children,
}: {
  isLoading: boolean;
  isError?: boolean;
  isEmpty?: boolean;
  error?: unknown;
  empty: ReactNode;
  onRetry?: () => void;
  children: ReactNode;
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading...
      </div>
    );
  }

  if (isError) {
    const message = error instanceof Error ? error.message : "Unable to load data.";
    return (
      <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
        <AlertCircle className="mb-3 size-6 text-destructive" />
        <h3 className="text-sm font-medium">Something went wrong</h3>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">{message}</p>
        {onRetry && (
          <Button className="mt-4" variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        )}
      </div>
    );
  }

  if (isEmpty) return <>{empty}</>;
  return <>{children}</>;
}
