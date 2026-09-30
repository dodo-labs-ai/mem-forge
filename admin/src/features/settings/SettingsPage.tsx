import { errorMessage } from "@/lib/errors";
import { ErrorNotice, PageHeader } from "@/patterns";
import { Skeleton } from "@/ui/skeleton";
import { useLlmConfig } from "./api";
import { LlmSettingsForm } from "./LlmSettingsForm";

export function SettingsPage() {
  const config = useLlmConfig();
  return (
    <div className="max-w-3xl space-y-5">
      <PageHeader title="Settings" description="The model endpoints MemForge uses to extract memories and to search them." />
      {config.isPending ? (
        <div aria-busy className="space-y-4">
          <Skeleton className="h-80 rounded-xl" />
          <Skeleton className="h-64 rounded-xl" />
        </div>
      ) : config.isError ? (
        <ErrorNotice title="Couldn't load settings" message={errorMessage(config.error)} onRetry={() => void config.refetch()} />
      ) : (
        <LlmSettingsForm config={config.data} />
      )}
    </div>
  );
}
