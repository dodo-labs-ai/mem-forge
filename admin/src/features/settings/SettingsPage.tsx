import { useModelSettings } from "@/api";
import { errorMessage } from "@/lib/errors";
import { ErrorNotice, PageHeader } from "@/patterns";
import { Skeleton } from "@/ui/skeleton";
import { useModelProviders } from "./api";
import { ModelSettingsView } from "./ModelSettingsView";

function LoadError({ query }: { query: { error: unknown; refetch: () => unknown } }) {
  return <ErrorNotice title="Couldn't load settings" message={errorMessage(query.error)} onRetry={() => void query.refetch()} />;
}

export function SettingsPage() {
  const settings = useModelSettings();
  const providers = useModelProviders();
  return (
    <div className="max-w-3xl space-y-5">
      <PageHeader title="Settings" description="The models MemForge uses to write memories and to search them." />
      {settings.isError ? (
        <LoadError query={settings} />
      ) : providers.isError ? (
        <LoadError query={providers} />
      ) : settings.isPending || providers.isPending ? (
        <div aria-busy className="space-y-4">
          <Skeleton className="h-80 rounded-xl" />
          <Skeleton className="h-64 rounded-xl" />
        </div>
      ) : (
        <ModelSettingsView settings={settings.data} providers={providers.data} />
      )}
    </div>
  );
}
