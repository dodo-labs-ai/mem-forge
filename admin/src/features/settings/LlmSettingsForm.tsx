import { zodResolver } from "@hookform/resolvers/zod";
import { Lock } from "lucide-react";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import { Notice } from "@/patterns";
import { useSaveLlmConfig } from "./api";
import { EndpointCard } from "./EndpointCard";
import { LeaveDialog } from "./LeaveDialog";
import { changedEndpoints, ENDPOINT_KINDS, settingsSchema, settingsValues, updateRequest, type SettingsValues } from "./model/settingsForm";
import type { LlmConfig } from "./model/types";
import { SaveBar } from "./SaveBar";
import { useUnsavedChangesGuard } from "./useUnsavedChangesGuard";

/**
 * The enrichment and embedding endpoints as one form. When the deployment
 * environment manages LLM settings, the server says so up front and the form
 * is shown read-only, with values the user can still select and copy.
 */
export function LlmSettingsForm({ config }: { config: LlmConfig }) {
  const saved = settingsValues(config);
  const form = useForm<SettingsValues>({ defaultValues: saved, resolver: zodResolver(settingsSchema) });
  const [enrichment, embedding] = useWatch({ control: form.control, name: ["enrichment", "embedding"] });
  const changed = changedEndpoints(saved, { enrichment, embedding });
  const save = useSaveLlmConfig();
  const blocker = useUnsavedChangesGuard(changed.length > 0);
  const readOnly = !config.writable;

  const submit = form.handleSubmit((values) =>
    save.mutate(updateRequest(saved, values), { onSuccess: (stored) => form.reset(settingsValues(stored)) }),
  );

  return (
    <FormProvider {...form}>
      <form onSubmit={submit} noValidate aria-label="Model endpoints" className="space-y-4">
        {readOnly ? (
          <Notice tone="live" icon={Lock}>
            <strong className="font-semibold">Managed by the deployment environment.</strong> These settings come from
            environment variables and cannot be changed here. Change them in the environment and restart the service.
          </Notice>
        ) : null}
        {ENDPOINT_KINDS.map((kind) => (
          <EndpointCard key={kind} kind={kind} config={config} readOnly={readOnly} />
        ))}
        {readOnly ? null : (
          <SaveBar changed={changed} pending={save.isPending} saved={save.isSuccess} error={save.error} />
        )}
      </form>
      <LeaveDialog blocker={blocker} changed={changed} />
    </FormProvider>
  );
}
