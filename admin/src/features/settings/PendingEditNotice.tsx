import { AlertCircle, Clock } from "lucide-react";
import { Notice } from "@/patterns";
import { providerLabel, validationSummary } from "./model/modelStatus";
import type { ModelBinding, ModelKind, ModelProvider } from "./model/types";

interface PendingEditNoticeProps {
  kind: ModelKind;
  /** The saved edit that has not passed validation. */
  edit: ModelBinding;
  /** The model the server keeps using meanwhile, when it has one. */
  inUse: ModelBinding | null;
  providers: readonly ModelProvider[];
}

/**
 * Says that the saved edit is not the model in use: either it has not been
 * validated, or its validation failed, and why. The model in use is named so
 * it is clear nothing stopped working.
 */
export function PendingEditNotice({ kind, edit, inUse, providers }: PendingEditNoticeProps) {
  const summary = validationSummary(edit, "candidate");
  const failed = edit.validation !== null && !edit.validation.passed;
  const stillInUse =
    inUse === null
      ? `No ${kind} model is in use yet.`
      : `MemForge keeps using ${providerLabel(inUse.provider, providers)} ${inUse.model_id}.`;
  return (
    <Notice
      tone={failed ? "danger" : "warn"}
      icon={failed ? AlertCircle : Clock}
      title={failed ? "These settings failed validation and are not in use" : "These settings are saved but not in use"}
    >
      <p>{failed ? `${summary.headline} ${summary.detail ?? ""}` : "Validate them to put them to use."}</p>
      <p>{stillInUse}</p>
    </Notice>
  );
}
