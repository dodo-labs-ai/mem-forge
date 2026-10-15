import { AlertCircle, Clock, ListChecks, RefreshCw } from "lucide-react";
import { Notice } from "@/patterns";
import { awaitsTokenLimits, providerLabel, validationSummary } from "./model/modelStatus";
import { THRESHOLDS_NOT_CALIBRATED } from "./model/searchIndex";
import type { ModelBinding, ModelKind, ModelProvider } from "./model/types";

interface PendingEditNoticeProps {
  kind: ModelKind;
  /** The saved edit that is not the model in use. */
  edit: ModelBinding;
  /** The model the server keeps using meanwhile, when it has one. */
  inUse: ModelBinding | null;
  providers: readonly ModelProvider[];
  /** Whether search reads the search index as it is, and so keeps doing that while it is rebuilt. */
  vectorSearchOn?: boolean;
}

/**
 * Says that the saved edit is not the model in use, and what keeps it from
 * use: it has not been validated, its validation failed, the model answered
 * and only its token limits are missing, or it passed and the search index has
 * to be rebuilt for it first. The model in use is named so it is clear nothing
 * stopped working.
 */
export function PendingEditNotice({ kind, edit, inUse, providers, vectorSearchOn = true }: PendingEditNoticeProps) {
  const stillInUse =
    inUse === null
      ? `No ${kind} model is in use yet.`
      : `MemForge keeps using ${providerLabel(inUse.provider, providers)} ${inUse.model_id}.`;

  if (edit.waits_for_reindex) {
    return (
      <Notice tone="live" icon={RefreshCw} title="These settings passed validation and wait for the search index">
        <p>
          Vectors of this model cannot be compared with the ones stored so far, so the search index is rebuilt before
          MemForge uses the model. The rebuild embeds every memory again; meanwhile{" "}
          {vectorSearchOn ? "search keeps working" : "search ranks by matching words, as it does now"}, and source syncs
          and changes to memories wait.
        </p>
        {edit.profile?.thresholds_calibrated === false ? (
          <p>
            {THRESHOLDS_NOT_CALIBRATED.label}: {THRESHOLDS_NOT_CALIBRATED.text}
          </p>
        ) : null}
        <p>{stillInUse}</p>
      </Notice>
    );
  }

  const summary = validationSummary(edit, "candidate");
  if (awaitsTokenLimits(edit)) {
    return (
      <Notice tone="warn" icon={ListChecks} title="One more step: enter the token limits">
        <p>{summary.detail}</p>
        <p>{stillInUse}</p>
      </Notice>
    );
  }
  const failed = edit.validation !== null && !edit.validation.passed;
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
