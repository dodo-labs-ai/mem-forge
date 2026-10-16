import { AlertCircle, Clock, ListChecks, ListRestart, Loader2, RefreshCw } from "lucide-react";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { awaitsTokenLimits, providerLabel, validationSummary } from "./model/modelStatus";
import { THRESHOLDS_NOT_CALIBRATED } from "./model/searchIndex";
import type { ModelBinding, ModelProvider } from "./model/types";

/** Filling in the token limits from the provider's model list, for a provider whose listing can give them. */
export interface ListFill {
  /** Loads the model list and fills in the limits it gives for the model; `null` once there is nothing to fill in. */
  onFill: (() => void) | null;
  loading: boolean;
  /** Why the list could not fill in every limit, once it was looked up. */
  message: string | null;
}

interface PendingEditNoticeProps {
  /** The saved edit that is not the model in use. */
  edit: ModelBinding;
  /** The model the server keeps using meanwhile, when it has one. */
  inUse: ModelBinding | null;
  providers: readonly ModelProvider[];
  /** Whether search reads the search index as it is, and so keeps doing that while it is rebuilt. */
  vectorSearchOn?: boolean;
  /** Offered while only the token limits are missing and the provider's model list can give them. */
  listFill?: ListFill;
}

/**
 * Says that the saved edit is not the model in use, and what keeps it from
 * use: it has not been validated, its validation failed, the model answered
 * and only its token limits are missing, or it passed and the search index has
 * to be rebuilt for it first. The model in use is named so it is clear nothing
 * stopped working; while there is none, the card says so.
 */
export function PendingEditNotice({ edit, inUse, providers, vectorSearchOn = true, listFill }: PendingEditNoticeProps) {
  const stillInUse =
    inUse === null ? null : <p>{`MemForge keeps using ${providerLabel(inUse.provider, providers)} ${inUse.model_id}.`}</p>;

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
        {stillInUse}
      </Notice>
    );
  }

  const summary = validationSummary(edit, "candidate");
  if (awaitsTokenLimits(edit)) {
    return (
      <Notice
        tone="warn"
        icon={ListChecks}
        title="One more step: enter the token limits"
        action={
          listFill?.onFill ? (
            <Button type="button" variant="outline" size="sm" onClick={listFill.onFill} disabled={listFill.loading}>
              {listFill.loading ? <Loader2 className="animate-spin" /> : <ListRestart />}
              {listFill.loading ? "Loading the model list…" : "Fill from the model list"}
            </Button>
          ) : null
        }
      >
        <p>{summary.detail}</p>
        {listFill?.message ? <p>{listFill.message}</p> : null}
        {stillInUse}
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
      {stillInUse}
    </Notice>
  );
}
