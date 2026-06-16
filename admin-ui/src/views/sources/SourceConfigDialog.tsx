import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Check, Loader2, RefreshCw } from "lucide-react";
import client from "@/api/client";
import type {
  ConfigField,
  DiscoveryPreviewResponse,
  GeneConfigSchema,
  JiraAuthSession,
  ProjectBinding,
  Source,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  applyConfluenceUrlInference,
  isConfluenceFieldRequired,
  isConfluenceFieldVisible,
  parseConfluenceWikiUrl,
} from "./confluenceConfig";
import type { ParsedConfluenceWikiUrl } from "./confluenceConfig";
import { buildLocalMarkdownPushCommand } from "./localMarkdownConfig";
import { canConfigureSourceType } from "./managedSources";
import { ProjectBindingFields } from "./ProjectBindingFields";
import { projectBindingIsComplete } from "./projectBinding";

type ConfigValue = string | number | boolean | string[] | null;
type ConfigForm = Record<string, ConfigValue>;
const DISCOVERY_PREVIEW_LIMIT = 5;

export function SourceConfigDialog({
  open,
  onOpenChange,
  sourceType,
  source,
  onSaved,
  initialFocus,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceType: string | null;
  source?: Source | null;
  onSaved?: () => void;
  initialFocus?: { step: "project" };
}) {
  // A local repository is created from the CLI (which scans the folder and
  // pushes), not by hand in the UI, so a new one shows setup instructions
  // instead of a config form.
  const isNewLocalRepo = sourceType === "local_markdown" && !source;

  // Backend authority: an existing source the viewer cannot configure should
  // not open the form. Type-level managed sources (agent_session) are also
  // blocked. New-source flows have no source row yet, so the type check is
  // the only gate there.
  const canConfigureExisting = source ? source.capabilities?.can_configure !== false : true;

  const schemaQuery = useQuery<GeneConfigSchema>({
    queryKey: ["gene-config-schema", sourceType],
    queryFn: () =>
      client.get(`/api/genes/${sourceType}/config-schema`).then((response) => response.data),
    enabled:
      open
      && Boolean(sourceType)
      && canConfigureSourceType(sourceType ?? "")
      && canConfigureExisting
      && !isNewLocalRepo,
  });

  if (!sourceType || !canConfigureSourceType(sourceType)) return null;
  if (!canConfigureExisting) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[90vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-2xl">
        {isNewLocalRepo ? (
          <LocalRepoSetupInstructions onClose={() => onOpenChange(false)} />
        ) : schemaQuery.isPending ? (
          <div className="flex items-center justify-center gap-2 p-12 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading source schema...
          </div>
        ) : schemaQuery.isError ? (
          <div className="p-4">
            <DialogHeader>
              <DialogTitle>Configure source</DialogTitle>
            </DialogHeader>
            <div className="mt-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
              Failed to load source configuration schema.
            </div>
          </div>
        ) : !schemaQuery.data ? (
          <div className="flex items-center justify-center gap-2 p-12 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading source schema...
          </div>
        ) : (
          <SourceConfigForm
            key={`${source?.id ?? "new"}-${sourceType}`}
            sourceType={sourceType}
            source={source}
            schema={schemaQuery.data}
            onOpenChange={onOpenChange}
            onSaved={onSaved}
            initialFocus={initialFocus}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function SourceConfigForm({
  sourceType,
  source,
  schema,
  onOpenChange,
  onSaved,
  initialFocus,
}: {
  sourceType: string;
  source?: Source | null;
  schema: GeneConfigSchema;
  onOpenChange: (open: boolean) => void;
  onSaved?: () => void;
  initialFocus?: { step: "project" };
}) {
  const queryClient = useQueryClient();
  const isEdit = Boolean(source);
  const [name, setName] = useState(source?.name ?? "");
  const [config, setConfig] = useState<ConfigForm>(() => ({
    ...buildDefaultConfig(schema.fields),
    ...initialSourceConfig(sourceType, (source?.config ?? {}) as ConfigForm),
  }));
  const [binding, setBinding] = useState<ProjectBinding | null>(
    () => source?.project_binding ?? null,
  );
  const [scheduleEnabled, setScheduleEnabled] = useState(
    () => Boolean(source?.sync_schedule?.enabled),
  );
  const [scheduleInterval, setScheduleInterval] = useState(
    () => String(source?.sync_schedule?.interval_minutes ?? 1440),
  );
  const projectSectionRef = useRef<HTMLDivElement | null>(null);
  const authMode = stringValue(config.auth_mode) || "browser_cookie";
  const jiraBaseUrl = stringValue(config.base_url).trim();
  const confluenceUrlInfo = useMemo(
    () => sourceType === "confluence" ? parseConfluenceWikiUrl(stringValue(config.base_url)) : null,
    [config.base_url, sourceType],
  );

  useEffect(() => {
    if (initialFocus?.step === "project" && projectSectionRef.current) {
      projectSectionRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [initialFocus]);

  const jiraSessionQuery = useQuery<JiraAuthSession>({
    queryKey: ["jira-session", jiraBaseUrl],
    queryFn: () =>
      client.get("/api/auth/jira-session", { params: { base_url: jiraBaseUrl } }).then((response) => response.data),
    enabled: sourceType === "jira" && authMode === "browser_cookie" && jiraBaseUrl.startsWith("https://"),
  });

  const saveSource = useMutation({
    mutationFn: async (payload: {
      name: string;
      config: ConfigForm;
      project_binding: ProjectBinding | null;
    }) => {
      const intervalMinutes = parseScheduleInterval(scheduleInterval);
      const payloadWithSchedule = {
        ...payload,
        sync_schedule: {
          enabled: scheduleEnabled,
          interval_minutes: intervalMinutes,
        },
      };
      if (source) {
        await client.put(`/api/sources/${source.id}`, payloadWithSchedule);
        return { id: source.id };
      }
      const response = await client.post("/api/sources", {
        type: sourceType,
        ...payloadWithSchedule,
      });
      return { id: String(response.data.id) };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      onSaved?.();
      onOpenChange(false);
    },
  });

  const previewDiscovery = useMutation<DiscoveryPreviewResponse, unknown, void>({
    mutationFn: () =>
      client
        .post(`/api/genes/${sourceType}/preview-discovery`, {
          config: serializeConfig(schema.fields, config),
          limit: DISCOVERY_PREVIEW_LIMIT,
        })
        .then((response) => response.data as DiscoveryPreviewResponse),
  });

  const fieldsByGroup = useMemo(() => {
    const fields = [...schema.fields].sort((a, b) => a.order - b.order);
    return [...schema.groups]
      .sort((a, b) => a.order - b.order)
      .map((group) => ({
        ...group,
        fields: fields.filter((field) => field.group === group.key && !field.advanced && isFieldVisible(sourceType, field, config)),
      }))
      .filter((group) => group.fields.length > 0);
  }, [config, schema, sourceType]);
  const advancedFields = useMemo(
    () => [...schema.fields]
      .filter((field) => field.advanced && isFieldVisible(sourceType, field, config))
      .sort((a, b) => a.order - b.order),
    [config, schema, sourceType],
  );

  const canSave =
    name.trim().length > 0 &&
    requiredFieldsAreFilled(sourceType, schema.fields, config) &&
    projectBindingIsComplete(binding) &&
    parseScheduleInterval(scheduleInterval) >= 5;

  const previewReady = requiredFieldsAreFilled(sourceType, schema.fields, config);

  const updateField = (field: ConfigField, value: ConfigValue) => {
    setConfig((current) => {
      const next = { ...current, [field.key]: value };
      if (sourceType === "confluence" && field.key === "base_url") {
        return applyConfluenceUrlInference(next) as ConfigForm;
      }
      return next;
    });
  };

  const handleSave = () => {
    saveSource.mutate({
      name: name.trim(),
      config: serializeConfig(schema.fields, config),
      project_binding: binding,
    });
  };

  return (
    <>
        <div className="flex min-h-0 flex-1 flex-col gap-4 p-4">
          <DialogHeader>
            <DialogTitle>
              {isEdit ? "Configure source" : `Configure ${sourceType ?? "source"}`}
            </DialogTitle>
          </DialogHeader>

          <Field label="Source name" required>
            <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Source name" />
          </Field>

          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto pr-1">
            {fieldsByGroup.map((group) => (
              <section key={group.key} className="space-y-3">
                <h3 className="text-sm font-semibold">{group.label}</h3>
                <div className="space-y-3">
                  {group.fields.map((field) => (
                    <div key={field.key} className="space-y-3">
                      <ConfigFieldInput
                        field={field}
                        value={config[field.key]}
                        hasExistingSecret={Boolean(config[`${field.key}_configured`])}
                        decryptFailed={Boolean(config[`${field.key}_decrypt_failed`])}
                        required={isFieldRequired(sourceType, field, config)}
                        onChange={(value) => updateField(field, value)}
                      />
                      {sourceType === "confluence" && field.key === "base_url" && confluenceUrlInfo && (
                        <ConfluenceDetectedPanel info={confluenceUrlInfo} />
                      )}
                      {sourceType === "jira" && field.key === "jql" && (
                        <JiraEffectiveQueryPanel jql={stringValue(config.jql)} />
                      )}
                      {sourceType === "jira" && field.key === "auth_mode" && authMode === "browser_cookie" && (
                        <JiraBrowserSessionPanel
                          baseUrl={jiraBaseUrl}
                          session={jiraSessionQuery.data}
                          loading={jiraSessionQuery.isFetching}
                          error={jiraSessionQuery.error}
                          onRefresh={() => {
                            void jiraSessionQuery.refetch();
                          }}
                        />
                      )}
                      {sourceType === "local_markdown" && field.key === "vault_id" && (
                        <LocalMarkdownPushPanel
                          sourceId={source?.id ?? null}
                          vaultId={stringValue(config.vault_id).trim()}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </section>
            ))}
            {advancedFields.length > 0 && (
              <details className="space-y-3">
                <summary className="cursor-pointer text-sm font-semibold">Advanced</summary>
                <div className="space-y-3 pt-2">
                  {advancedFields.map((field) => (
                    <ConfigFieldInput
                      key={field.key}
                      field={field}
                      value={config[field.key]}
                      hasExistingSecret={Boolean(config[`${field.key}_configured`])}
                      decryptFailed={Boolean(config[`${field.key}_decrypt_failed`])}
                      required={isFieldRequired(sourceType, field, config)}
                      onChange={(value) => updateField(field, value)}
                    />
                  ))}
                </div>
              </details>
            )}

            {/* Local repositories are pushed from the CLI, not discovered server-side,
                so the discovery preview does not apply (and its inbox is empty here). */}
            {sourceType !== "local_markdown" && (
              <DiscoveryPreviewPanel
                ready={previewReady}
                isPending={previewDiscovery.isPending}
                error={previewDiscovery.isError ? previewDiscovery.error : null}
                data={previewDiscovery.data}
                onPreview={() => previewDiscovery.mutate()}
              />
            )}

            <div ref={projectSectionRef}>
              <ProjectBindingFields
                schema={schema}
                sourceId={source?.id ?? null}
                value={binding}
                onChange={setBinding}
              />
            </div>

            <section className="space-y-3">
              <h3 className="text-sm font-semibold">Automatic sync</h3>
              <label className="flex items-start gap-3 rounded-lg border p-3">
                <input
                  type="checkbox"
                  className="mt-0.5 size-4"
                  checked={scheduleEnabled}
                  onChange={(event) => setScheduleEnabled(event.target.checked)}
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">Sync on a schedule</span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    Runs through the same queue as manual syncs.
                  </span>
                </span>
              </label>
              <Field
                label="Interval"
                helpText="Minimum 5 minutes. Existing syncs are not interrupted."
              >
                <Select<string>
                  value={scheduleInterval}
                  onValueChange={(value) => {
                    if (value) setScheduleInterval(value);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue>{scheduleIntervalLabel(scheduleInterval)}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="30">Every 30 minutes</SelectItem>
                    <SelectItem value="60">Hourly</SelectItem>
                    <SelectItem value="360">Every 6 hours</SelectItem>
                    <SelectItem value="720">Every 12 hours</SelectItem>
                    <SelectItem value="1440">Daily</SelectItem>
                    <SelectItem value="10080">Weekly</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
            </section>
          </div>

          {saveSource.isError && (
            <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
              {extractSaveError(saveSource.error)}
            </div>
          )}
        </div>

        <DialogFooter className="mx-0 mb-0 flex-row justify-between rounded-none rounded-b-xl bg-background p-3 sm:justify-between">
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" onClick={handleSave} disabled={!canSave || saveSource.isPending}>
            {saveSource.isPending ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
            {isEdit ? "Save Changes" : "Create Source"}
          </Button>
        </DialogFooter>
    </>
  );
}

function ConfigFieldInput({
  field,
  value,
  hasExistingSecret,
  decryptFailed,
  required,
  onChange,
}: {
  field: ConfigField;
  value: ConfigValue | undefined;
  hasExistingSecret?: boolean;
  decryptFailed?: boolean;
  required?: boolean;
  onChange: (value: ConfigValue) => void;
}) {
  if (field.field_type === "boolean") {
    return (
      <label className="flex items-start gap-3 rounded-lg border p-3">
        <input
          type="checkbox"
          className="mt-0.5 size-4"
          checked={toBoolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          <span className="block text-sm font-medium">
            {field.label}
            {required && <span className="text-destructive"> *</span>}
          </span>
          {field.help_text && <span className="mt-1 block text-xs text-muted-foreground">{field.help_text}</span>}
        </span>
      </label>
    );
  }

  if (field.field_type === "select") {
    const selected = stringValue(value || field.default || field.options[0] || "");
    return (
      <Field label={field.label} required={required} helpText={field.help_text}>
        <Select<string>
          value={optionValue(field, selected)}
          onValueChange={(next) => onChange(optionFromValue(field, stringValue(next)))}
        >
          <SelectTrigger>
            <SelectValue>{selected ? optionLabel(field, selected) : "Select..."}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {field.options.map((option) => (
              <SelectItem key={option} value={optionValue(field, option)}>
                {optionLabel(field, option)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
    );
  }

  if (field.field_type === "multi_select") {
    const selected = new Set(listValue(value));
    return (
      <Field label={field.label} required={required} helpText={field.help_text}>
        <div className="flex flex-wrap gap-2">
          {field.options.map((option) => (
            <label key={option} className="flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm">
              <input
                type="checkbox"
                checked={selected.has(option)}
                onChange={(event) => {
                  const next = new Set(selected);
                  if (event.target.checked) next.add(option);
                  else next.delete(option);
                  onChange([...next]);
                }}
              />
              {option}
            </label>
          ))}
        </div>
      </Field>
    );
  }

  if (field.key === "jql") {
    return (
      <Field label={field.label} required={required} helpText={field.help_text}>
        <textarea
          className="min-h-20 w-full rounded-md border border-input bg-background px-2.5 py-1.5 font-mono text-xs shadow-xs outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          rows={4}
          value={stringValue(value)}
          placeholder={field.placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      </Field>
    );
  }

  const isInteger = field.field_type === "integer";
  const isList = field.field_type === "tag_list";
  const isSecret = field.field_type === "secret";

  return (
    <Field label={field.label} required={required} helpText={field.help_text}>
      <Input
        type={isInteger ? "number" : isSecret ? "password" : "text"}
        value={isList ? listValue(value).join(", ") : stringValue(value)}
        onChange={(event) => {
          if (isInteger) {
            onChange(event.target.value === "" ? "" : Number(event.target.value));
          } else if (isList) {
            onChange(event.target.value);
          } else {
            onChange(event.target.value);
          }
        }}
        placeholder={isSecret && hasExistingSecret ? "Leave blank to keep existing token" : field.placeholder}
      />
      {isSecret && decryptFailed && (
        <span className="block text-xs text-destructive">Stored token cannot be decrypted. Re-enter it.</span>
      )}
    </Field>
  );
}

function ConfluenceDetectedPanel({ info }: { info: ParsedConfluenceWikiUrl }) {
  const rows = [
    ["Site", info.normalizedBaseUrl],
    ["API path", info.apiPrefix],
    ["Space", info.spaceKey],
    ["Root page", info.pageId],
  ].filter(([, value]) => value);

  if (rows.length === 0) return null;

  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs">
      <div className="font-medium">Detected</div>
      <dl className="mt-1 grid gap-1 sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="truncate font-medium">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function buildEffectiveJqlPreview(raw: string): string {
  const query = raw.trim();
  if (!query) return "";
  const match = query.match(/\border\s+by\b/i);
  const where = match ? query.slice(0, match.index).trim() : query;
  const order = match ? query.slice(match.index).trim() : "ORDER BY updated DESC";
  const delta = "updated >= '<last-sync>'";
  const whereWithDelta = where ? `(${where}) AND ${delta}` : delta;
  return `${whereWithDelta} ${order}`.trim();
}

function JiraEffectiveQueryPanel({ jql }: { jql: string }) {
  const preview = buildEffectiveJqlPreview(jql);
  if (!preview) return null;
  return (
    <div className="rounded-lg border bg-muted/30 p-3 text-xs">
      <div className="text-sm font-medium text-foreground">Effective query at sync</div>
      <p className="mt-1 text-muted-foreground">
        Your JQL runs as-is. MemForge inserts an incremental updated-since filter before your
        ORDER BY so each sync only fetches changed issues.
      </p>
      <code className="mt-2 block break-all rounded-md border bg-background p-2 font-mono text-[11px] text-foreground">
        {preview}
      </code>
    </div>
  );
}

function JiraBrowserSessionPanel({
  baseUrl,
  session,
  loading,
  error,
  onRefresh,
}: {
  baseUrl: string;
  session?: JiraAuthSession;
  loading: boolean;
  error: unknown;
  onRefresh: () => void;
}) {
  const status = session?.status ?? "missing";
  const isActive = status === "active";
  const statusLabel =
    status === "active" ? "Active" :
      status === "expired" ? "Expired" :
        status === "failed" ? "Failed" :
          "Missing";
  const principal = session?.principal_name || session?.principal_id || "unknown user";
  const errorText = error ? extractSaveError(error) : session?.last_error || "";
  const command = `memforge adapter auth jira refresh --base-url ${baseUrl || "<jira-base-url>"}`;
  const handleCopy = () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    void navigator.clipboard.writeText(command);
  };

  return (
    <div className="rounded-lg border p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm font-medium">
        <span>Browser session (local adapter)</span>
        <span className={isActive ? "text-emerald-600" : "text-destructive"}>{statusLabel}</span>
        {loading && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isActive
          ? `Signed in as ${principal}${session?.browser ? ` via ${session.browser}` : ""}`
          : "MemForge never reads your browser. Capture the session from the machine where you're signed in to Jira, using the local CLI adapter; it appears here once captured."}
      </p>
      {session?.origin && (
        <p className="mt-1 break-all text-xs text-muted-foreground">{session.origin}</p>
      )}
      <div className="mt-3 flex items-center gap-2 rounded-md border bg-background p-2">
        <code className="flex-1 break-all font-mono text-[11px] text-foreground">{command}</code>
        <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
          Copy
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : "Refresh"}
        </Button>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {isActive
          ? "Re-run this on your machine to refresh when the session expires."
          : "Run this on your machine. A different Jira login? Re-run it (the CLI confirms the principal change)."}
      </p>
      {errorText && (
        <div className="mt-2 flex items-start gap-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
          <AlertCircle className="mt-0.5 size-3 shrink-0" />
          <span className="min-w-0 whitespace-normal break-words">
            {errorText}
          </span>
        </div>
      )}
    </div>
  );
}

function CliCommand({ command }: { command: string }) {
  const copy = () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) void navigator.clipboard.writeText(command);
  };
  return (
    <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-2">
      <code className="flex-1 break-all font-mono text-[11px] text-foreground">{command}</code>
      <Button type="button" variant="outline" size="sm" onClick={copy}>
        Copy
      </Button>
    </div>
  );
}

function LocalRepoSetupInstructions({ onClose }: { onClose: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 p-4">
      <DialogHeader>
        <DialogTitle>Add a local repository</DialogTitle>
      </DialogHeader>
      <p className="text-sm text-muted-foreground">
        A local repository is set up from the CLI. MemForge does not read your filesystem, so the
        CLI scans your folder and pushes its files into a source it creates for you.
      </p>

      <div className="space-y-2">
        <div className="text-sm font-medium text-foreground">Guided setup (recommended)</div>
        <CliCommand command="memforge" />
        <p className="text-xs text-muted-foreground">
          Run the CLI and choose{" "}
          <span className="font-medium text-foreground">Local repository &rarr; Set up a repository</span>.
          It walks you through the folder, shows a quick scan, creates this source, and runs the
          first sync.
        </p>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
          Prefer one-off commands?
        </summary>
        <div className="mt-2 space-y-2">
          <CliCommand command="memforge adapter kb add my-repo --root /path/to/folder --create-source" />
          <CliCommand command="memforge adapter kb push my-repo --process-now" />
        </div>
      </details>

      <DialogFooter className="mx-0 mb-0 flex-row justify-end rounded-none rounded-b-xl bg-background p-3">
        <Button type="button" onClick={onClose}>
          Got it
        </Button>
      </DialogFooter>
    </div>
  );
}

function LocalMarkdownPushPanel({
  sourceId,
  vaultId,
}: {
  sourceId: string | null;
  vaultId: string;
}) {
  const command = buildLocalMarkdownPushCommand({ vaultId, sourceId });
  const ready = Boolean(sourceId) && vaultId.length > 0;
  const handleCopy = () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    void navigator.clipboard.writeText(command);
  };

  return (
    <div className="rounded-lg border bg-muted/30 p-3 text-xs">
      <div className="text-sm font-medium text-foreground">Push from the local CLI adapter</div>
      <p className="mt-1 text-muted-foreground">
        MemForge does not read your filesystem. The local CLI adapter scans your markdown
        folder, normalizes each file, and pushes documents into this source. Configure a
        local profile with <code>memforge adapter kb add</code>, then run the push command
        below to send documents through the service.
      </p>
      <div className="mt-3 flex items-center gap-2 rounded-md border bg-background p-2">
        <code className="flex-1 break-all font-mono text-[11px] text-foreground">{command}</code>
        <Button type="button" variant="outline" size="sm" onClick={handleCopy} disabled={!ready}>
          Copy
        </Button>
      </div>
      {!sourceId && (
        <p className="mt-2 text-muted-foreground">
          Save this source first to get a stable source id, then run the push command.
        </p>
      )}
    </div>
  );
}

function DiscoveryPreviewPanel({
  ready,
  isPending,
  error,
  data,
  onPreview,
}: {
  ready: boolean;
  isPending: boolean;
  error: unknown;
  data: DiscoveryPreviewResponse | undefined;
  onPreview: () => void;
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Preview discovery</h3>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onPreview}
          disabled={!ready || isPending}
        >
          {isPending ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {data ? "Refresh" : "Preview"}
        </Button>
      </div>
      {!ready ? (
        <p className="text-xs text-muted-foreground">
          Fill in the required fields above to preview discoverable items.
        </p>
      ) : !data && !error && !isPending ? (
        <p className="text-xs text-muted-foreground">
          Run a dry discovery against the configured source to verify the result set before saving.
        </p>
      ) : null}
      {error != null && (
        <div className="flex items-start gap-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
          <AlertCircle className="mt-0.5 size-3 shrink-0" />
          <span className="min-w-0 whitespace-normal break-words">{extractSaveError(error)}</span>
        </div>
      )}
      {data && (
        <div className="rounded-lg border p-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
            <span>
              {data.count} item{data.count === 1 ? "" : "s"} discovered
            </span>
            {data.truncated && (
              <span>Showing first {data.items.length} of {data.count}</span>
            )}
          </div>
          {data.items.length === 0 ? (
            <p className="mt-2 text-xs text-muted-foreground">No items matched the current configuration.</p>
          ) : (
            <ul className="mt-2 space-y-1.5">
              {data.items.map((item) => (
                <li key={item.item_id} className="text-xs">
                  <a
                    href={item.source_url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="font-medium text-primary underline-offset-2 hover:underline break-all"
                  >
                    {item.title || item.source_url}
                  </a>
                  {item.last_modified && (
                    <span className="ml-2 text-muted-foreground">{formatPreviewDate(item.last_modified)}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {data.truncated && (
            <p className="mt-2 text-[11px] text-muted-foreground">
              Result set truncated by the server. The first {data.items.length} items are shown above.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function formatPreviewDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().slice(0, 10);
}

function Field({
  label,
  required = false,
  helpText,
  children,
}: {
  label: string;
  required?: boolean;
  helpText?: string;
  children: ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        {required && <span className="text-destructive"> *</span>}
      </span>
      {children}
      {helpText && <span className="block text-xs text-muted-foreground">{helpText}</span>}
    </label>
  );
}

function buildDefaultConfig(fields: ConfigField[]): ConfigForm {
  return fields.reduce<ConfigForm>((acc, field) => {
    if (field.default !== "") {
      acc[field.key] = defaultValueForField(field);
    }
    return acc;
  }, {});
}

function initialSourceConfig(sourceType: string, sourceConfig: ConfigForm): ConfigForm {
  const next = { ...sourceConfig };
  if (sourceType === "jira" && !next.auth_mode) {
    if (next.pat_configured) next.auth_mode = "pat";
    else if (next.jira_cookie_configured) next.auth_mode = "browser_cookie";
  }
  if (sourceType === "confluence") {
    return applyConfluenceUrlInference(next) as ConfigForm;
  }
  return next;
}

function defaultValueForField(field: ConfigField): ConfigValue {
  if (field.field_type === "boolean") return field.default === "true";
  if (field.field_type === "integer") return Number(field.default);
  if (field.field_type === "tag_list" || field.field_type === "multi_select") {
    return parseCommaList(field.default);
  }
  return field.default;
}

function serializeConfig(fields: ConfigField[], config: ConfigForm): ConfigForm {
  return fields.reduce<ConfigForm>((acc, field) => {
    const value = config[field.key];
    if (field.field_type === "tag_list" || field.field_type === "multi_select") {
      acc[field.key] = listValue(value);
    } else if (field.field_type === "boolean") {
      acc[field.key] = toBoolean(value);
    } else if (field.field_type === "integer") {
      acc[field.key] = value === "" || value == null ? null : Number(value);
    } else if (field.field_type === "secret") {
      const text = stringValue(value);
      if (text.trim().length > 0 || !config[`${field.key}_configured`]) {
        acc[field.key] = text;
      }
    } else {
      acc[field.key] = stringValue(value);
    }
    return acc;
  }, {});
}

function requiredFieldsAreFilled(sourceType: string, fields: ConfigField[], config: ConfigForm): boolean {
  return fields.every((field) => {
    if (!isFieldVisible(sourceType, field, config)) return true;
    if (!isFieldRequired(sourceType, field, config)) return true;
    const value = config[field.key];
    if (field.field_type === "tag_list" || field.field_type === "multi_select") {
      return listValue(value).length > 0;
    }
    if (field.field_type === "secret" && config[`${field.key}_configured`]) {
      return true;
    }
    return stringValue(value).trim().length > 0;
  });
}

function isFieldVisible(sourceType: string, field: ConfigField, config: ConfigForm): boolean {
  if (sourceType === "confluence") {
    return isConfluenceFieldVisible(field.key, config);
  }
  if (sourceType === "jira") {
    const authMode = stringValue(config.auth_mode) || "browser_cookie";
    if (field.key === "jira_cookie") return false;
    if (field.key === "pat") return authMode === "pat";
    const queryMode = stringValue(config.query_mode) || "simple";
    if (field.key === "jql") return queryMode === "advanced";
    if (field.key === "projects" || field.key === "issue_types" || field.key === "jql_filter") {
      return queryMode === "simple";
    }
    return true;
  }
  if (sourceType === "github_pages") {
    const authMode = stringValue(config.auth_mode) || "github_pat";
    const syncMode = stringValue(config.sync_mode) || "single_page";
    if (field.field_type === "secret") return authMode !== "none";
    if (GITHUB_PAGES_MODE_FIELDS.has(field.key)) {
      return GITHUB_PAGES_VISIBILITY[syncMode]?.has(field.key) ?? false;
    }
    return true;
  }
  return true;
}

function isFieldRequired(sourceType: string, field: ConfigField, config: ConfigForm): boolean {
  if (sourceType === "confluence") {
    return field.required || isConfluenceFieldRequired(field.key, config);
  }
  if (sourceType === "jira") {
    const authMode = stringValue(config.auth_mode) || "browser_cookie";
    if (field.key === "jira_cookie") return false;
    if (field.key === "pat") return authMode === "pat";
    const queryMode = stringValue(config.query_mode) || "simple";
    if (field.key === "projects") return queryMode === "simple";
    if (field.key === "jql") return queryMode === "advanced";
  }
  if (sourceType === "github_pages") {
    const syncMode = stringValue(config.sync_mode) || "single_page";
    const authMode = stringValue(config.auth_mode) || "github_pat";
    if (field.key === "pat") return authMode === "github_pat";
    if (field.key === "page_url") return syncMode === "single_page";
    if (field.key === "root_url") return syncMode === "subtree";
    if (field.key === "pages") return syncMode === "explicit_list";
  }
  return field.required;
}

const GITHUB_PAGES_VISIBILITY: Record<string, Set<string>> = {
  single_page: new Set(["page_url"]),
  subtree: new Set(["root_url", "max_depth", "max_pages", "exclude_url_patterns"]),
  explicit_list: new Set(["pages", "max_pages", "exclude_url_patterns"]),
};

const GITHUB_PAGES_MODE_FIELDS = new Set<string>([
  "page_url",
  "root_url",
  "max_depth",
  "max_pages",
  "exclude_url_patterns",
  "pages",
]);

function optionLabel(field: ConfigField, option: string): string {
  if (field.key === "query_mode") {
    if (option === "simple") return "Simple (projects & issue types)";
    if (option === "advanced") return "Advanced (raw JQL)";
  }
  if (field.key === "auth_mode") {
    if (option === "browser_cookie") return "Browser session (local adapter)";
    if (option === "pat") return "Personal access token";
    if (option === "github_pat") return "Personal access token";
    if (option === "none") return "No authentication";
  }
  if (field.key === "sync_mode") {
    if (option === "page_tree") return "This page tree";
    if (option === "space") return "Whole space";
    if (option === "single_page") return "Single page";
    if (option === "subtree") return "Subtree";
    if (option === "explicit_list") return "Explicit list";
  }
  return option;
}

function optionValue(field: ConfigField, option: string): string {
  if (field.key === "auth_mode") {
    if (option === "browser_cookie") return "browser_session";
    if (option === "pat") return "personal_access_token";
  }
  return option;
}

function optionFromValue(field: ConfigField, value: string): string {
  if (field.key === "auth_mode") {
    if (value === "browser_session") return "browser_cookie";
    if (value === "personal_access_token") return "pat";
  }
  return value;
}

function listValue(value: ConfigValue | undefined): string[] {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return parseCommaList(value);
  return [];
}

function parseCommaList(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function stringValue(value: ConfigValue | undefined): string {
  if (Array.isArray(value)) return value.join(", ");
  if (value == null) return "";
  return String(value);
}

function parseScheduleInterval(value: string): number {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : 1440;
}

function scheduleIntervalLabel(value: string): string {
  switch (value) {
    case "30":
      return "Every 30 minutes";
    case "60":
      return "Hourly";
    case "360":
      return "Every 6 hours";
    case "720":
      return "Every 12 hours";
    case "10080":
      return "Weekly";
    case "1440":
    default:
      return "Daily";
  }
}

function extractSaveError(error: unknown): string {
  const fallback = "Failed to save source configuration.";
  if (!error) return fallback;
  if (typeof error === "object" && error !== null && "response" in error) {
    const response = (error as { response?: { data?: { detail?: unknown; error?: unknown } } }).response;
    const detail = response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      return String((detail as { message: unknown }).message);
    }
    if (typeof response?.data?.error === "string") return response.data.error;
  }
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

function toBoolean(value: ConfigValue | undefined): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value === "true";
  return Boolean(value);
}
