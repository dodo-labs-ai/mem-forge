import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, ChevronRight, FolderOpen, Loader2, RefreshCw } from "lucide-react";
import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getLocalAgentJob } from "@/api/localAgentJobs";
import type {
  ConfigField,
  DiscoveryPreviewResponse,
  GeneConfigSchema,
  JiraAuthSession,
  LocalAgentJobStatusResponse,
  ProjectBinding,
  Source,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
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
  parseConfluenceWikiUrl,
} from "./confluenceConfig";
import type { ParsedConfluenceWikiUrl } from "./confluenceConfig";
import { GitHubRepoFolderPicker } from "./GitHubRepoFolderPicker";
import { isImmutableExecutionModeField } from "./localAgentSources";
import { ProjectBindingFields } from "./ProjectBindingFields";
import { projectBindingIsComplete } from "./projectBinding";
import { SourceSetupShell } from "./SourceSetupShell";
import type { SourceSetupSection, SourceSetupSectionId } from "./SourceSetupShell";
import {
  booleanValue,
  buildDefaultConfig,
  firstMissingRequiredField,
  isSchemaSourceType,
  listValue,
  optionLabel,
  serializeConfig,
  sourceSetupAdapterFor,
  stringValue,
  type ConfigForm,
  type ConfigValue,
  type SourceSetupAdapter,
} from "./sourceSetupAdapters";

const DISCOVERY_PREVIEW_LIMIT = 5;
const LOCAL_AGENT_PREVIEW_POLL_ATTEMPTS = 90;
const LOCAL_AGENT_PREVIEW_POLL_INTERVAL_MS = 2_000;

export function SchemaSourceSetup({
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
  // Backend authority: an existing source the viewer cannot configure should
  // not open the form. Type-level managed sources (agent_session) are also
  // blocked. New-source flows have no source row yet, so the type check is
  // the only gate there.
  const canConfigureExisting = source ? source.capabilities?.can_configure === true : true;
  const canConfigureConnection = source ? source.capabilities?.can_configure_connection === true : true;

  const schemaQuery = useQuery<GeneConfigSchema>({
    queryKey: ["gene-config-schema", sourceType],
    queryFn: () =>
      resourceClient.get(`/genes/${sourceType}/config-schema`).then((response) => response.data),
    enabled:
      open
      && Boolean(sourceType)
      && isSchemaSourceType(sourceType ?? "")
      && canConfigureExisting,
  });

  if (!sourceType || !isSchemaSourceType(sourceType)) return null;
  if (!canConfigureExisting) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden p-0 sm:max-w-3xl">
        {schemaQuery.isPending ? (
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
            canConfigureConnection={canConfigureConnection}
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
  canConfigureConnection,
}: {
  sourceType: string;
  source?: Source | null;
  schema: GeneConfigSchema;
  onOpenChange: (open: boolean) => void;
  onSaved?: () => void;
  initialFocus?: { step: "project" };
  canConfigureConnection: boolean;
}) {
  const queryClient = useQueryClient();
  const isEdit = Boolean(source);
  const adapter = sourceSetupAdapterFor(sourceType);
  const [name, setName] = useState(source?.name ?? "");
  const [config, setConfig] = useState<ConfigForm>(() => ({
    ...buildDefaultConfig(schema),
    ...adapter.normalizeInitialConfig((source?.config ?? {}) as ConfigForm),
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
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [focusSection, setFocusSection] = useState<SourceSetupSectionId>(
    initialFocus?.step === "project" ? "project" : isEdit ? "basics" : "basics",
  );
  const authMode = stringValue(config.auth_mode) || "browser_cookie";
  const githubConnectionMode = sourceType === "github_repo"
    ? stringValue(config.connection_mode) || "cloud_pull"
    : "";
  const githubRepoUrl = sourceType === "github_repo" ? stringValue(config.repo_url).trim() : "";
  const githubPickerConfig = sourceType === "github_repo" ? serializeConfig(schema.fields, config) : {};
  const jiraBaseUrl = stringValue(config.base_url).trim();
  const confluenceUrlInfo = useMemo(
    () => sourceType === "confluence" ? parseConfluenceWikiUrl(stringValue(config.base_url)) : null,
    [config.base_url, sourceType],
  );

  const jiraSessionQuery = useQuery<JiraAuthSession>({
    queryKey: ["jira-session", jiraBaseUrl],
    queryFn: () =>
      resourceClient.get("/auth/jira-session", { params: { base_url: jiraBaseUrl } }).then((response) => response.data),
    enabled: canConfigureConnection
      && sourceType === "jira"
      && authMode === "browser_cookie"
      && jiraBaseUrl.startsWith("https://"),
  });

  const saveSource = useMutation({
    mutationFn: async (payload: {
      name: string;
      config?: ConfigForm;
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
        await resourceClient.put(`/sources/${source.id}`, payloadWithSchedule);
        return { id: source.id };
      }
      const response = await resourceClient.post("/sources", {
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
    mutationFn: async () => {
      const serializedConfig = serializeConfig(schema.fields, config);
      if (sourceType === "local_markdown") {
        const created = await createLocalAgentJob({
          sourceId: source?.id ?? "",
          sourceType: "local_markdown",
          operation: "local_markdown_preview_tree",
          payload: {
            ...localMarkdownPreviewJobConfig(serializedConfig, source),
            limit: DISCOVERY_PREVIEW_LIMIT,
          },
        });
        const status = await pollLocalAgentPreviewJob(created.job_id);
        if (status.status === "failed") {
          throw new Error(status.last_error || "Local daemon could not preview this folder.");
        }
        return localMarkdownPreviewFromJob(status);
      }
      return resourceClient
        .post(`/genes/${sourceType}/preview-discovery`, {
          config: serializedConfig,
          limit: DISCOVERY_PREVIEW_LIMIT,
        })
        .then((response) => response.data as DiscoveryPreviewResponse);
    },
  });
  const pickLocalMarkdownRoot = useMutation<string | null, unknown, void>({
    mutationFn: async () => {
      const created = await createLocalAgentJob({
        sourceId: source?.id ?? "",
        sourceType: "local_markdown",
        operation: "local_markdown_pick_root",
        payload: {
          title: "Choose folder to sync",
          initial_directory: stringValue(config.root).trim() || undefined,
        },
      });
      const status = await pollLocalAgentPreviewJob(created.job_id);
      if (status.status === "failed") {
        throw new Error(status.last_error || "Local daemon could not open the folder picker.");
      }
      const result = status.result as { cancelled?: unknown; root?: unknown } | null;
      if (result?.cancelled === true) {
        return null;
      }
      const root = typeof result?.root === "string" ? result.root.trim() : "";
      if (!root) {
        throw new Error("Local daemon did not return a folder path.");
      }
      return root;
    },
    onSuccess: (root) => {
      if (root === null) return;
      setValidationMessage(null);
      setConfig((current) => ({ ...current, root }));
    },
  });
  const pickGitHubRepoPath = useMutation<string | null, unknown, void>({
    mutationFn: async () => {
      const created = await createLocalAgentJob({
        sourceId: source?.id ?? "",
        sourceType: "github_repo",
        operation: "github_repo_pick_root",
        payload: {
          title: "Choose local repository clone",
          initial_directory: stringValue(config.repo_path).trim() || undefined,
        },
      });
      const status = await pollLocalAgentPreviewJob(created.job_id);
      if (status.status === "failed") {
        throw new Error(status.last_error || "Local daemon could not open the folder picker.");
      }
      const result = status.result as { cancelled?: unknown; root?: unknown } | null;
      if (result?.cancelled === true) {
        return null;
      }
      const root = typeof result?.root === "string" ? result.root.trim() : "";
      if (!root) {
        throw new Error("Local daemon did not return a folder path.");
      }
      return root;
    },
    onSuccess: (repoPath) => {
      if (repoPath === null) return;
      setValidationMessage(null);
      setConfig((current) => ({ ...current, repo_path: repoPath }));
    },
  });

  const sortedFields = useMemo(
    () => [...schema.fields].sort((a, b) => a.order - b.order),
    [schema.fields],
  );
  const connectionFields = sortedFields.filter(
    (field) => adapter.sectionForField(field, config) === "connection",
  );
  const contentFields = sortedFields.filter(
    (field) => adapter.sectionForField(field, config) === "content",
  );
  const advancedFields = sortedFields.filter(
    (field) => adapter.sectionForField(field, config) === "advanced",
  );
  const firstMissingField = firstMissingRequiredField(adapter, schema.fields, config);
  const connectionMissing = firstMissingRequiredField(adapter, connectionFields, config);
  const contentMissing = firstMissingRequiredField(adapter, [...contentFields, ...advancedFields], config);
  const scheduleIntervalValid = parseScheduleInterval(scheduleInterval) >= 5;

  const previewReady = firstMissingField === null;
  const showDiscoveryPreview = sourceType !== "github_repo"
    && !(sourceType === "jira" && stringValue(config.sync_mode) === "local_agent");
  const scheduleSummary = scheduleEnabled
    ? scheduleIntervalLabel(scheduleInterval)
    : "Manual only";

  const updateField = (field: ConfigField, value: ConfigValue) => {
    setValidationMessage(null);
    setConfig((current) => adapter.normalizeFieldChange(field, value, current));
  };

  const handleSave = () => {
    if (name.trim().length === 0) {
      setValidationMessage("Enter a source name before saving.");
      setFocusSection("basics");
      return;
    }
    if (canConfigureConnection && firstMissingField) {
      setValidationMessage(`Complete ${firstMissingField.label} before saving.`);
      const section = adapter.sectionForField(firstMissingField, config);
      setFocusSection(section === "connection" ? "connection" : "content");
      return;
    }
    if (!projectBindingIsComplete(binding)) {
      setValidationMessage("Choose where this source should land, or leave it unmapped.");
      setFocusSection("project");
      return;
    }
    if (scheduleEnabled && !scheduleIntervalValid) {
      setValidationMessage("Choose an automatic sync interval of at least 5 minutes.");
      setFocusSection("schedule");
      return;
    }
    saveSource.mutate({
      name: name.trim(),
      ...(canConfigureConnection ? { config: serializeConfig(schema.fields, config) } : {}),
      project_binding: binding,
    });
  };

  const renderField = (field: ConfigField) => {
    if (sourceType === "local_markdown" && field.key === "root") {
      return (
        <FolderSelectionField
          key={field.key}
          label="Root folder"
          path={stringValue(config.root)}
          emptyLabel="No folder selected"
          isPending={pickLocalMarkdownRoot.isPending}
          error={pickLocalMarkdownRoot.isError ? pickLocalMarkdownRoot.error : null}
          onChoose={() => pickLocalMarkdownRoot.mutate()}
        />
      );
    }
    if (sourceType === "github_repo" && field.key === "repo_path") {
      return (
        <FolderSelectionField
          key={field.key}
          label="Local repository clone"
          path={stringValue(config.repo_path)}
          emptyLabel="No folder selected"
          isPending={pickGitHubRepoPath.isPending}
          error={pickGitHubRepoPath.isError ? pickGitHubRepoPath.error : null}
          onChoose={() => pickGitHubRepoPath.mutate()}
        />
      );
    }
    return (
      <div key={field.key} className="space-y-3">
        <ConfigFieldInput
          adapter={adapter}
          field={field}
          value={config[field.key]}
          hasExistingSecret={Boolean(config[`${field.key}_configured`])}
          decryptFailed={Boolean(config[`${field.key}_decrypt_failed`])}
          required={adapter.isRequired(field, config)}
          disabled={source ? isImmutableExecutionModeField(source, field.key) : false}
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
            onRefresh={() => void jiraSessionQuery.refetch()}
          />
        )}
        {sourceType === "github_repo" && field.key === "ref" && (
          <GitHubRepoFolderPicker
            connectionMode={githubConnectionMode}
            config={githubPickerConfig}
            value={listValue(config.include_paths)}
            onChange={(paths) => {
              setValidationMessage(null);
              setConfig((current) => ({ ...current, include_paths: paths }));
            }}
          />
        )}
        {sourceType === "github_repo" && field.key === "repo_url" && (
          <GitHubRepoDetectedPanel repoUrl={githubRepoUrl} />
        )}
      </div>
    );
  };

  const unavailableConnection = (
    <p className="text-sm text-muted-foreground">
      Connection and content settings are managed by the source owner. You can still change its project and sync frequency.
    </p>
  );
  const contentBody = canConfigureConnection ? (
    <>
      {contentFields.map(renderField)}
      {showDiscoveryPreview && (
        <VerifySetupPanel
          ready={previewReady}
          isPending={previewDiscovery.isPending}
          error={previewDiscovery.isError ? previewDiscovery.error : null}
          data={previewDiscovery.data}
          onVerify={() => previewDiscovery.mutate()}
        />
      )}
      {advancedFields.length > 0 && (
        <details className="group space-y-3 border-t pt-3">
          <summary className="inline-flex cursor-pointer select-none items-center gap-1 rounded-md px-1 py-0.5 text-sm font-semibold hover:bg-muted focus:outline-none focus-visible:ring-1 focus-visible:ring-ring/40">
            <ChevronRight className="size-4 transition-transform group-open:rotate-90" />
            Advanced settings
          </summary>
          <div className="space-y-3 pt-2">{advancedFields.map(renderField)}</div>
        </details>
      )}
    </>
  ) : unavailableConnection;

  const sections: SourceSetupSection[] = [
    {
      id: "basics",
      title: "Basics",
      summary: name.trim() ? `Name · ${name.trim()}` : "Name this source",
      state: name.trim() ? "complete" : "incomplete",
      content: (
        <Field label="Source name" required helpText="Use a name your workspace members will recognize.">
          <Input
            value={name}
            autoComplete="off"
            onChange={(event) => {
              setValidationMessage(null);
              setName(event.target.value);
            }}
            placeholder="Source name"
          />
        </Field>
      ),
    },
    {
      id: "connection",
      title: adapter.connectionTitle,
      summary: canConfigureConnection ? adapter.connectionSummary(config) : "Managed by source owner",
      state: canConfigureConnection && connectionMissing ? "incomplete" : "complete",
      content: canConfigureConnection ? <>{connectionFields.map(renderField)}</> : unavailableConnection,
    },
    {
      id: "content",
      title: adapter.contentTitle,
      summary: canConfigureConnection ? adapter.contentSummary(config) : "Managed by source owner",
      state: canConfigureConnection && contentMissing ? "incomplete" : "complete",
      content: contentBody,
    },
    {
      id: "project",
      title: "Save memories to",
      summary: projectBindingSummary(binding),
      state: projectBindingIsComplete(binding) ? "complete" : "incomplete",
      content: (
        <ProjectBindingFields
          schema={schema}
          sourceId={source?.id ?? null}
          value={binding}
          onChange={(nextBinding) => {
            setValidationMessage(null);
            setBinding(nextBinding);
          }}
        />
      ),
    },
    {
      id: "schedule",
      title: "Automatic sync",
      summary: scheduleSummary,
      state: !scheduleEnabled || scheduleIntervalValid ? "complete" : "incomplete",
      content: (
        <>
          <label className="flex items-start gap-3 rounded-lg border bg-background p-3">
            <input
              type="checkbox"
              className="mt-0.5 size-4"
              checked={scheduleEnabled}
              onChange={(event) => {
                setValidationMessage(null);
                setScheduleEnabled(event.target.checked);
              }}
            />
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium">Sync automatically</span>
              <span className="mt-1 block text-xs text-muted-foreground">Manual sync remains available at any time.</span>
            </span>
          </label>
          {scheduleEnabled && (
            <Field label="Frequency">
              <Select<string>
                value={scheduleInterval}
                onValueChange={(value) => {
                  if (!value) return;
                  setValidationMessage(null);
                  setScheduleInterval(value);
                }}
              >
                <SelectTrigger><SelectValue>{scheduleIntervalLabel(scheduleInterval)}</SelectValue></SelectTrigger>
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
          )}
        </>
      ),
    },
  ];

  return (
    <SourceSetupShell
      sourceType={sourceType}
      sourceLabel={adapter.displayName}
      sourceName={name}
      connection={adapter.connection}
      isEdit={isEdit}
      sections={sections}
      openSection={focusSection}
      onOpenSectionChange={setFocusSection}
      error={saveSource.isError ? extractSaveError(saveSource.error) : null}
      validationMessage={validationMessage}
      saving={saveSource.isPending}
      onCancel={() => onOpenChange(false)}
      onSave={handleSave}
    />
  );
}

function ConfigFieldInput({
  adapter,
  field,
  value,
  hasExistingSecret,
  decryptFailed,
  required,
  disabled,
  onChange,
}: {
  adapter: SourceSetupAdapter;
  field: ConfigField;
  value: ConfigValue | undefined;
  hasExistingSecret?: boolean;
  decryptFailed?: boolean;
  required?: boolean;
  disabled?: boolean;
  onChange: (value: ConfigValue) => void;
}) {
  if (field.field_type === "boolean") {
    return (
      <label className="flex items-start gap-3 rounded-lg border p-3">
        <input
          type="checkbox"
          className="mt-0.5 size-4"
          checked={booleanValue(value)}
          disabled={disabled}
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
          value={selected}
          disabled={disabled}
          onValueChange={(next) => onChange(stringValue(next))}
        >
          <SelectTrigger>
            <SelectValue>{selected ? optionLabel(adapter, field, selected) : "Select..."}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {field.options.map((option) => (
              <SelectItem key={option} value={option}>
                {optionLabel(adapter, field, option)}
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
                disabled={disabled}
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
          disabled={disabled}
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
        autoComplete={isSecret ? "new-password" : "off"}
        name={`source-${field.key}`}
        value={isList ? listValue(value).join(", ") : stringValue(value)}
        disabled={disabled}
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

function parseGitHubRepoUrl(repoUrl: string): { host: string; owner: string; repo: string; normalized: string } | null {
  const value = repoUrl.trim();
  if (!value.startsWith("https://")) return null;
  try {
    const url = new URL(value);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    const owner = parts[0];
    const repo = parts[1].replace(/\.git$/, "");
    return {
      host: url.host.toLowerCase(),
      owner,
      repo,
      normalized: `https://${url.host.toLowerCase()}/${owner}/${repo}`,
    };
  } catch {
    return null;
  }
}

function FolderSelectionField({
  label,
  path,
  emptyLabel,
  isPending,
  error,
  onChoose,
}: {
  label: string;
  path: string;
  emptyLabel: string;
  isPending: boolean;
  error: unknown;
  onChoose: () => void;
}) {
  return (
    <div className="overflow-hidden rounded-lg border bg-background">
      <div className="flex flex-wrap items-center justify-between gap-3 p-3">
        <div>
          <div className="text-sm font-medium">{label}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">Selected through the local sync app.</div>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={onChoose} disabled={isPending}>
          {isPending ? <Loader2 className="size-3.5 animate-spin" /> : <FolderOpen className="size-3.5" />}
          Choose folder
        </Button>
      </div>
      <div className="border-t bg-muted/30 px-3 py-2 font-mono text-xs">
        {path || <span className="font-sans text-muted-foreground">{emptyLabel}</span>}
      </div>
      {error ? <div className="border-t px-3 py-2 text-xs text-destructive">{extractSaveError(error)}</div> : null}
    </div>
  );
}

function GitHubRepoDetectedPanel({ repoUrl }: { repoUrl: string }) {
  const parsed = parseGitHubRepoUrl(repoUrl);
  if (!parsed) return null;
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs">
      <div className="font-medium">Detected repository</div>
      <dl className="mt-1 grid gap-1 sm:grid-cols-2">
        <div className="min-w-0">
          <dt className="text-muted-foreground">Host</dt>
          <dd className="truncate font-medium">{parsed.host}</dd>
        </div>
        <div className="min-w-0">
          <dt className="text-muted-foreground">Repository</dt>
          <dd className="truncate font-medium">{parsed.owner}/{parsed.repo}</dd>
        </div>
      </dl>
    </div>
  );
}

function VerifySetupPanel({
  ready,
  isPending,
  error,
  data,
  onVerify,
}: {
  ready: boolean;
  isPending: boolean;
  error: unknown;
  data: DiscoveryPreviewResponse | undefined;
  onVerify: () => void;
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Verify setup</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">Checks the connection and returns a small content sample.</p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onVerify}
          disabled={!ready || isPending}
        >
          {isPending ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {data ? "Verify again" : "Verify setup"}
        </Button>
      </div>
      {!ready ? (
        <p className="text-xs text-muted-foreground">
          Complete the required connection and content settings before verifying.
        </p>
      ) : !data && !error && !isPending ? (
        <p className="text-xs text-muted-foreground">
          Confirm that MemForge can reach the source and see the intended content.
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

function localMarkdownPreviewJobConfig(config: ConfigForm, source?: Source | null): ConfigForm {
  const sourceVaultId = source?.config?.vault_id;
  return {
    ...config,
    vault_id: (typeof sourceVaultId === "string" ? sourceVaultId.trim() : "") || "preview",
  };
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

function projectBindingSummary(binding: ProjectBinding | null): string {
  if (!binding) return "Unmapped";
  if (binding.mode === "fixed") return binding.project_key || "Choose a project";
  const mapped = Object.keys(binding.map ?? {}).length;
  return `${mapped} mapped value${mapped === 1 ? "" : "s"} · default ${binding.default || "not selected"}`;
}

async function pollLocalAgentPreviewJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < LOCAL_AGENT_PREVIEW_POLL_ATTEMPTS; attempt += 1) {
    const status = await getLocalAgentJob(jobId);
    if (status.status === "succeeded" || status.status === "failed") {
      return status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, LOCAL_AGENT_PREVIEW_POLL_INTERVAL_MS));
  }
  throw new Error("Timed out waiting for the local daemon.");
}

function localMarkdownPreviewFromJob(status: LocalAgentJobStatusResponse): DiscoveryPreviewResponse {
  const result = status.result as {
    counts?: { included?: number };
    items?: Array<{ relative_path?: string; title?: string }>;
  } | null;
  const items = result?.items ?? [];
  return {
    source_type: "local_markdown",
    count: Number(result?.counts?.included ?? items.length),
    truncated: false,
    items: items.map((item) => {
      const path = item.relative_path ?? "";
      return {
        item_id: path,
        title: item.title || path,
        source_url: `local-agent://${path}`,
        last_modified: null,
      };
    }),
  };
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
