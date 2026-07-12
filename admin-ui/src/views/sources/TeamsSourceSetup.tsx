import { useCallback, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Check, Hash, Loader2, MessageSquare, Search, User, X } from "lucide-react";

import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getLocalAgentJob } from "@/api/localAgentJobs";
import type {
  GeneConfigSchema,
  LocalAgentJobStatusResponse,
  ProjectBinding,
  Source,
  TeamsAuthStatus,
  TeamsBrowseData,
  TeamsChannel,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { ProjectBindingFields } from "./ProjectBindingFields";
import { projectBindingIsComplete } from "./projectBinding";
import { SourceSetupShell } from "./SourceSetupShell";
import type { SourceSetupSection, SourceSetupSectionId } from "./SourceSetupShell";
import {
  buildDefaultTeamsSourceConfig,
  buildTeamsSourcePayload,
  buildTeamsSourceUpdatePayload,
  editableTeamsSourceState,
  existingTeamsSelection,
  teamsSelectionLabel,
  type TeamsSelectionItem,
  type TeamsSourceConfig,
} from "./teamsSourceConfig";

const TEAMS_JOB_POLL_ATTEMPTS = 120;
const TEAMS_JOB_POLL_INTERVAL_MS = 1_000;

export function TeamsSourceSetup({
  source,
  schema,
  onOpenChange,
  onSaved,
  initialFocus,
}: {
  source?: Source | null;
  schema: GeneConfigSchema;
  onOpenChange: (open: boolean) => void;
  onSaved?: () => void;
  initialFocus?: { step: "project" };
}) {
  const queryClient = useQueryClient();
  const isEdit = Boolean(source);
  const initial = useMemo(() => source ? editableTeamsSourceState(source) : null, [source]);
  const [config, setConfig] = useState<TeamsSourceConfig>(
    () => initial?.config ?? buildDefaultTeamsSourceConfig(),
  );
  const [selections, setSelections] = useState<Map<string, TeamsSelectionItem>>(
    () => new Map((initial?.conversationIds ?? []).map((id) => [id, existingTeamsSelection(id)])),
  );
  const [binding, setBinding] = useState<ProjectBinding | null>(() => source?.project_binding ?? null);
  const [scheduleEnabled, setScheduleEnabled] = useState(() => Boolean(source?.sync_schedule?.enabled));
  const [scheduleInterval, setScheduleInterval] = useState(
    () => String(source?.sync_schedule?.interval_minutes ?? 1440),
  );
  const [search, setSearch] = useState("");
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [focusSection, setFocusSection] = useState<SourceSetupSectionId>(
    initialFocus?.step === "project" ? "project" : "basics",
  );

  const authQuery = useQuery<TeamsAuthStatus>({
    queryKey: ["teams-auth-check", config.region],
    queryFn: () => runTeamsAuthCheck(config.region),
    retry: false,
  });
  const authenticated = authQuery.data?.authenticated === true;
  const browseQuery = useQuery<TeamsBrowseData>({
    queryKey: ["teams-browse", config.region],
    queryFn: () => runTeamsBrowse(config.region),
    enabled: authenticated,
    retry: false,
    staleTime: 60_000,
  });
  const connectTeams = useMutation({
    mutationFn: async () => {
      const status = await runTeamsLocalAgentJob("teams_auth", { region: config.region });
      if (status.status === "failed") throw new Error(teamsJobMessage(status));
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["teams-auth-check", config.region] });
      await authQuery.refetch();
    },
  });

  const items = useMemo(() => flattenTeamsData(browseQuery.data), [browseQuery.data]);
  const resolvedSelections = useMemo(() => {
    if (items.length === 0 || selections.size === 0) return selections;
    const browsed = new Map(items.map((item) => [item.id, item]));
    const next = new Map(selections);
    for (const [id, current] of selections) {
      const resolved = browsed.get(id);
      if (resolved && (resolved.type !== current.type || teamsSelectionLabel(resolved) !== teamsSelectionLabel(current))) {
        next.set(id, resolved);
      }
    }
    return next;
  }, [items, selections]);

  const visibleItems = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) => `${item.teamName ?? ""} ${item.displayName}`.toLowerCase().includes(needle));
  }, [items, search]);

  const toggleSelection = useCallback((item: TeamsSelectionItem) => {
    setValidationMessage(null);
    setSelections((current) => {
      const next = new Map(current);
      if (next.has(item.id)) next.delete(item.id);
      else next.set(item.id, item);
      return next;
    });
    setConfig((current) => current.name.trim()
      ? current
      : { ...current, name: `Teams - ${teamsSelectionLabel(item)}`.slice(0, 60) });
  }, []);

  const saveSource = useMutation({
    mutationFn: async () => {
      const selected = [...resolvedSelections.values()];
      const sourcePayload = source
        ? buildTeamsSourceUpdatePayload({ selections: selected, config })
        : buildTeamsSourcePayload({ selections: selected, config });
      const payload = {
        ...sourcePayload,
        project_binding: binding,
        sync_schedule: {
          enabled: scheduleEnabled,
          interval_minutes: Number(scheduleInterval),
        },
      };
      if (source) return resourceClient.put(`/sources/${source.id}`, payload);
      return resourceClient.post("/sources", payload);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["sources"] }),
        queryClient.invalidateQueries({ queryKey: ["projects"] }),
        queryClient.invalidateQueries({ queryKey: ["stats"] }),
      ]);
      onSaved?.();
      onOpenChange(false);
    },
  });

  const handleSave = () => {
    if (!config.name.trim()) {
      setValidationMessage("Enter a source name before saving.");
      setFocusSection("basics");
      return;
    }
    if (selections.size === 0) {
      setValidationMessage("Select at least one Teams conversation.");
      setFocusSection("content");
      return;
    }
    if (!projectBindingIsComplete(binding)) {
      setValidationMessage("Choose where this source should land, or leave it unmapped.");
      setFocusSection("project");
      return;
    }
    if (scheduleEnabled && Number(scheduleInterval) < 5) {
      setValidationMessage("Choose an automatic sync interval of at least 5 minutes.");
      setFocusSection("schedule");
      return;
    }
    saveSource.mutate();
  };

  const sessionError = connectTeams.error ?? authQuery.error;
  const selectedItems = [...resolvedSelections.values()];
  const sections: SourceSetupSection[] = [
    {
      id: "basics",
      title: "Basics",
      summary: config.name.trim() ? `Name · ${config.name.trim()}` : "Name this source",
      state: config.name.trim() ? "complete" : "incomplete",
      content: (
        <Field label="Source name" help="Use a name your workspace members will recognize.">
          <Input
            value={config.name}
            autoComplete="off"
            onChange={(event) => {
              setValidationMessage(null);
              setConfig({ ...config, name: event.target.value });
            }}
            placeholder="Teams - Engineering"
          />
        </Field>
      ),
    },
    {
      id: "connection",
      title: "Teams session",
      summary: authenticated ? "Signed in through the local sync app" : "Session needs attention",
      state: authenticated ? "complete" : "attention",
      content: (
        <div className="space-y-3">
          <div className={`flex items-start gap-3 rounded-lg border p-3 ${authenticated ? "bg-emerald-50/50" : "bg-amber-50/50"}`}>
            {authQuery.isFetching || connectTeams.isPending
              ? <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin" />
              : authenticated
                ? <Check className="mt-0.5 size-4 shrink-0 text-emerald-700" />
                : <AlertCircle className="mt-0.5 size-4 shrink-0 text-amber-700" />}
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium">{authenticated ? "Teams session active" : "Connect Microsoft Teams"}</div>
              <p className="mt-1 text-xs text-muted-foreground">
                {authenticated
                  ? "The local sync app can browse and sync Teams conversations."
                  : cleanTeamsAuthMessage(authQuery.data?.error) || "Sign in to Teams in Chrome, then connect through the local sync app."}
              </p>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={authQuery.isFetching || connectTeams.isPending}
              onClick={() => authenticated ? void authQuery.refetch() : connectTeams.mutate()}
            >
              {authenticated ? "Check again" : "Connect"}
            </Button>
          </div>
          <Field label="Teams region" help="Use the region assigned to your Microsoft 365 tenant.">
            <Select<string>
              value={config.region}
              onValueChange={(value) => value && setConfig({ ...config, region: value })}
            >
              <SelectTrigger><SelectValue>{config.region.toUpperCase()}</SelectValue></SelectTrigger>
              <SelectContent>
                <SelectItem value="emea">EMEA</SelectItem>
                <SelectItem value="amer">Americas</SelectItem>
                <SelectItem value="apac">Asia Pacific</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          {sessionError && <p className="text-xs text-destructive">{errorMessage(sessionError)}</p>}
        </div>
      ),
    },
    {
      id: "content",
      title: "Conversations to sync",
      summary: selections.size > 0
        ? `${selections.size} conversation${selections.size === 1 ? "" : "s"} selected`
        : "Select conversations",
      state: selections.size > 0 ? "complete" : "incomplete",
      content: (
        <div className="space-y-3">
          {selectedItems.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {selectedItems.map((item) => (
                <Badge key={item.id} variant="secondary" className="gap-1 pr-1">
                  <TeamsTypeIcon type={item.type} className="size-3" />
                  <span className="max-w-52 truncate text-xs">{teamsSelectionLabel(item)}</span>
                  <button type="button" className="rounded-full p-0.5 hover:bg-muted-foreground/20" onClick={() => toggleSelection(item)} aria-label={`Remove ${item.displayName}`}>
                    <X className="size-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )}
          {authenticated ? (
            <>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input value={search} onChange={(event) => setSearch(event.target.value)} className="pl-9" placeholder="Search Teams conversations..." />
              </div>
              {browseQuery.isLoading || browseQuery.isFetching ? (
                <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Loading conversations...</div>
              ) : browseQuery.isError ? (
                <div className="rounded-lg border p-4 text-center text-sm text-destructive">
                  Could not load conversations.
                  <Button type="button" variant="outline" size="sm" className="ml-3" onClick={() => void browseQuery.refetch()}>Retry</Button>
                </div>
              ) : (
                <div className="max-h-72 overflow-y-auto rounded-lg border bg-background">
                  {visibleItems.length === 0 ? (
                    <p className="p-6 text-center text-sm text-muted-foreground">No conversations found.</p>
                  ) : visibleItems.map((item) => (
                    <button key={item.id} type="button" className="flex w-full items-center gap-3 border-b px-3 py-2 text-left text-sm last:border-0 hover:bg-muted/60" onClick={() => toggleSelection(item)}>
                      <TeamsTypeIcon type={item.type} className="size-4 text-muted-foreground" />
                      <span className="min-w-0 flex-1">
                        {item.teamName && <span className="block truncate text-xs text-muted-foreground">{item.teamName}</span>}
                        <span className="block truncate font-medium">{item.displayName}</span>
                      </span>
                      {selections.has(item.id) && <Check className="size-4" />}
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <p className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">
              Connect the Teams session to browse conversations. Existing selections remain saved.
            </p>
          )}
          {selections.size > 0 && (
            <div className="rounded-lg border bg-emerald-50/40 p-3 text-xs text-emerald-800">
              Connection ready · {selections.size} conversation{selections.size === 1 ? "" : "s"} selected
            </div>
          )}
          <details className="border-t pt-3">
            <summary className="cursor-pointer text-sm font-semibold">Advanced settings</summary>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <NumberField label="History (days)" value={config.max_age_days} onChange={(value) => setConfig({ ...config, max_age_days: value })} />
              <NumberField label="Conversation gap (minutes)" value={config.conversation_gap_minutes} onChange={(value) => setConfig({ ...config, conversation_gap_minutes: value })} />
              <NumberField label="Maximum messages per group" value={config.max_block_messages} onChange={(value) => setConfig({ ...config, max_block_messages: value })} />
            </div>
          </details>
        </div>
      ),
    },
    {
      id: "project",
      title: "Save memories to",
      summary: binding ? binding.mode === "fixed" ? binding.project_key || "Choose a project" : "Mapped by conversation" : "Unmapped",
      state: projectBindingIsComplete(binding) ? "complete" : "incomplete",
      content: <ProjectBindingFields schema={schema} sourceId={source?.id ?? null} value={binding} onChange={setBinding} />,
    },
    {
      id: "schedule",
      title: "Automatic sync",
      summary: scheduleEnabled ? scheduleLabel(scheduleInterval) : "Manual only",
      state: !scheduleEnabled || Number(scheduleInterval) >= 5 ? "complete" : "incomplete",
      content: (
        <div className="space-y-3">
          <label className="flex items-start gap-3 rounded-lg border bg-background p-3">
            <input type="checkbox" className="mt-0.5 size-4" checked={scheduleEnabled} onChange={(event) => setScheduleEnabled(event.target.checked)} />
            <span><span className="block text-sm font-medium">Sync automatically</span><span className="mt-1 block text-xs text-muted-foreground">Manual sync remains available at any time.</span></span>
          </label>
          {scheduleEnabled && (
            <Select<string> value={scheduleInterval} onValueChange={(value) => value && setScheduleInterval(value)}>
              <SelectTrigger><SelectValue>{scheduleLabel(scheduleInterval)}</SelectValue></SelectTrigger>
              <SelectContent>
                <SelectItem value="30">Every 30 minutes</SelectItem><SelectItem value="60">Hourly</SelectItem><SelectItem value="360">Every 6 hours</SelectItem><SelectItem value="720">Every 12 hours</SelectItem><SelectItem value="1440">Daily</SelectItem><SelectItem value="10080">Weekly</SelectItem>
              </SelectContent>
            </Select>
          )}
        </div>
      ),
    },
  ];

  return (
    <SourceSetupShell
      sourceType="teams"
      sourceLabel="Microsoft Teams"
      sourceName={config.name}
      connection={{ mode: "device", label: "Local sync" }}
      isEdit={isEdit}
      sections={sections}
      openSection={focusSection}
      onOpenSectionChange={setFocusSection}
      error={saveSource.isError ? errorMessage(saveSource.error) : null}
      validationMessage={validationMessage}
      saving={saveSource.isPending}
      onCancel={() => onOpenChange(false)}
      onSave={handleSave}
    />
  );
}

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return <label className="block space-y-1.5"><span className="text-xs font-medium text-muted-foreground">{label}</span>{children}{help && <span className="block text-xs text-muted-foreground">{help}</span>}</label>;
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return <Field label={label}><Input type="number" min={1} value={value} onChange={(event) => onChange(Math.max(1, Number(event.target.value) || 1))} /></Field>;
}

function TeamsTypeIcon({ type, className }: { type: TeamsSelectionItem["type"]; className?: string }) {
  if (type === "channel") return <Hash className={className} />;
  if (type === "group_chat") return <MessageSquare className={className} />;
  if (type === "individual_chat") return <User className={className} />;
  return <MessageSquare className={className} />;
}

async function runTeamsAuthCheck(region: string): Promise<TeamsAuthStatus> {
  const status = await runTeamsLocalAgentJob("teams_auth_check", { region });
  const result = status.result as Partial<TeamsAuthStatus> | null;
  return {
    authenticated: result?.authenticated === true,
    expires_in_minutes: typeof result?.expires_in_minutes === "number" ? result.expires_in_minutes : null,
    error: typeof result?.error === "string" ? result.error : status.last_error ?? null,
  };
}

async function runTeamsBrowse(region: string): Promise<TeamsBrowseData> {
  const status = await runTeamsLocalAgentJob("teams_browse", { region });
  if (status.status === "failed") throw new Error(teamsJobMessage(status));
  const result = status.result as Partial<TeamsBrowseData> | null;
  return {
    favorites: Array.isArray(result?.favorites) ? result.favorites : [],
    teams: Array.isArray(result?.teams) ? result.teams : [],
    group_chats: Array.isArray(result?.group_chats) ? result.group_chats : [],
    individual_chats: Array.isArray(result?.individual_chats) ? result.individual_chats : [],
  };
}

async function runTeamsLocalAgentJob(operation: string, payload: Record<string, unknown>): Promise<LocalAgentJobStatusResponse> {
  const created = await createLocalAgentJob({ sourceType: "teams", operation, payload });
  for (let attempt = 0; attempt < TEAMS_JOB_POLL_ATTEMPTS; attempt += 1) {
    const status = await getLocalAgentJob(created.job_id);
    if (status.status === "succeeded" || status.status === "failed") return status;
    await new Promise((resolve) => window.setTimeout(resolve, TEAMS_JOB_POLL_INTERVAL_MS));
  }
  throw new Error("Timed out waiting for the local sync app.");
}

function flattenTeamsData(data: TeamsBrowseData | undefined): TeamsSelectionItem[] {
  if (!data) return [];
  const channels = data.teams.flatMap((team) => team.channels.map((channel: TeamsChannel) => ({ id: channel.id, displayName: channel.displayName, type: "channel" as const, teamName: team.displayName })));
  const groups = data.group_chats.map((chat) => ({ id: chat.id, displayName: chat.topic, type: "group_chat" as const }));
  const direct = data.individual_chats.map((chat) => ({ id: chat.id, displayName: chat.topic, type: "individual_chat" as const }));
  return [...channels, ...groups, ...direct];
}

function teamsJobMessage(status: LocalAgentJobStatusResponse): string {
  const result = status.result as { error?: unknown } | null;
  if (typeof result?.error === "string" && result.error.trim()) return result.error.trim();
  return status.last_error?.trim() || "Could not connect Microsoft Teams.";
}

function cleanTeamsAuthMessage(value: string | null | undefined): string | null {
  const text = value?.trim();
  if (!text) return null;
  const normalized = text.toLowerCase();
  if (normalized.includes("tokens") || normalized.includes("no teams session") || normalized.includes("session expired")) {
    return "No active Teams session found. Sign in to Teams in Chrome, then select Connect.";
  }
  return text;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "object" && error && "response" in error) {
    const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return "Could not save the Teams source.";
}

function scheduleLabel(value: string): string {
  return ({ "30": "Every 30 minutes", "60": "Hourly", "360": "Every 6 hours", "720": "Every 12 hours", "1440": "Daily", "10080": "Weekly" } as Record<string, string>)[value] ?? "Daily";
}
