import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Check,
  Hash,
  Loader2,
  MessageSquare,
  Search,
  User,
  X,
} from "lucide-react";
import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getLocalAgentJob } from "@/api/localAgentJobs";
import type {
  LocalAgentJobStatusResponse,
  TeamsAuthStatus,
  TeamsBrowseData,
  TeamsChannel,
  Source,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
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

const TEAMS_AUTH_POLL_ATTEMPTS = 120;
const TEAMS_AUTH_POLL_INTERVAL_MS = 1_000;

export function TeamsSourceWizard({
  open,
  onOpenChange,
  onCreated,
  source,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
  source?: Source | null;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden p-0 sm:max-w-2xl">
        {open && (
          <TeamsSourceWizardBody
            onOpenChange={onOpenChange}
            onCreated={onCreated}
            source={source}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function TeamsSourceWizardBody({
  onOpenChange,
  onCreated,
  source,
}: {
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
  source?: Source | null;
}) {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const initialState = useMemo(
    () => source ? editableTeamsSourceState(source) : null,
    [source],
  );
  const [selections, setSelections] = useState<Map<string, TeamsSelectionItem>>(
    () => new Map(
      (initialState?.conversationIds ?? []).map((id) => [id, existingTeamsSelection(id)]),
    ),
  );
  const [config, setConfig] = useState(
    () => initialState?.config ?? buildDefaultTeamsSourceConfig(),
  );

  return (
    <>
      {step === 0 && <AuthCheckStep region={config.region} onAuthenticated={() => setStep(1)} />}
      {step === 1 && (
        <BrowseSelectStep
          region={config.region}
          selections={selections}
          onSelectionsChange={setSelections}
          onBack={() => onOpenChange(false)}
          onNext={() => {
            const names = [...selections.values()]
              .slice(0, 3)
              .map((item) => item.displayName);
            const suffix = selections.size > 3 ? ` +${selections.size - 3} more` : "";
            setConfig((current) => ({
              ...current,
              name: current.name || `Teams - ${names.join(", ")}${suffix}`.slice(0, 60),
            }));
            setStep(2);
          }}
        />
      )}
      {step === 2 && (
        <ConfirmStep
          selections={selections}
          config={config}
          onConfigChange={setConfig}
          onBack={() => setStep(1)}
          onCreated={onCreated}
          source={source}
        />
      )}
    </>
  );
}

function AuthCheckStep({ region, onAuthenticated }: { region: string; onAuthenticated: () => void }) {
  const [message, setMessage] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { data, isLoading, isFetching, refetch } = useQuery<TeamsAuthStatus>({
    queryKey: ["teams-auth-check", region],
    queryFn: () => runTeamsAuthCheck(region),
    retry: false,
  });
  const connectMutation = useMutation({
    mutationFn: async () => {
      setMessage(null);
      const status = await runTeamsLocalAgentJob("teams_auth", { region });
      if (status.status === "failed") {
        throw new Error(teamsAuthJobMessage(status));
      }
      return status;
    },
    onSuccess: async () => {
      setMessage("Connected. Loading conversations...");
      await queryClient.invalidateQueries({ queryKey: ["teams-auth-check", region] });
      onAuthenticated();
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : "Could not connect Teams.");
    },
  });

  useEffect(() => {
    if (data?.authenticated) onAuthenticated();
  }, [data?.authenticated, onAuthenticated]);

  const busy = isFetching || connectMutation.isPending;
  const statusMessage = message || cleanTeamsAuthMessage(data?.error) || "Teams session required.";

  return (
    <>
      <div className="p-4">
        <DialogHeader>
          <DialogTitle>Connect Microsoft Teams</DialogTitle>
        </DialogHeader>

        {isLoading || isFetching ? (
          <div className="flex items-center justify-center gap-3 py-12 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Checking authentication...
          </div>
        ) : data?.authenticated ? (
          <div className="flex items-center justify-center gap-3 py-12 text-sm">
            <Check className="size-4 text-emerald-600" />
            Authenticated. Loading conversations...
          </div>
        ) : (
          <div className="space-y-4 pt-4">
            <div className="flex items-start gap-3 rounded-lg bg-muted p-3 text-sm">
              <AlertCircle className="mt-0.5 size-4 shrink-0 text-amber-600" />
              <span>{statusMessage}</span>
            </div>
          </div>
        )}
      </div>

      {!data?.authenticated && !isLoading && (
        <DialogFooter className="mx-0 mb-0 shrink-0 p-5 rounded-none rounded-b-xl bg-background">
          <Button type="button" variant="outline" onClick={() => refetch()} disabled={busy}>
            {isFetching && !connectMutation.isPending && <Loader2 className="size-4 animate-spin" />}
            Check Again
          </Button>
          <Button type="button" onClick={() => connectMutation.mutate()} disabled={busy}>
            {connectMutation.isPending && <Loader2 className="size-4 animate-spin" />}
            Connect
          </Button>
        </DialogFooter>
      )}
    </>
  );
}

async function runTeamsAuthCheck(region: string): Promise<TeamsAuthStatus> {
  const status = await runTeamsLocalAgentJob("teams_auth_check", { region });
  return teamsAuthStatusFromJob(status);
}

async function runTeamsLocalAgentJob(operation: string, payload: Record<string, unknown>): Promise<LocalAgentJobStatusResponse> {
  const created = await createLocalAgentJob({
    sourceType: "teams",
    operation,
    payload,
  });
  return pollTeamsLocalAgentJob(created.job_id);
}

async function pollTeamsLocalAgentJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < TEAMS_AUTH_POLL_ATTEMPTS; attempt += 1) {
    const status = await getLocalAgentJob(jobId);
    if (status.status === "succeeded" || status.status === "failed") {
      return status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, TEAMS_AUTH_POLL_INTERVAL_MS));
  }
  throw new Error("Timed out waiting for the local daemon.");
}

function teamsAuthStatusFromJob(status: LocalAgentJobStatusResponse): TeamsAuthStatus {
  const result = status.result as Partial<TeamsAuthStatus> | null;
  return {
    authenticated: result?.authenticated === true,
    expires_in_minutes: typeof result?.expires_in_minutes === "number" ? result.expires_in_minutes : null,
    error: typeof result?.error === "string" ? result.error : status.last_error ?? null,
  };
}

function teamsAuthJobMessage(status: LocalAgentJobStatusResponse): string {
  const result = status.result as { error?: unknown } | null;
  if (typeof result?.error === "string" && result.error.trim()) {
    return result.error.trim();
  }
  if (status.last_error?.trim()) {
    return cleanTeamsAuthMessage(status.last_error) || "Could not connect Teams.";
  }
  return "Could not connect Teams.";
}

function cleanTeamsAuthMessage(value: string | null | undefined): string | null {
  const text = value?.trim();
  if (!text) return null;
  const normalized = text.toLowerCase();
  if (
    normalized.includes("tokens")
    || normalized.includes("run:")
    || normalized.includes("no teams session")
    || normalized.includes("session expired")
  ) {
    return "No Teams session found. Select Connect after signing in to Teams in Chrome.";
  }
  return text;
}

function BrowseSelectStep({
  region,
  selections,
  onSelectionsChange,
  onBack,
  onNext,
}: {
  region: string;
  selections: Map<string, TeamsSelectionItem>;
  onSelectionsChange: (selections: Map<string, TeamsSelectionItem>) => void;
  onBack: () => void;
  onNext: () => void;
}) {
  const [search, setSearch] = useState("");
  const { data, isLoading, isError, refetch } = useQuery<TeamsBrowseData>({
    queryKey: ["teams-browse", region],
    queryFn: () => runTeamsBrowse(region),
    retry: false,
    staleTime: 60_000,
  });

  const toggleSelection = useCallback(
    (item: TeamsSelectionItem) => {
      const next = new Map(selections);
      if (next.has(item.id)) {
        next.delete(item.id);
      } else {
        next.set(item.id, item);
      }
      onSelectionsChange(next);
    },
    [onSelectionsChange, selections],
  );

  const items = useMemo(() => flattenTeamsData(data), [data]);
  useEffect(() => {
    if (items.length === 0 || selections.size === 0) return;
    const browsedById = new Map(items.map((item) => [item.id, item]));
    const next = new Map(selections);
    let changed = false;
    for (const [id, selected] of selections) {
      const browsed = browsedById.get(id);
      if (
        browsed
        && (
          browsed.type !== selected.type
          || teamsSelectionLabel(browsed) !== teamsSelectionLabel(selected)
        )
      ) {
        next.set(id, browsed);
        changed = true;
      }
    }
    if (changed) onSelectionsChange(next);
  }, [items, onSelectionsChange, selections]);
  const visibleItems = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) =>
      `${item.teamName ?? ""} ${item.displayName}`.toLowerCase().includes(needle),
    );
  }, [items, search]);

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-4 p-4">
        <DialogHeader>
          <DialogTitle>Select Teams conversations to sync</DialogTitle>
        </DialogHeader>

        {selections.size > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {[...selections.values()].map((item) => (
              <Badge key={item.id} variant="secondary" className="gap-1 pr-1">
                <TypeIcon type={item.type} className="size-3" />
                <span className="max-w-[180px] truncate text-xs">{teamsSelectionLabel(item)}</span>
                <button
                  type="button"
                  onClick={() => toggleSelection(item)}
                  className="rounded-full p-0.5 hover:bg-muted-foreground/20"
                  aria-label={`Remove ${item.displayName}`}
                >
                  <X className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}

        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={search} onChange={(event) => setSearch(event.target.value)} className="pl-9" placeholder="Search Teams conversations..." />
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center gap-3 py-12 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading Teams conversations...
          </div>
        ) : isError ? (
          <div className="rounded-lg border p-6 text-center">
            <p className="text-sm text-destructive">Failed to load conversations.</p>
            <Button type="button" variant="outline" size="sm" className="mt-3" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        ) : visibleItems.length === 0 ? (
          <div className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
            No conversations found.
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border">
            {visibleItems.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => toggleSelection(item)}
                className="flex w-full items-center gap-3 border-b px-3 py-2 text-left text-sm last:border-0 hover:bg-muted/60"
              >
                <TypeIcon type={item.type} className="size-4 text-muted-foreground" />
                <span className="min-w-0 flex-1">
                  {item.teamName && (
                    <span className="block truncate text-xs text-muted-foreground">{item.teamName}</span>
                  )}
                  <span className="block truncate font-medium">{item.displayName}</span>
                </span>
                {selections.has(item.id) && <Check className="size-4 text-foreground" />}
              </button>
            ))}
          </div>
        )}
      </div>

      <DialogFooter className="mx-0 mb-0 shrink-0 p-5 flex-row justify-between rounded-none rounded-b-xl bg-background sm:justify-between">
        <Button type="button" variant="ghost" onClick={onBack}>
          <ArrowLeft className="size-4" />
          Cancel
        </Button>
        <Button type="button" onClick={onNext} disabled={selections.size === 0}>
          Next
          <ArrowRight className="size-4" />
          {selections.size > 0 && <Badge variant="secondary">{selections.size}</Badge>}
        </Button>
      </DialogFooter>
    </>
  );
}

function ConfirmStep({
  selections,
  config,
  onConfigChange,
  onBack,
  onCreated,
  source,
}: {
  selections: Map<string, TeamsSelectionItem>;
  config: TeamsSourceConfig;
  onConfigChange: (config: TeamsSourceConfig) => void;
  onBack: () => void;
  onCreated: () => void;
  source?: Source | null;
}) {
  const queryClient = useQueryClient();
  const selectedItems = [...selections.values()];
  const saveSource = useMutation({
    mutationFn: async () => {
      if (source) {
        return resourceClient.put(`/sources/${source.id}`, buildTeamsSourceUpdatePayload({
          selections: selectedItems,
          config,
          existingConfig: source.config,
        }));
      }
      return resourceClient.post("/sources", buildTeamsSourcePayload({ selections: selectedItems, config }));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      onCreated();
    },
  });

  const updateNumber = (field: keyof TeamsSourceConfig, fallback: number) => (value: string) => {
    onConfigChange({ ...config, [field]: Number(value) || fallback });
  };

  return (
    <>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <DialogHeader>
          <DialogTitle>Configure Teams source</DialogTitle>
        </DialogHeader>

        <Field label="Source name">
          <Input
            value={config.name}
            onChange={(event) => onConfigChange({ ...config, name: event.target.value })}
            placeholder="Teams - Engineering"
          />
        </Field>

        <Field label={`Selected Teams conversations (${selectedItems.length})`}>
          <div className="flex flex-wrap gap-1.5">
            {selectedItems.map((item) => (
              <Badge key={item.id} variant="secondary" className="gap-1">
                <TypeIcon type={item.type} className="size-3" />
                {teamsSelectionLabel(item)}
              </Badge>
            ))}
          </div>
        </Field>

        <Separator />

        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="History (days)">
            <Input type="number" value={config.max_age_days} onChange={(event) => updateNumber("max_age_days", 14)(event.target.value)} />
          </Field>
          <Field label="Gap (minutes)">
            <Input type="number" value={config.conversation_gap_minutes} onChange={(event) => updateNumber("conversation_gap_minutes", 60)(event.target.value)} />
          </Field>
          <Field label="Max messages/block">
            <Input type="number" value={config.max_block_messages} onChange={(event) => updateNumber("max_block_messages", 100)(event.target.value)} />
          </Field>
        </div>

        {saveSource.isError && (
          <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            Failed to {source ? "update" : "create"} source. Please try again.
          </div>
        )}
      </div>

      <DialogFooter className="mx-0 mb-0 shrink-0 p-5 flex-row justify-between rounded-none rounded-b-xl bg-background sm:justify-between">
        <Button type="button" variant="ghost" onClick={onBack}>
          <ArrowLeft className="size-4" />
          Back
        </Button>
        <Button type="button" onClick={() => saveSource.mutate()} disabled={!config.name.trim() || saveSource.isPending}>
          {saveSource.isPending ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
          {source ? "Save Changes" : "Create Source"}
        </Button>
      </DialogFooter>
    </>
  );
}

async function runTeamsBrowse(region: string): Promise<TeamsBrowseData> {
  const status = await runTeamsLocalAgentJob("teams_browse", { region });
  if (status.status === "failed") {
    throw new Error(teamsAuthJobMessage(status));
  }
  return teamsBrowseDataFromJob(status);
}

function teamsBrowseDataFromJob(status: LocalAgentJobStatusResponse): TeamsBrowseData {
  const result = status.result as Partial<TeamsBrowseData> | null;
  return {
    favorites: Array.isArray(result?.favorites) ? result.favorites : [],
    teams: Array.isArray(result?.teams) ? result.teams : [],
    group_chats: Array.isArray(result?.group_chats) ? result.group_chats : [],
    individual_chats: Array.isArray(result?.individual_chats) ? result.individual_chats : [],
  };
}

function flattenTeamsData(data: TeamsBrowseData | undefined): TeamsSelectionItem[] {
  if (!data) return [];

  const channels: TeamsSelectionItem[] = data.teams.flatMap((team) =>
    team.channels.map((channel: TeamsChannel) => ({
      id: channel.id,
      displayName: channel.displayName,
      type: "channel" as const,
      teamName: team.displayName,
    })),
  );
  const groups: TeamsSelectionItem[] = data.group_chats.map((chat) => ({
    id: chat.id,
    displayName: chat.topic,
    type: "group_chat",
  }));
  const dms: TeamsSelectionItem[] = data.individual_chats.map((chat) => ({
    id: chat.id,
    displayName: chat.topic,
    type: "individual_chat",
  }));

  return [...channels, ...groups, ...dms];
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function TypeIcon({
  type,
  className,
}: {
  type: TeamsSelectionItem["type"];
  className?: string;
}) {
  if (type === "channel") return <Hash className={className} />;
  if (type === "group_chat") return <MessageSquare className={className} />;
  if (type === "individual_chat") return <User className={className} />;
  return null;
}
