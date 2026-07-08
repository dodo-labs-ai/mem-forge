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
  Terminal,
  User,
  X,
} from "lucide-react";
import client from "@/api/client";
import type { TeamsAuthStatus, TeamsBrowseData, TeamsChannel } from "@/api/types";
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
  teamsSelectionLabel,
  type TeamsSelectionItem,
  type TeamsSourceConfig,
} from "./teamsSourceConfig";

export function TeamsSourceWizard({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="gap-0 overflow-hidden p-0 sm:max-w-2xl">
        {open && (
          <TeamsSourceWizardBody
            onOpenChange={onOpenChange}
            onCreated={onCreated}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function TeamsSourceWizardBody({
  onOpenChange,
  onCreated,
}: {
  onOpenChange: (open: boolean) => void;
  onCreated: () => void;
}) {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const [selections, setSelections] = useState<Map<string, TeamsSelectionItem>>(new Map());
  const [config, setConfig] = useState(buildDefaultTeamsSourceConfig);

  return (
    <>
      {step === 0 && <AuthCheckStep onAuthenticated={() => setStep(1)} />}
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
        />
      )}
    </>
  );
}

function AuthCheckStep({ onAuthenticated }: { onAuthenticated: () => void }) {
  const { data, isLoading, isFetching, refetch } = useQuery<TeamsAuthStatus>({
    queryKey: ["teams-auth-check"],
    queryFn: () => client.get("/api/genes/teams/auth-check").then((response) => response.data),
    retry: false,
  });

  useEffect(() => {
    if (data?.authenticated) onAuthenticated();
  }, [data?.authenticated, onAuthenticated]);

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
              <span>{data?.error || "Teams authentication required."}</span>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Run this from the project directory:</p>
              <div className="flex items-center gap-2 rounded-lg bg-muted p-3 font-mono text-sm">
                <Terminal className="size-4 shrink-0 text-muted-foreground" />
                <code>.venv/bin/memforge auth teams</code>
              </div>
              <p className="text-xs text-muted-foreground">
                Log into Teams in Chrome first, then run the command to extract your session.
              </p>
            </div>
          </div>
        )}
      </div>

      {!data?.authenticated && !isLoading && (
        <DialogFooter>
          <Button type="button" onClick={() => refetch()} disabled={isFetching}>
            {isFetching && <Loader2 className="size-4 animate-spin" />}
            Check Again
          </Button>
        </DialogFooter>
      )}
    </>
  );
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
    queryFn: () =>
      client.get("/api/genes/teams/browse", { params: { region } }).then((response) => response.data),
    retry: 1,
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
  const visibleItems = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) =>
      `${item.teamName ?? ""} ${item.displayName}`.toLowerCase().includes(needle),
    );
  }, [items, search]);

  return (
    <>
      <div className="space-y-4 p-4">
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
          <div className="max-h-[22rem] overflow-y-auto rounded-lg border">
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

      <DialogFooter className="flex-row justify-between sm:justify-between">
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
}: {
  selections: Map<string, TeamsSelectionItem>;
  config: TeamsSourceConfig;
  onConfigChange: (config: TeamsSourceConfig) => void;
  onBack: () => void;
  onCreated: () => void;
}) {
  const queryClient = useQueryClient();
  const createSource = useMutation({
    mutationFn: (payload: { type: string; name: string; config: Record<string, unknown> }) =>
      client.post("/api/sources", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      onCreated();
    },
  });

  const selectedItems = [...selections.values()];
  const updateNumber = (field: keyof TeamsSourceConfig, fallback: number) => (value: string) => {
    onConfigChange({ ...config, [field]: Number(value) || fallback });
  };

  const handleCreate = () => {
    createSource.mutate(buildTeamsSourcePayload({ selections: selectedItems, config }));
  };

  return (
    <>
      <div className="max-h-[32rem] space-y-4 overflow-y-auto p-4">
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
            <Input type="number" value={config.max_age_days} onChange={(event) => updateNumber("max_age_days", 90)(event.target.value)} />
          </Field>
          <Field label="Gap (minutes)">
            <Input type="number" value={config.conversation_gap_minutes} onChange={(event) => updateNumber("conversation_gap_minutes", 60)(event.target.value)} />
          </Field>
          <Field label="Max messages/block">
            <Input type="number" value={config.max_block_messages} onChange={(event) => updateNumber("max_block_messages", 100)(event.target.value)} />
          </Field>
        </div>

        {createSource.isError && (
          <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            Failed to create source. Please try again.
          </div>
        )}
      </div>

      <DialogFooter className="flex-row justify-between sm:justify-between">
        <Button type="button" variant="ghost" onClick={onBack}>
          <ArrowLeft className="size-4" />
          Back
        </Button>
        <Button type="button" onClick={handleCreate} disabled={!config.name.trim() || createSource.isPending}>
          {createSource.isPending ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
          Create Source
        </Button>
      </DialogFooter>
    </>
  );
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
  return <User className={className} />;
}
