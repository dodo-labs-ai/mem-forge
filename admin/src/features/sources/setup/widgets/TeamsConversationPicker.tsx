import { Check, Hash, Loader2, MessageSquare, Search, User, X } from "lucide-react";
import { useMemo, useState } from "react";
import { errorMessage } from "@/lib/errors";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { teamsRegion } from "../../useTeamsSignIn";
import {
  browsedConversations,
  conversationLabel,
  matchingConversations,
  storedConversation,
  type TeamsConversation,
} from "../model/teamsConversations";
import { useTeamsConversations, useTeamsSessionCheck } from "../teamsSession";
import type { ConfigControlProps } from "../widgets";

function ConversationIcon({ kind, className }: { kind: TeamsConversation["kind"]; className?: string }) {
  if (kind === "channel") return <Hash aria-hidden className={className} />;
  if (kind === "individual_chat") return <User aria-hidden className={className} />;
  return <MessageSquare aria-hidden className={className} />;
}

/**
 * Chooses the Teams conversations a Source syncs: the selected list, then a
 * searchable list of what the Teams session can read (`teams_browse`), once
 * the session check says it is usable. A stored id shows as "Existing
 * conversation · …" until the list names it. The field holds the ids.
 */
export default function TeamsConversationPicker({ id, labelId, describedBy, value, onChange, onBlur, invalid, readOnly, config }: ConfigControlProps) {
  const region = teamsRegion(config);
  const selectedIds = Array.isArray(value) ? value : [];
  const session = useTeamsSessionCheck(region);
  const authenticated = session.data?.authenticated === true;
  const browse = useTeamsConversations(region, authenticated);
  const [search, setSearch] = useState("");

  const browsed = useMemo(() => browsedConversations(browse.data), [browse.data]);
  const byId = useMemo(() => new Map(browsed.map((conversation) => [conversation.id, conversation])), [browsed]);
  const selected = selectedIds.map((selectedId) => byId.get(selectedId) ?? storedConversation(selectedId));
  const shown = matchingConversations(browsed, search);

  function toggle(conversationId: string) {
    onChange(selectedIds.includes(conversationId) ? selectedIds.filter((item) => item !== conversationId) : [...selectedIds, conversationId]);
    onBlur?.();
  }

  return (
    <div id={id} role="group" aria-labelledby={labelId} aria-describedby={describedBy} aria-invalid={invalid || undefined} className="space-y-3">
      {selected.length > 0 ? (
        <ul aria-label="Selected conversations" className="flex flex-wrap gap-1.5">
          {selected.map((conversation) => (
            <li key={conversation.id} className="flex max-w-full items-center gap-1 rounded-full border bg-muted/40 py-0.5 pr-1 pl-2 text-xs">
              <ConversationIcon kind={conversation.kind} className="size-3 shrink-0 text-muted-foreground" />
              <span className="truncate">{conversationLabel(conversation)}</span>
              {readOnly ? null : (
                <button
                  type="button"
                  className="rounded-full p-0.5 hover:bg-muted"
                  aria-label={`Remove ${conversationLabel(conversation)}`}
                  onClick={() => toggle(conversation.id)}
                >
                  <X aria-hidden className="size-3" />
                </button>
              )}
            </li>
          ))}
        </ul>
      ) : null}

      {readOnly ? null : !authenticated ? (
        <p className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">
          {session.isPending
            ? "Checking the Teams session…"
            : "Connect the Teams session in the Teams session step to list conversations. Conversations already selected stay selected."}
        </p>
      ) : browse.isPending ? (
        <p className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 aria-hidden className="size-4 animate-spin" />
          Listing conversations…
        </p>
      ) : browse.isError ? (
        <div className="flex flex-wrap items-center gap-3 rounded-lg border p-3 text-sm">
          <span role="alert" className="text-tone-danger-foreground">
            Could not list conversations. {errorMessage(browse.error)}
          </span>
          <Button type="button" size="sm" variant="outline" onClick={() => void browse.refetch()}>
            Try again
          </Button>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="relative">
            <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="Search conversations"
              className="pl-8"
              placeholder="Search conversations"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <ul aria-label="Teams conversations" className="max-h-72 overflow-y-auto rounded-lg border">
            {shown.length === 0 ? (
              <li className="p-6 text-center text-sm text-muted-foreground">No conversations found.</li>
            ) : (
              shown.map((conversation) => {
                const isSelected = selectedIds.includes(conversation.id);
                return (
                  <li key={conversation.id} className="border-b last:border-b-0">
                    <button
                      type="button"
                      aria-pressed={isSelected}
                      className="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-muted/60"
                      onClick={() => toggle(conversation.id)}
                    >
                      <ConversationIcon kind={conversation.kind} className="size-4 shrink-0 text-muted-foreground" />
                      <span className="min-w-0 flex-1">
                        {conversation.team ? <span className="block truncate text-xs text-muted-foreground">{conversation.team}</span> : null}
                        <span className="block truncate font-medium">{conversation.name}</span>
                      </span>
                      {isSelected ? <Check aria-hidden className="size-4 shrink-0" /> : null}
                    </button>
                  </li>
                );
              })
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
