import type { SetupJobResults } from "@/api";

type TeamsBrowseResult = SetupJobResults["teams_browse"];

/** One conversation the picker offers: a team channel, a group chat or a one-to-one chat. */
export interface TeamsConversation {
  id: string;
  name: string;
  kind: "channel" | "group_chat" | "individual_chat" | "unknown";
  /** The team a channel belongs to. */
  team?: string;
}

/** How the picker names a conversation: a channel with its team, a chat by its topic. */
export function conversationLabel(conversation: TeamsConversation): string {
  return conversation.kind === "channel" && conversation.team ? `${conversation.team} / ${conversation.name}` : conversation.name;
}

/**
 * The conversations a `teams_browse` result lists: every team's channels,
 * then group chats, then one-to-one chats, each list in the order Teams sent.
 * Favourites repeat conversations already listed, so they are not offered twice.
 */
export function browsedConversations(result: TeamsBrowseResult | undefined): TeamsConversation[] {
  if (!result) return [];
  const channels = (result.teams ?? []).flatMap((team) =>
    team.channels.map((channel) => ({ id: channel.id, name: channel.displayName, kind: "channel" as const, team: team.displayName })),
  );
  const groups = (result.group_chats ?? []).map((chat) => ({ id: chat.id, name: chat.topic, kind: "group_chat" as const }));
  const direct = (result.individual_chats ?? []).map((chat) => ({ id: chat.id, name: chat.topic, kind: "individual_chat" as const }));
  return [...channels, ...groups, ...direct];
}

/** A stored conversation id the picker has not resolved by listing, named by the end of its id. */
export function storedConversation(id: string): TeamsConversation {
  return { id, name: `Existing conversation · ${id.length > 16 ? id.slice(-16) : id}`, kind: "unknown" };
}

/** The conversations whose name or team contains the search, ignoring case. */
export function matchingConversations(conversations: readonly TeamsConversation[], search: string): TeamsConversation[] {
  const needle = search.trim().toLowerCase();
  if (!needle) return [...conversations];
  return conversations.filter((conversation) => `${conversation.team ?? ""} ${conversation.name}`.toLowerCase().includes(needle));
}
