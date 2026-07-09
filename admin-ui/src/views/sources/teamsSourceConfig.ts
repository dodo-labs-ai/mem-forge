export interface TeamsSelectionItem {
  id: string;
  displayName: string;
  type: "channel" | "group_chat" | "individual_chat";
  teamName?: string;
}

export interface TeamsSourceConfig {
  name: string;
  region: string;
  max_age_days: number;
  conversation_gap_minutes: number;
  max_block_messages: number;
}

export interface TeamsSourcePayload {
  type: "teams";
  name: string;
  config: {
    region: string;
    conversation_ids: string;
    max_age_days: number;
    conversation_gap_minutes: number;
    max_block_messages: number;
  };
}

export function buildDefaultTeamsSourceConfig(): TeamsSourceConfig {
  return {
    name: "",
    region: "emea",
    max_age_days: 14,
    conversation_gap_minutes: 60,
    max_block_messages: 100,
  };
}

export function teamsSelectionLabel(item: TeamsSelectionItem): string {
  if (item.type === "channel" && item.teamName) {
    return `${item.teamName} / ${item.displayName}`;
  }
  return item.displayName;
}

export function buildTeamsSourcePayload({
  selections,
  config,
}: {
  selections: TeamsSelectionItem[];
  config: TeamsSourceConfig;
}): TeamsSourcePayload {
  return {
    type: "teams",
    name: config.name,
    config: {
      region: config.region,
      conversation_ids: selections.map((item) => item.id).join(", "),
      max_age_days: config.max_age_days,
      conversation_gap_minutes: config.conversation_gap_minutes,
      max_block_messages: config.max_block_messages,
    },
  };
}
