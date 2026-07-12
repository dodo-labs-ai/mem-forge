export interface TeamsSelectionItem {
  id: string;
  displayName: string;
  type: "channel" | "group_chat" | "individual_chat" | "unknown";
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

export interface EditableTeamsSource {
  id: string;
  name: string;
  config: Record<string, unknown>;
}

export interface EditableTeamsSourceState {
  config: TeamsSourceConfig;
  conversationIds: string[];
}

export interface TeamsSourceUpdatePayload {
  name: string;
  config: Record<string, unknown>;
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

export function editableTeamsSourceState(source: EditableTeamsSource): EditableTeamsSourceState {
  const defaults = buildDefaultTeamsSourceConfig();
  return {
    config: {
      name: source.name,
      region: stringConfig(source.config.region, defaults.region),
      max_age_days: numberConfig(source.config.max_age_days, defaults.max_age_days),
      conversation_gap_minutes: numberConfig(
        source.config.conversation_gap_minutes,
        defaults.conversation_gap_minutes,
      ),
      max_block_messages: numberConfig(
        source.config.max_block_messages,
        defaults.max_block_messages,
      ),
    },
    conversationIds: stringListConfig(source.config.conversation_ids),
  };
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

export function buildTeamsSourceUpdatePayload({
  selections,
  config,
}: {
  selections: TeamsSelectionItem[];
  config: TeamsSourceConfig;
}): TeamsSourceUpdatePayload {
  const payload = buildTeamsSourcePayload({ selections, config });
  return { name: payload.name, config: payload.config };
}

export function existingTeamsSelection(id: string): TeamsSelectionItem {
  const suffix = id.length > 16 ? id.slice(-16) : id;
  return {
    id,
    displayName: `Existing conversation · ${suffix}`,
    type: "unknown",
  };
}

function stringConfig(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function numberConfig(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function stringListConfig(value: unknown): string[] {
  const values = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [];
  return [...new Set(values.map((item) => String(item).trim()).filter(Boolean))];
}

export function teamsConversationCount(config: Record<string, unknown>): number | null {
  const values = stringList(config.conversation_ids);
  return values.length > 0 ? new Set(values).size : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === "string") {
    return value.split(",").map((item) => item.trim()).filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  return [];
}
