export interface TeamsSelectionItem {
  id: string;
  displayName: string;
  type: "channel" | "group_chat" | "individual_chat" | "unknown";
  teamName?: string;
}

export interface TeamsSourceConfig {
  name: string;
  region: string;
  initial_history_days: number;
  rolling_retention_days: number | null;
}

export const TEAMS_ROLLING_RETENTION_OPTIONS = [
  { value: null, label: "Forever" },
  { value: 365, label: "1 year" },
  { value: 730, label: "2 years" },
  { value: 1095, label: "3 years" },
] as const;

const TEAMS_ROLLING_RETENTION_DAYS = new Set<number>(
  TEAMS_ROLLING_RETENTION_OPTIONS.flatMap((option) => option.value === null ? [] : [option.value]),
);

export interface TeamsSourcePayload {
  type: "teams";
  name: string;
  config: {
    region: string;
    conversation_ids: string;
    initial_history_days: number;
    rolling_retention_days: number | null;
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
    initial_history_days: 14,
    rolling_retention_days: null,
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
      initial_history_days: numberConfig(
        source.config.initial_history_days ?? source.config.max_age_days,
        defaults.initial_history_days,
      ),
      rolling_retention_days: retentionConfig(source.config.rolling_retention_days),
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
      initial_history_days: config.initial_history_days,
      rolling_retention_days: config.rolling_retention_days,
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

function retentionConfig(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return TEAMS_ROLLING_RETENTION_DAYS.has(parsed) ? parsed : null;
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
