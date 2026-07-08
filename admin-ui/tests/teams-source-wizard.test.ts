import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  buildDefaultTeamsSourceConfig,
  buildTeamsSourcePayload,
  teamsSelectionLabel,
  type TeamsSelectionItem,
} from "../src/views/sources/teamsSourceConfig.js";

const selections: TeamsSelectionItem[] = [
  {
    id: "19:channel@thread.tacv2",
    displayName: "Architecture",
    type: "channel",
    teamName: "Engineering",
  },
  {
    id: "19:group@thread.v2",
    displayName: "Planning Chat",
    type: "group_chat",
  },
  {
    id: "19:dm@thread.v2",
    displayName: "Ada Lovelace",
    type: "individual_chat",
  },
];

const config = buildDefaultTeamsSourceConfig();
assert.equal(config.conversation_gap_minutes, 60);

const payload = buildTeamsSourcePayload({
  selections,
  config: {
    ...config,
    name: "Teams - Engineering",
    max_age_days: 14,
    max_block_messages: 25,
  },
});

assert.deepEqual(payload, {
  type: "teams",
  name: "Teams - Engineering",
  config: {
    region: "emea",
    conversation_ids: "19:channel@thread.tacv2, 19:group@thread.v2, 19:dm@thread.v2",
    max_age_days: 14,
    conversation_gap_minutes: 60,
    max_block_messages: 25,
  },
});

assert.equal(teamsSelectionLabel(selections[0]), "Engineering / Architecture");
assert.equal(teamsSelectionLabel(selections[1]), "Planning Chat");
assert.equal(teamsSelectionLabel(selections[2]), "Ada Lovelace");

assert.equal(Object.hasOwn(payload.config, "channels"), false);
assert.equal(Object.hasOwn(payload.config, "group_chats"), false);
assert.equal(Object.hasOwn(payload.config, "individual_chats"), false);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const teamsWizardSource = readFileSync("src/views/sources/TeamsSourceWizard.tsx", "utf8");
assert.match(
  sourcesPageSource,
  /isTeams\s*\?\s*onTeamsSelected\(\)\s*:\s*onConfigureSelected\(gene\.name\)/,
  "Teams source card should route its primary action to the Teams browser wizard",
);
assert.match(
  sourcesPageSource,
  /isTeams\s*\?\s*"Browse Teams"\s*:\s*"Configure"/,
  "Teams source card should not expose the generic hand-written config path as the primary action",
);
assert.match(
  sourcesPageSource,
  /source\.type === "teams"[\s\S]*return "teams_sync"/,
  "Teams source sync should enqueue a local-agent teams_sync job instead of server-side sync",
);
assert.match(
  teamsWizardSource,
  /operation:\s*"teams_auth"/,
  "Teams auth should be triggered through the local daemon instead of CLI instructions",
);
assert.match(
  teamsWizardSource,
  /Connect\s*<\/Button>/,
  "Teams auth prompt should expose a short Connect action",
);
assert.doesNotMatch(
  teamsWizardSource,
  /memforge auth teams|\.venv\/bin\/memforge auth teams|Run this from the project directory|Terminal/,
  "Teams auth prompt should not expose CLI fallback instructions",
);
