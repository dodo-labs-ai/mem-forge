import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

import {
  buildDefaultTeamsSourceConfig,
  buildTeamsSourcePayload,
  buildTeamsSourceUpdatePayload,
  editableTeamsSourceState,
  existingTeamsSelection,
  teamsConversationCount,
  teamsSelectionLabel,
  type TeamsSelectionItem,
} from "../src/views/sources/teamsSourceConfig.js";

const selections: TeamsSelectionItem[] = [
  { id: "19:channel@thread.tacv2", displayName: "Architecture", type: "channel", teamName: "Engineering" },
  { id: "19:group@thread.v2", displayName: "Planning Chat", type: "group_chat" },
  { id: "19:dm@thread.v2", displayName: "Ada Lovelace", type: "individual_chat" },
];
const config = { ...buildDefaultTeamsSourceConfig(), name: "Teams - Engineering", max_block_messages: 25 };

const payload = buildTeamsSourcePayload({ selections, config });
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

assert.deepEqual(buildTeamsSourceUpdatePayload({ selections, config }), {
  name: payload.name,
  config: payload.config,
});
assert.equal(teamsConversationCount({ conversation_ids: payload.config.conversation_ids }), 3);
assert.equal(
  teamsConversationCount({ group_chats: ["legacy-chat"] }),
  null,
  "the UI reads only canonical conversation_ids and has no legacy selection fallback",
);
assert.equal(teamsSelectionLabel(selections[0]), "Engineering / Architecture");
assert.equal(existingTeamsSelection("19:existing@thread.v2").type, "unknown");

const state = editableTeamsSourceState({
  id: "src-teams",
  name: "PCC Agent Dev",
  config: { ...payload.config, max_age_days: 30 },
});
assert.equal(state.config.name, "PCC Agent Dev");
assert.equal(state.config.max_age_days, 30);
assert.deepEqual(state.conversationIds, selections.map((selection) => selection.id));

const setupSource = readFileSync("src/views/sources/TeamsSourceSetup.tsx", "utf8");
const entrySource = readFileSync("src/views/sources/SourceSetupDialog.tsx", "utf8");
const pageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
assert.match(setupSource, /<SourceSetupShell/);
assert.match(setupSource, /<ProjectBindingFields/);
assert.match(setupSource, /sync_schedule/);
assert.match(setupSource, /runTeamsLocalAgentJob\("teams_auth"/);
assert.match(setupSource, /runTeamsLocalAgentJob\("teams_browse"/);
assert.match(entrySource, /sourceType === "teams"/);
assert.match(pageSource, /<SourceSetupDialog/);
assert.doesNotMatch(pageSource, /TeamsSourceWizard|teamsWizardOpen|onTeamsSelected/);
assert.equal(existsSync("src/views/sources/TeamsSourceWizard.tsx"), false);
