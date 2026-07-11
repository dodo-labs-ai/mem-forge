import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  buildDefaultTeamsSourceConfig,
  buildTeamsSourcePayload,
  buildTeamsSourceUpdatePayload,
  editableTeamsSourceState,
  existingTeamsSelection,
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
assert.equal(existingTeamsSelection("19:existing@thread.v2").type, "unknown");

assert.equal(Object.hasOwn(payload.config, "channels"), false);
assert.equal(Object.hasOwn(payload.config, "group_chats"), false);
assert.equal(Object.hasOwn(payload.config, "individual_chats"), false);

const existingSource = {
  id: "src-teams",
  name: "PCC Agent Dev",
  config: {
    region: "emea",
    conversation_ids: "19:group@thread.v2, 19:dm@thread.v2",
    max_age_days: 30,
    conversation_gap_minutes: 45,
    max_block_messages: 80,
    language: "en",
    channels: ["legacy/channel"],
  },
};
const existingState = editableTeamsSourceState(existingSource);
assert.deepEqual(existingState.config, {
  name: "PCC Agent Dev",
  region: "emea",
  max_age_days: 30,
  conversation_gap_minutes: 45,
  max_block_messages: 80,
});
assert.deepEqual(existingState.conversationIds, [
  "19:group@thread.v2",
  "19:dm@thread.v2",
]);
assert.deepEqual(
  buildTeamsSourceUpdatePayload({
    selections: selections.slice(1),
    config: existingState.config,
    existingConfig: existingSource.config,
  }),
  {
    name: "PCC Agent Dev",
    config: {
      region: "emea",
      conversation_ids: "19:group@thread.v2, 19:dm@thread.v2",
      max_age_days: 30,
      conversation_gap_minutes: 45,
      max_block_messages: 80,
      language: "en",
    },
  },
);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const teamsWizardSource = readFileSync("src/views/sources/TeamsSourceWizard.tsx", "utf8");
const localAgentSourcesSource = readFileSync("src/views/sources/localAgentSources.ts", "utf8");
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
  localAgentSourcesSource,
  /source\.execution\?\.operation\s*\?\?\s*null/,
  "Teams source sync should consume the server-provided local-agent operation",
);
assert.match(
  sourcesPageSource,
  /localAgentJobErrorMessage\(status\)/,
  "Local-agent sync failures should surface the daemon job error instead of a generic no-op",
);
assert.match(
  sourcesPageSource,
  /Sign in to Teams in Chrome, then retry sync\./,
  "Teams sync failures should give a direct browser-session recovery action",
);
assert.match(
  teamsWizardSource,
  /runTeamsLocalAgentJob\("teams_auth"/,
  "Teams auth should be triggered through the local daemon instead of CLI instructions",
);
assert.match(
  sourcesPageSource,
  /source\.type\s*===\s*"teams"[\s\S]{0,240}setTeamsWizardSource\(source\)/,
  "Existing Teams sources should open the Teams wizard instead of the stale generic schema form",
);
assert.match(
  sourcesPageSource,
  /<TeamsSourceWizard[\s\S]{0,240}source=\{teamsWizardSource\}/,
  "The Teams wizard should receive the existing source when Configure is selected",
);
assert.match(
  teamsWizardSource,
  /resourceClient\.put\(`\/sources\/\$\{source\.id\}`/,
  "Editing through the Teams wizard should update the existing source",
);
assert.match(
  teamsWizardSource,
  /runTeamsLocalAgentJob\("teams_auth_check"/,
  "Teams auth check should be triggered through the local daemon",
);
assert.match(
  teamsWizardSource,
  /runTeamsLocalAgentJob\("teams_browse"/,
  "Teams browse should be triggered through the local daemon",
);
assert.doesNotMatch(
  teamsWizardSource,
  /resourceClient[^\n]*\/genes\/teams\/(?:auth-check|browse)/,
  "Teams wizard should not call direct server-side Teams auth or browse endpoints",
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
assert.match(
  teamsWizardSource,
  /<DialogContent className="flex max-h-\[calc\(100dvh-2rem\)\] flex-col gap-0 overflow-hidden p-0 sm:max-w-2xl">/,
  "Teams wizard should stay inside the visible viewport and lay out its steps vertically",
);
assert.match(
  teamsWizardSource,
  /<div className="flex min-h-0 flex-1 flex-col gap-4 p-4">/,
  "Teams conversation browsing should give the results list the remaining dialog height",
);
assert.match(
  teamsWizardSource,
  /<div className="min-h-0 flex-1 overflow-y-auto rounded-lg border">/,
  "Teams conversation results should scroll inside the dialog instead of pushing actions out of view",
);
assert.equal(
  teamsWizardSource.match(/<DialogFooter className="mx-0 mb-0/g)?.length,
  3,
  "Every Teams wizard step should keep its footer inside the zero-padding dialog boundary",
);
assert.equal(
  teamsWizardSource.match(/<DialogFooter className="mx-0 mb-0 shrink-0 p-5/g)?.length,
  3,
  "Every Teams wizard footer should keep comfortable spacing from the dialog edges",
);
