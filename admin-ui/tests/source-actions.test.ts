import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  getSourceActionEndpoint,
  getSourceMenuPlacement,
  getSourceMenuStyle,
  sourceActionLayout,
} from "../src/views/sources/sourceActions.js";
import {
  isImmutableExecutionModeField,
  isLocalAgentBackedSource,
  localAgentSyncOperation,
} from "../src/views/sources/localAgentSources.js";

assert.deepEqual(
  sourceActionLayout.primary.map((action) => action.id),
  ["configure", "sync"],
  "source cards should keep only Configure and Sync as visible primary actions",
);

assert.deepEqual(
  sourceActionLayout.menu.map((action) => action.id),
  ["toggle-status", "force-resync", "delete"],
  "source cards should move source lifecycle, expensive, and destructive actions into the overflow menu",
);

const toggleStatus = sourceActionLayout.menu.find((action) => action.id === "toggle-status");
assert.equal(toggleStatus?.tone, "neutral");
assert.equal(
  toggleStatus?.description,
  "Pause or resume source discovery without deleting configuration or extracted memories.",
);

const forceResync = sourceActionLayout.menu.find((action) => action.id === "force-resync");
assert.equal(forceResync?.label, "Refresh source");
assert.equal(forceResync?.tone, "neutral");
assert.equal("disabled" in (forceResync ?? {}), false);
assert.equal(
  forceResync?.description,
  "Look for new, changed, or removed documents. Existing memories are not rebuilt unless source content changed.",
);
assert.equal(getSourceActionEndpoint("src-1", "force-resync"), "/sources/src-1/force-resync");

const deleteSource = sourceActionLayout.menu.find((action) => action.id === "delete");
assert.equal(deleteSource?.tone, "destructive");
assert.equal(deleteSource?.requiresConfirmation, true);
assert.equal(getSourceActionEndpoint("src-1", "delete"), "/sources/src-1");

assert.deepEqual(
  getSourceMenuPlacement({
    triggerTop: 650,
    triggerBottom: 686,
    viewportHeight: 720,
    menuHeight: 224,
  }),
  { direction: "up", top: 418 },
  "menus near the bottom of the viewport should open upward instead of being clipped",
);

assert.deepEqual(
  getSourceMenuPlacement({
    triggerTop: 120,
    triggerBottom: 156,
    viewportHeight: 720,
    menuHeight: 224,
  }),
  { direction: "down", top: 164 },
  "menus with enough lower viewport space should open downward with an 8px gap",
);

assert.deepEqual(
  getSourceMenuStyle({
    triggerRight: 1_224,
    triggerTop: 560,
    triggerBottom: 596,
    viewportWidth: 1_280,
    viewportHeight: 720,
    menuHeight: 160,
  }),
  { position: "fixed", top: 392, left: 936, width: 288 },
  "source action menus should align to the trigger and stay within the viewport",
);

assert.deepEqual(
  getSourceMenuStyle({
    triggerRight: 240,
    triggerTop: 120,
    triggerBottom: 156,
    viewportWidth: 320,
    viewportHeight: 720,
    menuHeight: 160,
  }),
  { position: "fixed", top: 164, left: 8, width: 288 },
  "source action menus should clamp horizontally on narrow viewports",
);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const sourceRowSource = readFileSync("src/views/sources/SourceRow.tsx", "utf8");
const localAgentSourcesSource = readFileSync("src/views/sources/localAgentSources.ts", "utf8");
const syncStatusCardSource = readFileSync("src/components/admin/SourceSyncStatusCard.tsx", "utf8");
const syncActivitySource = readFileSync("src/views/sources/sourceSyncActivity.ts", "utf8");
const localAgentJobsSource = readFileSync("src/api/localAgentJobs.ts", "utf8");
const apiTypesSource = readFileSync("src/api/types.ts", "utf8");

assert.match(
  sourcesPageSource,
  /setSourceStatus\s*=\s*useMutation/,
  "SourcesPage should update source lifecycle through the generic source update endpoint",
);
assert.match(
  sourcesPageSource,
  /resourceClient\.put\(`\/sources\/\$\{sourceId\}`,\s*\{\s*status\s*\}\)/,
  "Pause and resume should use the relative PUT /sources/{id} resource path",
);
assert.match(
  sourcesPageSource,
  /pollLocalAgentSyncJob/,
  "Internal network GitHub sync should keep the row pending until the local daemon job finishes",
);
assert.match(
  sourcesPageSource,
  /getLocalAgentJob\(jobId\)/,
  "Internal network GitHub sync should poll through the target-aware local-agent helper",
);
assert.match(
  sourcesPageSource,
  /Waiting for local daemon/,
  "Local-agent sync should tell users when Cloud is waiting for their daemon",
);
assert.match(
  sourcesPageSource,
  /LOCAL_AGENT_TIMEOUT_MESSAGE/,
  "Local-agent sync should use a distinct timeout message after polling gives up",
);
assert.match(
  sourcesPageSource,
  /memforge adapter daemon run/,
  "Local-agent sync timeout should show the daemon command when a job is still waiting",
);
assert.match(
  sourcesPageSource,
  /localAgentJobBySource/,
  "Local-agent sync should retain the durable job record used by the shared activity projector",
);
assert.match(
  sourcesPageSource,
  /LOCAL_AGENT_TERMINAL_PROGRESS_RETENTION_MS/,
  "Successful local-agent sync should keep a short row-level terminal summary visible",
);
assert.match(
  sourcesPageSource,
  /function localAgentJobPayload/,
  "Local-agent job payload shaping should be centralized before enqueueing daemon work",
);
assert.match(
  sourcesPageSource,
  /function localAgentJobPayload\(\s*forceFullSync = false,?[\s\S]*force_full_sync: forceFullSync/,
  "Local-agent job creation should send only execution controls; the server supplies canonical source config",
);
assert.doesNotMatch(
  sourcesPageSource,
  /localAgentJobPayload[\s\S]{0,180}process_now/,
  "Local-agent sync jobs should not expose the old raw-upload process_now switch",
);
assert.match(
  sourceRowSource,
  /syncActivity/,
  "SourceRow should render the normalized sync activity for the matching source row",
);
assert.match(
  sourceRowSource,
  /isLocalAgentBackedSource\(source\)/,
  "SourceRow should use the same local-agent source predicate as sync job routing",
);
assert.match(
  sourceRowSource,
  /showLocalAgentStatus\s*=\s*!isPaused\s*&&\s*isLocalAgentBackedSource\(source\)\s*&&\s*capabilities\.can_sync/,
  "Only the execution owner should query and display local daemon readiness",
);
assert.match(
  sourcesPageSource,
  /\["pending", "running", "recovering"\]\.includes\(source\.sync\?\.status/,
  "Durable queued and recovering runs should keep source status polling active",
);
assert.match(
  syncActivitySource,
  /Waiting to sync[\s\S]*Recovering sync/,
  "The shared presenter should distinguish queued work from worker recovery",
);
assert.match(
  apiTypesSource,
  /execution_owner_user_id:\s*string \| null;/,
  "source ownership types should expose the persisted local execution owner",
);
assert.match(
  apiTypesSource,
  /can_configure_connection:\s*boolean;/,
  "source capabilities should distinguish connector configuration from workspace management",
);
assert.match(
  sourcesPageSource,
  /function safeSourceErrorMessage/,
  "Source sync errors should pass through only explicitly safe user-facing messages",
);
assert.doesNotMatch(
  sourcesPageSource,
  /setAuthorityMessage\(error instanceof Error && error\.message/,
  "Source sync should not expose arbitrary backend Error.message text in the UI banner",
);
assert.match(
  sourcesPageSource,
  /onToggleStatus=\{\(\)\s*=>\s*\{/,
  "SourceActionsMenu should receive a pause/resume action per source row",
);
assert.match(
  sourceRowSource,
  /const isPaused = source\.status === "paused";/,
  "SourceRow should derive paused state from the source status",
);
assert.match(
  sourceRowSource,
  /disabled=\{isSyncing \|\| isDeleting \|\| isPaused\}/,
  "Paused sources should not expose an enabled primary Sync button",
);
assert.match(
  sourceRowSource,
  /onRetry=\{isPaused \|\| !capabilities\.can_sync \? undefined : onSync\}/,
  "Paused sources and non-owners should not expose retry sync from the status bar",
);
assert.match(
  sourceRowSource,
  /source\.auth_session\s*&&\s*capabilities\.can_configure_connection/,
  "local Jira auth status should be visible only to the execution owner",
);
assert.match(
  sourceRowSource,
  /source\.sync_schedule\?\.enabled/,
  "SourceRow should show automatic sync metadata when a source schedule is enabled",
);
assert.match(
  sourceRowSource,
  /formatRelativeFuture\(source\.sync_schedule\.next_run_at\)/,
  "SourceRow should format the next scheduled sync as a future time instead of using the last-sync formatter",
);
assert.doesNotMatch(
  sourceRowSource,
  /New memories/,
  "last-sync details should not label extraction candidates as new durable memories",
);
assert.match(
  syncStatusCardSource,
  /aria-valuenow=\{determinate/,
  "the shared status card should expose numeric accessibility state only for determinate progress",
);

assert.match(
  sourcesPageSource,
  /className="[^"]*cursor-pointer[^"]*disabled:cursor-not-allowed[^"]*"/,
  "enabled overflow menu actions should use a pointer cursor while disabled actions keep not-allowed",
);
assert.equal(
  localAgentSyncOperation({ execution: { kind: "local_agent", operation: "teams_sync", immutable_config_fields: [] } } as never),
  "teams_sync",
  "Teams sources should be local-agent backed",
);
assert.equal(
  localAgentSyncOperation({ execution: { kind: "server", operation: null, immutable_config_fields: ["sync_mode"] } } as never),
  null,
  "Cloud Jira sources should not be treated as local-agent backed",
);
assert.equal(
  isLocalAgentBackedSource({ execution: { kind: "local_agent", operation: "jira_sync", immutable_config_fields: ["sync_mode"] } } as never),
  true,
  "Jira local-agent mode should share the daemon status badge path",
);
const jiraExecutionSource = {
  execution: { kind: "local_agent", operation: "jira_sync", immutable_config_fields: ["sync_mode"] },
} as never;
assert.equal(isImmutableExecutionModeField(jiraExecutionSource, "sync_mode"), true);
assert.equal(isImmutableExecutionModeField(jiraExecutionSource, "auth_mode"), false);
assert.match(
  sourcesPageSource,
  /getLocalAgentJob/,
  "Internal network GitHub source sync should use the centralized local-agent queue helper",
);
assert.doesNotMatch(
  localAgentJobsSource,
  /workspace_id|requireCurrentWorkspaceId/,
  "Local-agent jobs should select Cloud workspaces only through the configured host path",
);
assert.match(
  localAgentJobsSource,
  /hostClient\.post<LocalAgentJobCreateResponse>\(localAgentUrl\("\/jobs"\)/,
  "Local-agent job creation should use the controller-derived host client URL",
);
assert.match(
  sourcesPageSource,
  /forceResyncSource[\s\S]*createLocalAgentSyncJob\(source,\s*\{/,
  "Force refresh for local-agent sources should use the daemon job path instead of Cloud-side source sync",
);
assert.match(
  sourcesPageSource,
  /Configure a folder path before syncing this local source\./,
  "Local markdown sources without a daemon folder should fail explicitly instead of falling through",
);
assert.doesNotMatch(
  sourcesPageSource,
  /localMarkdownCanUseServerInbox/,
  "Local markdown sync should not keep a legacy server-inbox compatibility branch",
);

const sourceSetupSource = readFileSync("src/views/sources/SchemaSourceSetup.tsx", "utf8");
const sourceSetupAdaptersSource = readFileSync("src/views/sources/sourceSetupAdapters.ts", "utf8");
const teamsSourceSetupSource = readFileSync("src/views/sources/TeamsSourceSetup.tsx", "utf8");
const githubRepoFolderPickerSource = readFileSync("src/views/sources/GitHubRepoFolderPicker.tsx", "utf8");
assert.match(
  sourceSetupSource,
  /const canConfigureConnection = source \? source\.capabilities\?\.can_configure_connection === true : true;/,
  "existing local sources should consume the backend connection capability",
);
assert.match(
  sourceSetupSource,
  /content: canConfigureConnection \? <>\{connectionFields\.map\(renderField\)\}<\/> : unavailableConnection/,
  "non-owner admins should not render local connector fields or pickers",
);
assert.match(
  sourceSetupSource,
  /disabled=\{source\s*\?\s*isImmutableExecutionModeField\(source, field\.key\)\s*:\s*false\}/,
  "existing sources should render execution-mode selectors as read-only",
);
assert.match(sourceSetupSource, /type="checkbox"[\s\S]*disabled=\{disabled\}/);
assert.match(sourceSetupSource, /<textarea[\s\S]*disabled=\{disabled\}/);
assert.match(sourceSetupSource, /<Input[\s\S]*disabled=\{disabled\}/);
assert.doesNotMatch(
  localAgentSourcesSource,
  /source\.type\s*===|sync_mode|connection_mode|local_markdown/,
  "the UI should consume the server execution descriptor instead of reclassifying source types",
);
assert.match(
  sourceSetupSource,
  /\.\.\.\(canConfigureConnection\s*\?\s*\{\s*config:\s*serializeConfig\(schema\.fields, config\)\s*\}\s*:\s*\{\}\)/,
  "management-only saves must omit connector config from the API payload",
);
assert.match(
  sourceSetupSource,
  /const DISCOVERY_PREVIEW_LIMIT = 5;/,
  "source discovery preview should request a small bounded result set",
);
assert.match(
  sourceSetupSource,
  /createLocalAgentJob/,
  "Local markdown local-agent preview jobs should use the centralized target-aware helper",
);
assert.match(
  teamsSourceSetupSource,
  /createLocalAgentJob/,
  "Teams auth and browse jobs should use the centralized target-aware helper",
);
assert.match(
  githubRepoFolderPickerSource,
  /createLocalAgentJob/,
  "GitHub local-agent browse jobs should use the centralized target-aware helper",
);
assert.doesNotMatch(
  [sourcesPageSource, sourceSetupSource, teamsSourceSetupSource, githubRepoFolderPickerSource].join("\n"),
  /(?:resourceClient|hostClient)\.post<[^>]+>\([^)]*\/local-agent\/jobs/,
  "Source UI components should not create local-agent job envelopes directly",
);
assert.match(
  sourceSetupSource,
  /limit: DISCOVERY_PREVIEW_LIMIT/,
  "source discovery preview requests should send the bounded limit to the API",
);
assert.match(
  sourceSetupSource,
  /memforge adapter auth jira refresh --base-url/,
  "Jira browser-session guidance should use the refresh subcommand that uploads the local browser session",
);
assert.match(
  sourceSetupSource,
  /jiraSessionQuery\.refetch\(\)/,
  "Jira browser-session guidance should allow users to re-check after running the CLI refresh",
);
assert.match(
  sourceSetupAdaptersSource,
  /field\.key === "auth_mode"[\s\S]*sync_mode: "cloud"/,
  "Jira PAT mode should not leave Local daemon sync selected because the UI cannot pass redacted PAT secrets to daemon jobs",
);
assert.match(
  sourceSetupAdaptersSource,
  /field\.key === "sync_mode"[\s\S]*auth_mode: "browser_cookie"/,
  "Jira Local daemon sync should use browser-session auth in the current contract",
);
assert.match(
  sourceSetupSource,
  /showDiscoveryPreview[\s\S]*sourceType === "jira"[\s\S]*config\.sync_mode[\s\S]*local_agent/,
  "Jira Local daemon sync should not expose the server-side discovery preview",
);
assert.match(
  sourceSetupSource,
  /preview-discovery[\s\S]*source \? \{ source_id: source\.id \} : \{\}/,
  "editing an existing source must let discovery preview reuse its stored credentials",
);
assert.match(
  sourceSetupSource,
  /const payloadWithSchedule = \{/,
  "Source saves should bundle automatic sync settings into the source payload",
);
assert.match(
  sourceSetupSource,
  /sync_schedule:\s*\{\s*enabled: scheduleEnabled,\s*interval_minutes: intervalMinutes,\s*\}/,
  "Source saves should send the schedule shape expected by the source API",
);
assert.doesNotMatch(
  sourceSetupSource,
  /resourceClient[^\n]*\/sources\/[^`]+\/schedule/,
  "source setup should not split config and schedule persistence into two requests",
);
assert.match(
  sourceSetupSource,
  /<span className="block text-sm font-medium">Sync automatically<\/span>/,
  "Source configuration should expose a clear automatic sync control",
);

const projectBindingSource = readFileSync("src/views/sources/ProjectBindingFields.tsx", "utf8");
assert.match(
  projectBindingSource,
  /focus-visible:ring-1 focus-visible:ring-ring\/40/,
  "project picker focus styling should be visible without creating a heavy shadow around the dropdown",
);
