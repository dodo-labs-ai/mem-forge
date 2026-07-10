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
assert.equal(getSourceActionEndpoint("src-1", "force-resync"), "/api/sources/src-1/force-resync");

const deleteSource = sourceActionLayout.menu.find((action) => action.id === "delete");
assert.equal(deleteSource?.tone, "destructive");
assert.equal(deleteSource?.requiresConfirmation, true);
assert.equal(getSourceActionEndpoint("src-1", "delete"), "/api/sources/src-1");

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
const syncStatusBarSource = readFileSync("src/components/admin/SyncStatusBar.tsx", "utf8");
const localAgentJobsSource = readFileSync("src/api/localAgentJobs.ts", "utf8");
const apiTypesSource = readFileSync("src/api/types.ts", "utf8");

assert.match(
  sourcesPageSource,
  /setSourceStatus\s*=\s*useMutation/,
  "SourcesPage should update source lifecycle through the generic source update endpoint",
);
assert.match(
  sourcesPageSource,
  /client\.put\(`\/api\/sources\/\$\{sourceId\}`,\s*\{\s*status\s*\}\)/,
  "Pause and resume should use PUT /api/sources/{id} with a status body",
);
assert.match(
  sourcesPageSource,
  /pollLocalAgentSyncJob/,
  "Internal network GitHub sync should keep the row pending until the local daemon job finishes",
);
assert.match(
  sourcesPageSource,
  /\/api\/cloud\/local-agent\/jobs\/\$\{jobId\}/,
  "Internal network GitHub sync should poll the local-agent job status endpoint",
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
  /localAgentProgressBySource/,
  "Local-agent sync should track row-level progress instead of relying only on a global banner",
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
  /delete payload\.local_agent_documents_dir;[\s\S]*delete payload\.local_agent_package_manifest;/,
  "Local-agent job payloads should not forward server-side package inbox metadata to the local daemon",
);
assert.match(
  sourceRowSource,
  /localAgentProgress/,
  "SourceRow should render local-agent job progress for the matching source row",
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
assert.doesNotMatch(
  syncStatusBarSource,
  /new memories|stored memories/i,
  "sync status details should avoid memory extraction counters that can differ from durable memory counts",
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
  /\/api\/cloud\/local-agent\/jobs/,
  "Internal network GitHub source sync should use the cloud local-agent queue",
);
assert.match(
  localAgentJobsSource,
  /workspace_id:\s*requireCurrentWorkspaceId\(\)/,
  "Local-agent jobs should bind to the selected workspace instead of relying on a primary workspace",
);
assert.match(
  localAgentJobsSource,
  /client\.post<LocalAgentJobCreateResponse>\("\/api\/cloud\/local-agent\/jobs"/,
  "Local-agent job creation should be centralized behind the API helper",
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

const sourceConfigDialogSource = readFileSync("src/views/sources/SourceConfigDialog.tsx", "utf8");
const teamsSourceWizardSource = readFileSync("src/views/sources/TeamsSourceWizard.tsx", "utf8");
const githubRepoFolderPickerSource = readFileSync("src/views/sources/GitHubRepoFolderPicker.tsx", "utf8");
assert.match(
  sourceConfigDialogSource,
  /const canConfigureConnection = source \? source\.capabilities\?\.can_configure_connection === true : true;/,
  "existing local sources should consume the backend connection capability",
);
assert.match(
  sourceConfigDialogSource,
  /canConfigureConnection\s*&&\s*fieldsByGroup\.map/,
  "non-owner admins should not render local connector fields or pickers",
);
assert.match(
  sourceConfigDialogSource,
  /disabled=\{source\s*\?\s*isImmutableExecutionModeField\(source, field\.key\)\s*:\s*false\}/,
  "existing sources should render execution-mode selectors as read-only",
);
assert.match(sourceConfigDialogSource, /type="checkbox"[\s\S]*disabled=\{disabled\}/);
assert.match(sourceConfigDialogSource, /<textarea[\s\S]*disabled=\{disabled\}/);
assert.match(sourceConfigDialogSource, /<Input[\s\S]*disabled=\{disabled\}/);
assert.doesNotMatch(
  localAgentSourcesSource,
  /source\.type\s*===|sync_mode|connection_mode|local_markdown/,
  "the UI should consume the server execution descriptor instead of reclassifying source types",
);
assert.match(
  sourceConfigDialogSource,
  /\.\.\.\(canConfigureConnection\s*\?\s*\{\s*config:\s*serializeConfig\(schema\.fields, config\)\s*\}\s*:\s*\{\}\)/,
  "management-only saves must omit connector config from the API payload",
);
assert.match(
  sourceConfigDialogSource,
  /const DISCOVERY_PREVIEW_LIMIT = 5;/,
  "source discovery preview should request a small bounded result set",
);
assert.match(
  sourceConfigDialogSource,
  /createLocalAgentJob/,
  "Local markdown local-agent preview jobs should bind to the selected workspace",
);
assert.match(
  teamsSourceWizardSource,
  /createLocalAgentJob/,
  "Teams auth and browse jobs should bind to the selected workspace",
);
assert.match(
  githubRepoFolderPickerSource,
  /createLocalAgentJob/,
  "GitHub local-agent browse jobs should bind to the selected workspace",
);
assert.doesNotMatch(
  [sourcesPageSource, sourceConfigDialogSource, teamsSourceWizardSource, githubRepoFolderPickerSource].join("\n"),
  /client\.post<[^>]+>\("\/api\/cloud\/local-agent\/jobs"/,
  "Source UI components should not create local-agent job envelopes directly",
);
assert.match(
  sourceConfigDialogSource,
  /function discoveryPreviewGroupKey/,
  "source discovery preview placement should be centralized instead of hard-coded inline",
);
assert.match(
  sourceConfigDialogSource,
  /group\.key === "scope"/,
  "source discovery preview should appear after the scope fields when a source has a What to Sync group",
);
assert.match(
  sourceConfigDialogSource,
  /limit: DISCOVERY_PREVIEW_LIMIT/,
  "source discovery preview requests should send the bounded limit to the API",
);
assert.match(
  sourceConfigDialogSource,
  /memforge adapter auth jira refresh --base-url/,
  "Jira browser-session guidance should use the refresh subcommand that uploads the local browser session",
);
assert.match(
  sourceConfigDialogSource,
  /jiraSessionQuery\.refetch\(\)/,
  "Jira browser-session guidance should allow users to re-check after running the CLI refresh",
);
assert.match(
  sourceConfigDialogSource,
  /field\.key === "auth_mode"[\s\S]*next\.sync_mode = "cloud"/,
  "Jira PAT mode should not leave Local daemon sync selected because the UI cannot pass redacted PAT secrets to daemon jobs",
);
assert.match(
  sourceConfigDialogSource,
  /field\.key === "sync_mode"[\s\S]*next\.auth_mode = "browser_cookie"/,
  "Jira Local daemon sync should use browser-session auth in the current contract",
);
assert.match(
  sourceConfigDialogSource,
  /showDiscoveryPreview[\s\S]*sourceType === "jira"[\s\S]*config\.sync_mode[\s\S]*local_agent/,
  "Jira Local daemon sync should not expose the server-side discovery preview",
);
assert.match(
  sourceConfigDialogSource,
  /const payloadWithSchedule = \{/,
  "Source saves should bundle automatic sync settings into the source payload",
);
assert.match(
  sourceConfigDialogSource,
  /sync_schedule:\s*\{\s*enabled: scheduleEnabled,\s*interval_minutes: intervalMinutes,\s*\}/,
  "Source saves should send the schedule shape expected by the source API",
);
assert.doesNotMatch(
  sourceConfigDialogSource,
  /\/api\/sources\/[^`]+\/schedule/,
  "SourceConfigDialog should not split config and schedule persistence into two requests",
);
assert.match(
  sourceConfigDialogSource,
  /<span className="block text-sm font-medium">Sync on a schedule<\/span>/,
  "Source configuration should expose a clear automatic sync control",
);

const projectBindingSource = readFileSync("src/views/sources/ProjectBindingFields.tsx", "utf8");
assert.match(
  projectBindingSource,
  /focus-visible:ring-1 focus-visible:ring-ring\/40/,
  "project picker focus styling should be visible without creating a heavy shadow around the dropdown",
);
