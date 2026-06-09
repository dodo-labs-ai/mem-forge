#!/usr/bin/env node
// Interactive Clack-based menu for the `memforge` CLI.
//
// This script never re-implements scriptable behavior. Every action collects
// inputs through `@clack/prompts`, then shells out to the canonical Python
// `memforge` commands (target, adapter, memory). The wrapper runs only when
// `memforge` is invoked with no subcommand, so the Python entrypoint sets
// `MEMFORGE_NO_INTERACTIVE=1` for every spawn to prevent recursion.
//
// The menu is organized by intent (areas) rather than by the command tree:
// connect a server, sync a local repository, connect Jira, search, inspect status.

import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";

let prompts;
try {
  prompts = await import("@clack/prompts");
} catch (error) {
  process.stderr.write(
    "memforge interactive UI requires @clack/prompts.\n" +
      "Install once with:\n" +
      "  cd cli && npm install\n" +
      "Then re-run `memforge`.\n",
  );
  process.exit(2);
}

const {
  intro,
  outro,
  group,
  select,
  text,
  confirm,
  isCancel,
  spinner,
  log,
  note,
} = prompts;

// Color is optional polish. picocolors ships with @clack/prompts; if it is ever
// unavailable, fall back to identity functions so the CLI still runs uncolored.
let pc;
try {
  pc = (await import("picocolors")).default;
} catch {
  const identity = (value) => value;
  pc = { bold: identity, green: identity, dim: identity, cyan: identity };
}

const MEMFORGE_BIN = process.env.MEMFORGE_CLI_BIN || "memforge";
const MENU_MAX_ITEMS = 12;

// ---------------------------------------------------------------------------
// Shell-out + result helpers
// ---------------------------------------------------------------------------

function runMemforge(args) {
  // Async spawn (never the blocking variant) so a clack spinner keeps
  // animating while the Python subcommand runs.
  return new Promise((resolve) => {
    const env = { ...process.env, MEMFORGE_NO_INTERACTIVE: "1" };
    const child = spawn(MEMFORGE_BIN, args, { env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => resolve({ code: null, stdout, stderr, error }));
    child.on("close", (code) => resolve({ code, stdout, stderr, error: null }));
  });
}

function parseJson(stdout) {
  if (!stdout || !stdout.trim()) return null;
  try {
    return JSON.parse(stdout);
  } catch {
    return null;
  }
}

function reportResult(label, result) {
  if (result.error) {
    log.error(`${label} failed to start: ${result.error.message}`);
    return null;
  }
  const payload = parseJson(result.stdout);
  if (result.code !== 0) {
    if (payload?.error) {
      log.error(`${label} failed: ${payload.error}${payload.detail ? ` - ${payload.detail}` : ""}`);
    } else {
      const detail = (result.stderr || result.stdout || "").trim().split("\n").slice(0, 6).join("\n");
      log.error(`${label} failed (exit ${result.code})${detail ? `:\n${detail}` : ""}`);
    }
    return payload;
  }
  log.success(`${label} ok`);
  return payload;
}

async function runStep(label, args) {
  const s = spinner();
  s.start(label);
  const result = await runMemforge(args);
  s.stop(label);
  return reportResult(label, result);
}

// ---------------------------------------------------------------------------
// Prompt helpers
// ---------------------------------------------------------------------------

// ESC / Ctrl-C in any prompt unwinds one level up (back to the parent menu)
// instead of quitting outright. Cancelled prompts throw BACK; each menu loop
// catches it (sub-menu returns to its parent; the top menu and the connect gate
// quit, since nothing sits above them).
const BACK = Symbol("back");

function goBack() {
  throw BACK;
}

function ensureNotCancelled(value) {
  if (isCancel(value)) throw BACK;
  return value;
}

// The address of the connected MemForge server, shown in the header so the user
// always knows which backend they are acting on. Resolved after the connect gate.
let activeServer = "";

function header() {
  // Clack has no flush API and is built for one-shot flows, so a persistent
  // menu accumulates a transcript. Mirror clack's own startup move (the basic
  // example does `console.clear()` then `intro`) on every menu render.
  console.clear();
  const title = pc.bold("MemForge");
  intro(activeServer ? `${title}  ${pc.green(activeServer)}` : title);
}

async function pause() {
  // No "press any key" in clack; a no-validate text prompt lets the user read
  // an action's output before the next screen clears. Cancel just returns.
  const value = await text({ message: "Press Enter to continue", placeholder: "" });
  return isCancel(value) ? undefined : value;
}

function required(value) {
  return value && value.trim() ? undefined : "Required";
}

function httpUrl(value) {
  return value && value.startsWith("http") ? undefined : "Must start with http:// or https://";
}

function httpsUrl(value) {
  return value && value.startsWith("https://") ? undefined : "Use an https:// URL";
}

function expandHome(value) {
  if (value === "~") return homedir();
  if (value.startsWith("~/")) return join(homedir(), value.slice(2));
  return value;
}

function validateFolder(value) {
  const raw = (value || "").trim();
  if (!raw) return "Required";
  const resolved = expandHome(raw);
  if (!existsSync(resolved)) return "That path does not exist";
  try {
    if (!statSync(resolved).isDirectory()) return "That path is not a folder";
  } catch {
    return "That path is not readable";
  }
  return undefined;
}

function slugify(value) {
  return (value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "repo";
}

function splitList(value) {
  if (!value) return [];
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function formatCounts(counts) {
  return Object.entries(counts)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
}

async function pickKbProfile(message) {
  const profiles = parseJson((await runMemforge(["adapter", "kb", "list"])).stdout)?.profiles ?? {};
  const names = Object.keys(profiles);
  if (!names.length) {
    log.warn("No repositories configured yet. Use 'Set up a repository' first.");
    return null;
  }
  return ensureNotCancelled(
    await select({
      message,
      maxItems: MENU_MAX_ITEMS,
      options: names.map((name) => ({
        value: name,
        label: name,
        hint: profiles[name]?.source_id ? `→ ${profiles[name].source_id}` : "not linked",
      })),
    }),
  );
}

function jiraOriginHint(origin) {
  const bits = [];
  bits.push(origin.status ? `${origin.status}${origin.principal_name ? ` (${origin.principal_name})` : ""}` : "no session");
  if (origin.configured) bits.push("configured");
  return bits.join(" · ");
}

async function promptNewSessionUrl() {
  const url = ensureNotCancelled(
    await text({ message: "Base URL", placeholder: "https://jira.example.test", validate: httpsUrl }),
  );
  return url.trim();
}

async function pickBrowserOrigin(provider, message, { allowNew = true } = {}) {
  // The "history" is the set of known origins for a browser-session provider:
  // authenticated sessions plus configured sources (`adapter auth <provider> list`).
  const listed = parseJson((await runMemforge(["adapter", "auth", provider, "list"])).stdout);
  const origins = Array.isArray(listed?.origins) ? listed.origins : [];
  if (!origins.length) {
    if (!allowNew) {
      log.warn("No remembered origins yet. Authenticate one first.");
      return null;
    }
    return promptNewSessionUrl();
  }
  const options = origins.map((origin) => ({
    value: origin.origin,
    label: origin.origin,
    hint: jiraOriginHint(origin),
  }));
  if (allowNew) options.push({ value: "__new__", label: "➕ Enter a new URL…" });
  const choice = ensureNotCancelled(await select({ message, maxItems: MENU_MAX_ITEMS, options }));
  return choice === "__new__" ? promptNewSessionUrl() : choice;
}

// ---------------------------------------------------------------------------
// Area: Connect a MemForge server
// ---------------------------------------------------------------------------

async function actionConnectServer() {
  const answers = await group(
    {
      name: () => text({ message: "Server name", placeholder: "sap.prod", validate: required }),
      apiUrl: () =>
        text({ message: "API URL", placeholder: "https://memforge.example.test", validate: httpUrl }),
      tokenEnv: () =>
        text({ message: "Env var holding the API token (optional)", placeholder: "MEMFORGE_API_TOKEN" }),
    },
    { onCancel: goBack },
  );

  const args = ["target", "add", answers.name.trim(), "--api-url", answers.apiUrl.trim()];
  if (answers.tokenEnv && answers.tokenEnv.trim()) args.push("--token-env", answers.tokenEnv.trim());

  await runStep("Adding server", args);
  await runStep("Activating server", ["target", "use", answers.name.trim()]);

  const checkNow = ensureNotCancelled(
    await confirm({ message: "Run a health check against this server now?", initialValue: true }),
  );
  if (checkNow) await runStep("Health check", ["target", "check"]);
}

async function actionSwitchServer() {
  const config = parseJson((await runMemforge(["target", "list"])).stdout) ?? {};
  const targets = config.targets && typeof config.targets === "object" ? config.targets : {};
  const names = Object.keys(targets);
  if (!names.length) {
    log.warn("No servers configured yet. Use 'Connect a server' first.");
    return;
  }
  const choice = ensureNotCancelled(
    await select({
      message: "Switch active server",
      maxItems: MENU_MAX_ITEMS,
      options: names.map((name) => ({
        value: name,
        label: name,
        hint: `${targets[name]?.api_url || ""}${config.active === name ? " (active)" : ""}`,
      })),
    }),
  );
  await runStep("Activating server", ["target", "use", choice]);
}

async function actionHealthCheck() {
  await runStep("Health check", ["target", "check"]);
}

// ---------------------------------------------------------------------------
// Area: Local repository
// ---------------------------------------------------------------------------

// Linking a repo needs a *reachable* server, not just a configured target.
// Probe health first; if the backend is down, the only useful action is to
// connect/start one; don't let the user fill out the whole wizard and fail at
// the link step.
async function probeServerReachable() {
  const s = spinner();
  s.start("Checking MemForge server");
  const probe = await runMemforge(["target", "check"]);
  s.stop("Checking MemForge server");
  const health = parseJson(probe.stdout);
  return probe.code === 0 && Boolean(health) && !health.error;
}

async function resolveServerAddress() {
  if (process.env.MEMFORGE_API_URL) return process.env.MEMFORGE_API_URL;
  const config = parseJson((await runMemforge(["target", "list"])).stdout) ?? {};
  const active = config.active;
  const url = active && config.targets?.[active]?.api_url;
  return url ? `${url} (${active})` : "http://127.0.0.1:8765 (default)";
}

async function ensureServerReachable() {
  if (await probeServerReachable()) return true;

  log.warn("No reachable MemForge server. Connect or start one before adding a repo.");
  const choice = ensureNotCancelled(
    await select({
      message: "MemForge server is not reachable.",
      options: [
        { value: "connect", label: "Connect a server" },
        { value: "back", label: "← Back" },
      ],
    }),
  );
  if (choice !== "connect") return false;

  await actionConnectServer();
  if (await probeServerReachable()) return true;
  log.warn("Still can't reach the server. Start the API, then run 'Set up a repository' again.");
  return false;
}

async function actionSetupRepository() {
  if (!(await ensureServerReachable())) return;
  const folderInput = ensureNotCancelled(
    await text({
      message: "Path to the repository folder",
      placeholder: "~/work/my-repo",
      validate: validateFolder,
    }),
  );
  const root = expandHome(folderInput.trim());
  const folderName = basename(root);
  if (existsSync(join(root, ".obsidian"))) {
    log.info(`Detected an Obsidian vault: ${folderName}`);
  }

  // Instant feedback: confirm the folder actually has files before going on.
  const scan = await runStep("Scanning folder", ["adapter", "kb", "scan", "--root", root, "--limit", "5"]);
  const found = scan?.counts?.included ?? 0;
  if (!found) {
    log.warn("No matching files in that folder. Nothing to sync yet.");
    return;
  }
  note(formatCounts(scan.counts), `Found ${found} matching files`);

  const setup = await group(
    {
      name: () =>
        text({ message: "Name this repository", initialValue: folderName, validate: required }),
      customize: () => confirm({ message: "Customize include / exclude patterns?", initialValue: false }),
    },
    { onCancel: goBack },
  );

  let includes = [];
  let excludes = [];
  if (setup.customize) {
    const globs = await group(
      {
        include: () => text({ message: "Include globs (comma-separated, blank for defaults)", placeholder: "**/*.md" }),
        exclude: () => text({ message: "Extra exclude globs (comma-separated, blank for defaults)", placeholder: "archive/**" }),
      },
      { onCancel: goBack },
    );
    includes = splitList(globs.include);
    excludes = splitList(globs.exclude);
  }

  const displayName = setup.name.trim();
  // The id MemForge and the CLI use to address this repository, derived from the
  // name you chose (lowercased, slugified) so it is stable and shell-safe.
  const vaultId = slugify(displayName);
  const profileName = vaultId;

  const addArgs = ["adapter", "kb", "add", profileName, "--root", root, "--vault-id", vaultId];
  for (const pattern of includes) addArgs.push("--include", pattern);
  for (const pattern of excludes) addArgs.push("--exclude", pattern);
  addArgs.push("--display-label", displayName, "--create-source");

  const added = await runStep("Linking repository to MemForge", addArgs);
  if (added?.source_link_error) {
    log.warn(`Saved locally, but couldn't link to a source: ${added.source_link_error}`);
    if (added.detail) log.info(added.detail);
    const unreachable = /unavailable|unreachable|connection|refused|ECONNREFUSED|timed out/i.test(
      `${added.source_link_error} ${added.detail ?? ""}`,
    );
    if (unreachable) {
      note(
        "Couldn't reach your MemForge server. Connect one ('Connect a MemForge server')\nor start the API, then run 'Sync now'.",
        "Server not reachable",
      );
    } else {
      note(
        `Create a 'local_markdown' source in the admin UI with this repository id, then\nrun 'Sync now':\n  ${vaultId}`,
        "Finish linking in the admin UI",
      );
    }
    return;
  }
  if (added?.source_id) {
    log.success(`Linked to source ${added.source_id}${added.source_reused ? " (reused existing)" : ""}`);
  }

  const doPush = ensureNotCancelled(
    await confirm({ message: `Push ${found} files to MemForge now?`, initialValue: true }),
  );
  if (!doPush) {
    note("Run 'Sync now' whenever you're ready.", "Setup complete");
    return;
  }
  const processNow = ensureNotCancelled(
    await confirm({ message: "Trigger extraction after the push?", initialValue: false }),
  );
  const pushArgs = ["adapter", "kb", "push", profileName, processNow ? "--process-now" : "--no-process-now"];
  const pushed = await runStep("Pushing files", pushArgs);
  if (pushed?.counts) note(formatCounts(pushed.counts), "Push counts");
  if (Array.isArray(pushed?.failed) && pushed.failed.length) {
    note(pushed.failed.map((entry) => `- ${entry.relative_path}: ${entry.error}`).join("\n"), "Failed files");
  }
  note(`"${displayName}" is set up (CLI id: ${profileName}). Run 'Sync now' anytime to push changes.`, "Done");
}

// --- per-repository actions (operate on an already-chosen repository) ---

async function syncRepo(name) {
  const processNow = ensureNotCancelled(
    await confirm({ message: "Run extraction after the push?", initialValue: false }),
  );
  const args = ["adapter", "kb", "push", name, processNow ? "--process-now" : "--no-process-now"];
  const payload = await runStep("Pushing files", args);
  if (payload?.counts) note(formatCounts(payload.counts), "Push counts");
  if (Array.isArray(payload?.failed) && payload.failed.length) {
    note(payload.failed.map((entry) => `- ${entry.relative_path}: ${entry.error}`).join("\n"), "Failed files");
  }
}

async function previewRepo(name) {
  const limit = ensureNotCancelled(await text({ message: "Max files to list", placeholder: "20" }));
  const args = ["adapter", "kb", "preview", name];
  if (limit && /^\d+$/.test(limit.trim())) args.push("--limit", limit.trim());
  const payload = await runStep("Previewing", args);
  if (payload?.counts) note(formatCounts(payload.counts), `${payload.profile} preview counts`);
  if (Array.isArray(payload?.items) && payload.items.length) {
    note(payload.items.map((item) => `- ${item.relative_path}`).join("\n"), "Sample files");
  }
}

async function scheduleRepo(name) {
  const every = ensureNotCancelled(
    await select({
      message: "How often should it sync?",
      maxItems: MENU_MAX_ITEMS,
      options: [
        { value: "hourly", label: "Hourly" },
        { value: "6h", label: "Every 6 hours" },
        { value: "12h", label: "Every 12 hours" },
        { value: "daily", label: "Daily" },
        { value: "weekly", label: "Weekly" },
        { value: "15m", label: "Every 15 minutes" },
      ],
    }),
  );
  const args = ["adapter", "kb", "schedule", name, "--every", every];
  if (every === "daily" || every === "weekly") {
    const at = ensureNotCancelled(await text({ message: "Time of day (HH:MM)", placeholder: "09:00" }));
    if (at && /^\d{1,2}:\d{2}$/.test(at.trim())) args.push("--at", at.trim());
  }
  const payload = await runStep("Scheduling sync", args);
  if (payload?.cron) note(`cron: ${payload.cron}\n${payload.command || ""}`, "Scheduled");
}

async function unscheduleRepo(name) {
  const payload = await runStep("Removing schedule", ["adapter", "kb", "unschedule", name]);
  if (payload && payload.removed === false) note("No schedule was installed.", "Nothing to remove");
}

async function removeRepo(name) {
  const confirmRemove = ensureNotCancelled(
    await confirm({ message: `Remove repository '${name}'? (local profile only; the server source stays)`, initialValue: false }),
  );
  if (!confirmRemove) return false;
  await runStep("Removing repository", ["adapter", "kb", "remove", name]);
  return true;
}

// Per-repository submenu: every action for one repository lives here.
async function runRepoMenu(name) {
  while (true) {
    const profile = parseJson((await runMemforge(["adapter", "kb", "list"])).stdout)?.profiles?.[name];
    if (!profile) return; // removed elsewhere
    const scheduled = Boolean(profile.schedule);
    header();
    let choice;
    try {
      choice = ensureNotCancelled(
        await select({
          message: `Repository: ${name}`,
          options: [
            { value: "sync", label: "Sync now", hint: "push new and changed files" },
            { value: "preview", label: "Preview (dry run)", hint: "show what would sync" },
            {
              value: "schedule",
              label: "Schedule sync",
              hint: scheduled ? `replace ${profile.schedule}` : "run automatically on a timer",
            },
            ...(scheduled ? [{ value: "unschedule", label: "Remove schedule" }] : []),
            { value: "remove", label: "Remove repository", hint: "local profile only" },
            { value: "__back__", label: "← Back" },
          ],
        }),
      );
    } catch (error) {
      if (error === BACK) return;
      throw error;
    }
    if (choice === "__back__") return;
    try {
      if (choice === "sync") await syncRepo(name);
      else if (choice === "preview") await previewRepo(name);
      else if (choice === "schedule") await scheduleRepo(name);
      else if (choice === "unschedule") await unscheduleRepo(name);
      else if (choice === "remove" && (await removeRepo(name))) return; // back to the repo list
    } catch (error) {
      if (error === BACK) continue; // ESC inside a per-repo action returns here
      throw error;
    }
    await pause();
  }
}

// --- top-level local-repository actions ---

async function actionSyncNow() {
  const name = await pickKbProfile("Sync which repository?");
  if (name) await syncRepo(name);
}

async function actionManageRepositories() {
  while (true) {
    const profiles = parseJson((await runMemforge(["adapter", "kb", "list"])).stdout)?.profiles ?? {};
    const names = Object.keys(profiles);
    if (!names.length) {
      log.warn("No repositories configured yet. Use 'Set up a repository' first.");
      return;
    }
    header();
    let choice;
    try {
      choice = ensureNotCancelled(
        await select({
          message: "Manage which repository?",
          maxItems: MENU_MAX_ITEMS,
          options: [
            ...names.map((name) => ({
              value: name,
              label: name,
              hint: `${profiles[name]?.root || "?"}${profiles[name]?.schedule ? ` · ${profiles[name].schedule}` : ""}`,
            })),
            { value: "__back__", label: "← Back" },
          ],
        }),
      );
    } catch (error) {
      if (error === BACK) return;
      throw error;
    }
    if (choice === "__back__") return;
    await runRepoMenu(choice);
  }
}

// ---------------------------------------------------------------------------
// Area: Jira
// ---------------------------------------------------------------------------

function formatJiraStatus(status) {
  const rows = [
    ["origin", status.origin],
    ["status", status.status],
    ["principal", status.principal_name || status.principal_email || status.principal_id],
    ["browser", status.browser],
    ["captured", status.captured_at],
    ["validated", status.validated_at],
    ["last error", status.last_error],
  ];
  return rows.filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`).join("\n");
}

async function actionJiraStatus() {
  const baseUrl = await pickBrowserOrigin("jira", "Check status for which Jira origin?");
  if (!baseUrl) return;
  const payload = await runStep("Jira session status", ["adapter", "auth", "jira", "status", "--base-url", baseUrl]);
  if (payload) note(formatJiraStatus(payload), `Jira session: ${payload.status || "unknown"}`);
}

async function actionAuthJira() {
  const baseUrl = await pickBrowserOrigin("jira", "Authenticate which Jira origin?");
  if (!baseUrl) return;
  const browser = ensureNotCancelled(
    await select({
      message: "Browser to read cookies from",
      options: [
        { value: "", label: "Auto-detect" },
        { value: "chrome", label: "Chrome" },
        { value: "edge", label: "Edge" },
        { value: "safari", label: "Safari" },
        { value: "firefox", label: "Firefox" },
      ],
    }),
  );

  const args = ["adapter", "auth", "jira", "refresh", "--base-url", baseUrl];
  if (browser) args.push("--browser", browser);

  const payload = await runStep("Refreshing Jira browser session", args);
  if (payload?.error === "principal_changed") {
    const confirmChange = ensureNotCancelled(
      await confirm({ message: "A different Jira user is signed in. Confirm principal change?", initialValue: false }),
    );
    if (confirmChange) {
      await runStep("Refreshing Jira browser session (confirmed)", [...args, "--confirm-principal-change"]);
    }
  }
}

async function actionJiraForget() {
  const baseUrl = await pickBrowserOrigin("jira", "Forget which Jira session?", { allowNew: false });
  if (!baseUrl) return;
  const confirmForget = ensureNotCancelled(
    await confirm({ message: `Forget the stored session for ${baseUrl}?`, initialValue: false }),
  );
  if (confirmForget) await runStep("Forgetting Jira session", ["adapter", "auth", "jira", "forget", "--base-url", baseUrl]);
}

async function actionJiraWatch() {
  const baseUrl = await pickBrowserOrigin("jira", "Keep which Jira origin fresh?");
  if (!baseUrl) return;
  note(
    `The watch daemon runs in the foreground and re-captures your Jira session on a timer.
Run it in its own terminal (or under launchd/systemd) so it keeps the server's session fresh:

  memforge adapter auth jira watch --base-url ${baseUrl}`,
    "Background refresh",
  );
  const startNow = ensureNotCancelled(
    await confirm({ message: "Start it here now? (blocks this menu until you stop it)", initialValue: false }),
  );
  if (startNow) {
    log.info("Starting watch. Press Ctrl-C to stop and return to the menu.");
    await new Promise((resolve) => {
      const env = { ...process.env, MEMFORGE_NO_INTERACTIVE: "1" };
      const child = spawn(MEMFORGE_BIN, ["adapter", "auth", "jira", "watch", "--base-url", baseUrl], {
        env,
        stdio: "inherit",
      });
      child.on("error", resolve);
      child.on("close", resolve);
    });
  }
}

// ---------------------------------------------------------------------------
// Area: Search memory
// ---------------------------------------------------------------------------

async function actionSearch() {
  const answers = await group(
    {
      query: () => text({ message: "Search query", validate: required }),
      topK: () => text({ message: "Top K", placeholder: "10" }),
      includeSuperseded: () => confirm({ message: "Include superseded memories?", initialValue: false }),
    },
    { onCancel: goBack },
  );

  const args = ["memory", "search", answers.query.trim()];
  if (answers.topK && /^\d+$/.test(answers.topK.trim())) args.push("--top-k", answers.topK.trim());
  if (answers.includeSuperseded) args.push("--include-superseded");

  const payload = await runStep("Searching", args);
  if (Array.isArray(payload?.results) && payload.results.length) {
    note(
      payload.results
        .slice(0, 10)
        .map((row) => {
          const id = row.memory_id || row.id || "(no id)";
          const summary = row.summary || row.content || "";
          return `- ${id}: ${summary.slice(0, 120)}`;
        })
        .join("\n"),
      "Top results",
    );
  } else if (payload && Array.isArray(payload?.results)) {
    note("No results", "Top results");
  }
}

// ---------------------------------------------------------------------------
// Area: Status & diagnostics
// ---------------------------------------------------------------------------

async function actionAdapterStatus() {
  const status = await runStep("Adapter status", ["adapter", "status"]);
  if (status?.capabilities) note(status.capabilities.join("\n"), "Adapter capabilities");
  const profiles = parseJson((await runMemforge(["adapter", "kb", "list"])).stdout)?.profiles ?? {};
  const names = Object.keys(profiles);
  note(names.length ? names.join("\n") : "(none)", "Configured repositories");
}

async function actionDiagnostics() {
  await runStep("Adapter status", ["adapter", "status"]);
  await runStep("Health check", ["target", "check"]);
}

// ---------------------------------------------------------------------------
// Menu structure (area -> actions)
// ---------------------------------------------------------------------------

const AREAS = [
  {
    value: "server",
    label: "Connect a MemForge server",
    hint: "where your memories are stored",
    actions: [
      { value: "connect", label: "Connect a server", hint: "add and activate a target", run: actionConnectServer },
      { value: "switch", label: "Switch active server", hint: "choose an existing target", run: actionSwitchServer },
      { value: "check", label: "Health check", hint: "probe the active server", run: actionHealthCheck },
    ],
  },
  {
    value: "markdown",
    label: "Local repository",
    hint: "sync a local folder (md, txt, json, html) into memory",
    // Until a repository exists, the only useful action is to set one up; the
    // rest appear once at least one is configured.
    actions: async () => {
      const hasRepos =
        Object.keys(parseJson((await runMemforge(["adapter", "kb", "list"])).stdout)?.profiles ?? {}).length > 0;
      const setup = {
        value: "setup",
        label: "Set up a repository",
        hint: "guided: folder → link → first sync",
        run: actionSetupRepository,
      };
      if (!hasRepos) return [setup];
      return [
        { value: "manage", label: "Manage repositories", hint: "per-repo: sync, preview, schedule, remove", run: actionManageRepositories, quiet: true },
        { value: "sync", label: "Sync now", hint: "push new and changed files", run: actionSyncNow },
        setup,
      ];
    },
  },
  {
    value: "jira",
    label: "Jira",
    hint: "let the server sync Jira as you",
    actions: [
      { value: "status", label: "Check session status", hint: "is the current login still valid?", run: actionJiraStatus },
      { value: "auth", label: "Authenticate browser session", hint: "hand over your Jira cookies", run: actionAuthJira },
      { value: "forget", label: "Forget a session", hint: "delete a stored browser session", run: actionJiraForget },
      { value: "watch", label: "Start background refresh", hint: "keep the session fresh on a timer", run: actionJiraWatch },
    ],
  },
  {
    value: "search",
    label: "Search memory",
    hint: "find stored facts and decisions",
    actions: [{ value: "search", label: "Search", hint: "query stored memories", run: actionSearch }],
  },
  {
    value: "status",
    label: "Status & diagnostics",
    hint: "connection, capabilities, sources",
    actions: [
      { value: "status", label: "Adapter capabilities & repositories", hint: "read-only status", run: actionAdapterStatus },
      { value: "diagnostics", label: "Run diagnostics", hint: "adapter status + server check", run: actionDiagnostics },
    ],
  },
];

async function runArea(area) {
  while (true) {
    header();
    const actions = typeof area.actions === "function" ? await area.actions() : area.actions;
    let choice;
    try {
      choice = ensureNotCancelled(
        await select({
          message: area.label,
          maxItems: MENU_MAX_ITEMS,
          options: [
            ...actions.map(({ value, label, hint }) => ({ value, label, hint })),
            { value: "__back__", label: "← Back" },
          ],
        }),
      );
    } catch (error) {
      if (error === BACK) return; // ESC at the area menu goes up to the top menu
      throw error;
    }
    if (choice === "__back__") return;
    const action = actions.find((item) => item.value === choice);
    if (!action) continue;
    try {
      await action.run();
    } catch (error) {
      if (error === BACK) continue; // ESC inside an action returns to this menu
      log.error(`${action.label}: ${error?.message || error}`);
      await pause(); // let the user read the error before the screen clears
      continue;
    }
    // Leaf actions print output, so pause to keep it on screen. Submenu actions
    // (quiet) only navigate, so returning from them should not need a keypress.
    if (!action.quiet) await pause();
  }
}

async function main() {
  // Until a MemForge server is reachable, the only useful action is to connect
  // one; everything else needs the backend, so don't offer the full menu yet.
  while (!(await probeServerReachable())) {
    header();
    let choice;
    try {
      choice = ensureNotCancelled(
        await select({
          message: "No MemForge server reachable. Connect one to continue.",
          options: [
            { value: "connect", label: "Connect a MemForge server", hint: "where your memories are stored" },
            { value: "__quit__", label: "Quit" },
          ],
        }),
      );
    } catch (error) {
      if (error === BACK) { outro("Bye"); return; } // nothing above the gate, so ESC quits
      throw error;
    }
    if (choice === "__quit__") {
      outro("Bye");
      return;
    }
    try {
      await actionConnectServer();
    } catch (error) {
      if (error !== BACK) throw error; // ESC during connect just re-shows the gate
    }
  }
  activeServer = await resolveServerAddress();

  while (true) {
    header();
    let choice;
    try {
      choice = ensureNotCancelled(
        await select({
          message: "Choose an area",
          maxItems: MENU_MAX_ITEMS,
          options: [
            ...AREAS.map(({ value, label, hint }) => ({ value, label, hint })),
            { value: "__quit__", label: "Quit" },
          ],
        }),
      );
    } catch (error) {
      if (error === BACK) { outro("Bye"); return; } // ESC at the top menu quits
      throw error;
    }
    if (choice === "__quit__") {
      outro("Bye");
      return;
    }
    const area = AREAS.find((item) => item.value === choice);
    if (area) {
      await runArea(area);
      // The server area can switch/add a target, so refresh the header label.
      if (area.value === "server") activeServer = await resolveServerAddress();
    }
  }
}

main().catch((error) => {
  if (error === BACK) return; // unwound above the top menu; nothing left to do
  process.stderr.write(`memforge interactive crashed: ${error?.stack || error}\n`);
  process.exit(1);
});
