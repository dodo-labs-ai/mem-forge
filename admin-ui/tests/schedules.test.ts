import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// Nav item in Sidebar
const sidebarSource = readFileSync("src/components/layout/Sidebar.tsx", "utf8");

assert.match(
  sidebarSource,
  /CalendarClock/,
  "Sidebar should import CalendarClock for the Scheduled Syncs nav item",
);

assert.match(
  sidebarSource,
  /to: "\/schedules"/,
  "Sidebar should include a /schedules nav item",
);

assert.match(
  sidebarSource,
  /label: "Scheduled Syncs"/,
  'Sidebar should label the schedules nav item "Scheduled Syncs"',
);

// Route in App.tsx
const appSource = readFileSync("src/App.tsx", "utf8");

assert.match(
  appSource,
  /SchedulesPage/,
  "App.tsx should import SchedulesPage",
);

assert.match(
  appSource,
  /path="\/schedules"[\s\S]*?SchedulesPage/,
  "App.tsx should register a /schedules route rendered by SchedulesPage",
);

// SchedulesPage rendering logic
const pageSource = readFileSync("src/views/schedules/SchedulesPage.tsx", "utf8");
const extensionSource = readFileSync("src/extension.ts", "utf8");

assert.match(
  pageSource,
  /function normalizeSources/,
  "SchedulesPage should define normalizeSources to handle both response shapes",
);

assert.match(
  pageSource,
  /Array\.isArray\(payload\)/,
  "normalizeSources should handle bare Source[] response",
);

assert.match(
  pageSource,
  /Array\.isArray\(payload\?\.data\)/,
  "normalizeSources should handle wrapped { data: Source[] } response",
);

assert.match(
  pageSource,
  /function formatInterval/,
  "SchedulesPage should define formatInterval for human-readable schedule intervals",
);

assert.match(
  pageSource,
  /interval_minutes/,
  "SchedulesPage should read interval_minutes from SourceSyncSchedule",
);

assert.match(
  pageSource,
  /next_run_at/,
  "SchedulesPage should display next_run_at from the schedule",
);

assert.match(
  pageSource,
  /function scheduleSortKey/,
  "SchedulesPage should sort scheduled sources ahead of unscheduled sources",
);

assert.match(
  pageSource,
  /queryKey.*sources/,
  'SchedulesPage should query the "sources" key to share the cache with SourcesPage',
);

assert.match(
  pageSource,
  /\/api\/sources/,
  "SchedulesPage should fetch from /api/sources",
);

assert.match(
  pageSource,
  /<table/,
  "SchedulesPage should render a table",
);

assert.match(
  pageSource,
  /overflow-x-auto/,
  "SchedulesPage should keep the schedule table horizontally scrollable on narrow screens",
);

assert.match(
  pageSource,
  /disabled for me/,
  "SchedulesPage should surface per-viewer subscription state",
);

assert.match(
  pageSource,
  /AsyncBoundary/,
  "SchedulesPage should wrap the table in AsyncBoundary for loading/error/empty states",
);

assert.match(
  pageSource,
  /onRetry/,
  "SchedulesPage should supply an onRetry handler to AsyncBoundary",
);

assert.match(
  pageSource,
  /to="\/sources"/,
  "SchedulesPage should link back to /sources for source management",
);

assert.match(
  extensionSource,
  /"schedules"/,
  "The /schedules route should be reserved from extension route collisions",
);

console.log("schedules.test.ts: all assertions passed");
