import assert from "node:assert/strict";

import { resolveSourceReadiness } from "../src/views/sources/sourceReadiness.js";

assert.equal(
  resolveSourceReadiness({ localExecution: true, daemon: "checking" }),
  "checking_local_sync",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: true,
    daemon: "unavailable",
    connectionStatus: { state: "action_required", reason: "authentication" },
  }),
  "local_sync_unavailable",
  "device readiness should take precedence when collection runs locally",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: true,
    daemon: "ready",
    connectionStatus: { state: "action_required", reason: "authentication" },
  }),
  "sign_in_required",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: false,
    connectionStatus: { state: "action_required", reason: "configuration" },
  }),
  "configuration_required",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: false,
    connectionStatus: { state: "action_required", reason: "identity_conflict" },
  }),
  "account_mismatch",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: false,
    connectionStatus: { state: "ready", reason: null },
  }),
  null,
  "healthy server-executed sources should not gain a redundant readiness badge",
);
assert.equal(
  resolveSourceReadiness({
    localExecution: true,
    daemon: "ready",
    connectionStatus: { state: "ready", reason: null },
  }),
  "local_sync_ready",
);
assert.equal(
  resolveSourceReadiness({ localExecution: true, daemon: "ready" }),
  "local_sync_ready",
  "local connectors without a separate connection dependency should be ready when their daemon is ready",
);
assert.equal(
  resolveSourceReadiness({ localExecution: false }),
  null,
  "server sources without a connection dependency should not gain another badge",
);
