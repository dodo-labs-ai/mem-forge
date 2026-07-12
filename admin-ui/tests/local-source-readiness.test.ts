import assert from "node:assert/strict";

import { resolveLocalSourceReadiness } from "../src/views/sources/localSourceReadiness.js";

assert.equal(
  resolveLocalSourceReadiness({ daemon: "checking" }),
  "checking",
);
assert.equal(
  resolveLocalSourceReadiness({
    daemon: "unavailable",
    connectionStatus: { state: "action_required", reason: "authentication" },
  }),
  "unavailable",
  "device readiness should take precedence while the daemon is unavailable",
);
assert.equal(
  resolveLocalSourceReadiness({
    daemon: "ready",
    connectionStatus: { state: "action_required", reason: "authentication" },
  }),
  "sign_in_required",
);
assert.equal(
  resolveLocalSourceReadiness({
    daemon: "ready",
    connectionStatus: { state: "action_required", reason: "configuration" },
  }),
  "configuration_required",
);
assert.equal(
  resolveLocalSourceReadiness({
    daemon: "ready",
    connectionStatus: { state: "action_required", reason: "identity_conflict" },
  }),
  "account_mismatch",
);
assert.equal(
  resolveLocalSourceReadiness({
    daemon: "ready",
    connectionStatus: { state: "ready", reason: null },
  }),
  "ready",
);
assert.equal(
  resolveLocalSourceReadiness({ daemon: "ready" }),
  "ready",
  "local connectors without a separate connection dependency should be ready when their daemon is ready",
);
