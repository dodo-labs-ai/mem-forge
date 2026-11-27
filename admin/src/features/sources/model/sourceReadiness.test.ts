import assert from "node:assert/strict";
import { makeSource } from "@/test/sourceFixtures";
import { isLocalAgentBackedSource } from "./localAgentSources";
import { resolveSourceReadiness, sourceDeviceReadiness } from "./sourceReadiness";
import type { SourceConnectionStatus } from "./types";
import { test } from "vitest";

test("resolves source readiness from local sync and connection state", () => {
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
});

/** A Source on its execution owner's device, as the source API lists it. */
function deviceSource(state: "online" | "offline" | "update_required", connection: SourceConnectionStatus) {
  return makeSource({
    execution: {
      kind: "local_agent",
      operation: "document_acquisition",
      immutable_config_fields: [],
      device: { id: "dev-macbook", name: "Alex's MacBook", state, not_ready: [], installed_version: "0.2.0b8", required_version: "0.2.0b7", credential: "secret" },
    },
    connection_status: connection,
  });
}

const READY: SourceConnectionStatus = { state: "ready", reason: null };
const NO_CREDENTIAL: SourceConnectionStatus = { state: "action_required", reason: "authentication" };

test.each([
  ["online", READY, "local_sync_ready"],
  ["offline", READY, "local_sync_unavailable"],
  ["update_required", READY, "update_local_sync"],
  ["online", NO_CREDENTIAL, "sign_in_required"],
  ["offline", NO_CREDENTIAL, "local_sync_unavailable"],
  ["update_required", NO_CREDENTIAL, "update_local_sync"],
] as const)("a device Source whose device is %s with connection %o reads %s", (state, connection, expected) => {
  const source = deviceSource(state, connection);
  // The viewer's own daemon says nothing about the execution owner's device.
  for (const daemon of ["checking", "ready", "unavailable"] as const) {
    assert.equal(
      resolveSourceReadiness({
        localExecution: isLocalAgentBackedSource(source),
        daemon: sourceDeviceReadiness(source, daemon),
        connectionStatus: source.connection_status,
      }),
      expected,
    );
  }
});

test("update local sync comes after offline and checking and before every connection blocker", () => {
  assert.equal(resolveSourceReadiness({ localExecution: true, daemon: "update_required", connectionStatus: { state: "action_required", reason: "identity_conflict" } }), "update_local_sync");
  assert.equal(resolveSourceReadiness({ localExecution: true, daemon: "update_required", connectionStatus: { state: "action_required", reason: "configuration" } }), "update_local_sync");
  assert.equal(resolveSourceReadiness({ localExecution: true, daemon: "unavailable", connectionStatus: READY }), "local_sync_unavailable");
  assert.equal(resolveSourceReadiness({ localExecution: true, daemon: "checking", connectionStatus: READY }), "checking_local_sync");
});

test("a Source on its type's local path reads the viewer's own daemon", () => {
  const source = makeSource({ execution: { kind: "local_agent", operation: "teams_sync", immutable_config_fields: [] } });
  assert.equal(sourceDeviceReadiness(source, "unavailable"), "unavailable");
  assert.equal(sourceDeviceReadiness(makeSource(), "ready"), "ready");
});
