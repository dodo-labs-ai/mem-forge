import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { extname, join } from "node:path";

import {
  createWorkspaceApiController,
  resourceClient,
} from "../src/api/client.js";

let clearCount = 0;
const queryClient = {
  clear() {
    clearCount += 1;
  },
};

const workspaceApi = createWorkspaceApiController(queryClient);
assert.equal(resourceClient.defaults.baseURL, "/api");
assert.equal(workspaceApi.current(), null);

workspaceApi.setTarget({
  resourceBaseUrl: "/api/workspaces/mount_tai/api/",
  localAgentBaseUrl: "/api/cloud/workspaces/mount_tai/local-agent/",
});
assert.equal(resourceClient.defaults.baseURL, "/api/workspaces/mount_tai/api");
assert.deepEqual(workspaceApi.current(), {
  resourceBaseUrl: "/api/workspaces/mount_tai/api",
  localAgentBaseUrl: "/api/cloud/workspaces/mount_tai/local-agent",
});
assert.equal(clearCount, 1);

workspaceApi.setTarget({
  resourceBaseUrl: "/api/workspaces/mount_tai/api",
  localAgentBaseUrl: "/api/cloud/workspaces/mount_tai/local-agent",
});
assert.equal(clearCount, 1, "a semantically identical target must preserve the query cache");

workspaceApi.setTarget(null);
assert.equal(resourceClient.defaults.baseURL, "/api");
assert.equal(workspaceApi.current(), null);
assert.equal(clearCount, 2);

workspaceApi.setTarget(null);
assert.equal(clearCount, 2, "resetting an already standalone target must preserve the query cache");

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return [".ts", ".tsx"].includes(extname(entry.name)) ? [path] : [];
  });
}

const apiPrefixedResourceCall =
  /resourceClient\s*\.\s*(?:get|post|put|patch|delete)\s*(?:<[^;()]*>\s*)?\(\s*["'`]\/api\//m;
const offenders = sourceFiles("src").filter((path) =>
  apiPrefixedResourceCall.test(readFileSync(path, "utf8")),
);
assert.deepEqual(
  offenders,
  [],
  `resourceClient calls must use API-base-relative paths: ${offenders.join(", ")}`,
);

const legacyClientCall = /\bclient\s*\.\s*(?:get|post|put|patch|delete)\s*[(<]/m;
const legacyClientOffenders = sourceFiles("src").filter((path) =>
  legacyClientCall.test(readFileSync(path, "utf8")),
);
assert.deepEqual(
  legacyClientOffenders,
  [],
  `resource calls must use the named resourceClient boundary: ${legacyClientOffenders.join(", ")}`,
);

const localAgentJobsSource = readFileSync("src/api/localAgentJobs.ts", "utf8");
assert.match(localAgentJobsSource, /hostClient/);
assert.doesNotMatch(localAgentJobsSource, /workspace_id|requireCurrentWorkspaceId/);

console.log("api-target.test.ts: all assertions passed");
