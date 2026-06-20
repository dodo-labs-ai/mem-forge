import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const memoryDetailSource = readFileSync("src/views/memories/MemoryDetailPage.tsx", "utf8");
const reviewQueueSource = readFileSync("src/views/review/ReviewQueuePage.tsx", "utf8");

assert.match(
  memoryDetailSource,
  /client\s*\.\s*get\(`\/api\/memories\/\$\{id\}`,\s*\{\s*params:\s*\{\s*include_private:\s*"true"\s*\}\s*\}\)/s,
  "the memory detail page should include the current user's private rows",
);

assert.match(
  reviewQueueSource,
  /client\s*\.\s*get\(`\/api\/memories\/\$\{id\}`,\s*\{\s*params:\s*\{\s*include_private:\s*"true"\s*\}\s*\}\)/s,
  "review queue memory snapshots should include the current user's private rows",
);

console.log("private-memory-detail.test.ts: all assertions passed");
