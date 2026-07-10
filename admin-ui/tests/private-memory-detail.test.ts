import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const memoryDetailSource = readFileSync("src/views/memories/MemoryDetailPage.tsx", "utf8");
const reviewQueueSource = readFileSync("src/views/review/ReviewQueuePage.tsx", "utf8");

assert.match(
  memoryDetailSource,
  /resourceClient\s*\.\s*get\(`\/memories\/\$\{id\}`,\s*\{\s*params:\s*\{\s*include_private:\s*"true"\s*\}\s*\}\)/s,
  "the memory detail page should include the current user's private rows",
);

assert.match(
  reviewQueueSource,
  /resourceClient\s*\.\s*get\("\/memory-reviews",\s*\{/s,
  "review queue should load review-specific memory snapshots from the review API",
);

assert.doesNotMatch(
  reviewQueueSource,
  /resourceClient[^\n]*\/memories\/\$\{id\}/,
  "review queue should not fetch pending challengers through the normal memory detail API",
);

console.log("private-memory-detail.test.ts: all assertions passed");
