import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const detail = readFileSync(
  fileURLToPath(new URL("../src/views/review/ReviewDetailPage.tsx", import.meta.url)),
  "utf8",
);
const queue = readFileSync(
  fileURLToPath(new URL("../src/views/review/ReviewQueuePage.tsx", import.meta.url)),
  "utf8",
);

assert.match(detail, /Use latest state/);
assert.match(detail, /Keep current state/);
assert.match(detail, /View technical details/);
assert.match(detail, /This proposal expired/);
assert.doesNotMatch(detail, /\/refresh/);
assert.doesNotMatch(detail, />\s*Approve\s*</);
assert.doesNotMatch(detail, />\s*Reject\s*</);
assert.doesNotMatch(detail, /SplitDiffBlock|diffWords/);

assert.match(queue, /review\.presentation\.summary/);
assert.match(queue, /review\.presentation\.why_human/);
assert.match(queue, /Review decision/);

console.log("review decision UX contract verified");
