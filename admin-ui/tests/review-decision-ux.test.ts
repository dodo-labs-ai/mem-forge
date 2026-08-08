import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildReviewChangeCue } from "../src/views/review/reviewQueueCue.js";

const detail = readFileSync(
  fileURLToPath(new URL("../../src/views/review/ReviewDetailPage.tsx", import.meta.url)),
  "utf8",
);
const queue = readFileSync(
  fileURLToPath(new URL("../../src/views/review/ReviewQueuePage.tsx", import.meta.url)),
  "utf8",
);
const cue = readFileSync(
  fileURLToPath(new URL("../../src/views/review/reviewQueueCue.ts", import.meta.url)),
  "utf8",
);

assert.match(detail, /action\.decision === "approve"/);
assert.match(detail, /action\.decision === "reject"/);
assert.match(detail, /expected_fingerprint: detailQuery\.data\?\.decision_fingerprint/);
assert.match(detail, /\{rejectAction\.label\}/);
assert.match(detail, /\{approveAction\.label\}/);
assert.match(detail, /View technical details/);
assert.match(detail, /This proposal expired/);
assert.doesNotMatch(detail, /\/refresh/);
assert.doesNotMatch(detail, />\s*Approve\s*</);
assert.doesNotMatch(detail, />\s*Reject\s*</);
assert.doesNotMatch(detail, /SplitDiffBlock|diffWords/);

assert.match(queue, /review\.presentation\.summary/);
assert.match(queue, /review\.presentation\.decision_label/);
assert.match(queue, /review\.presentation\.proposed_empty_text/);
assert.match(queue, /buildReviewChangeCue/);
assert.match(queue, /<mark/);
assert.match(queue, /REVIEW_QUEUE_PAGE_SIZE/);
assert.match(queue, /offset: page \* REVIEW_QUEUE_PAGE_SIZE/);
assert.match(queue, /<Pagination/);
assert.match(queue, /searchParams\.get\("page"\)/);
assert.match(queue, /review-queue-filter/);
assert.match(queue, /cross_source_conflict/);
assert.match(queue, /Source lifecycle/);
assert.doesNotMatch(queue, /review\.presentation\.why_human/);
assert.doesNotMatch(queue, /CardContent/);
assert.doesNotMatch(queue, />\s*Review\s*</);
assert.match(cue, /export function buildReviewChangeCue/);

const changedCue = buildReviewChangeCue(
  "The MCP Helper Tool exposes five endpoints for agents.",
  "The MCP Helper Tool exposes six endpoints for agents.",
);
assert.equal(changedCue.current?.changed, "five");
assert.equal(changedCue.proposed?.changed, "six");

const removedCue = buildReviewChangeCue("This memory still has content.", null);
assert.equal(removedCue.current?.changed, "");
assert.equal(removedCue.proposed, null);

console.log("review decision UX contract verified");
