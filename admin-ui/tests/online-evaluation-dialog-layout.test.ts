import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const dialogSource = readFileSync(
  "src/views/sources/SourceOnlineEvaluationDialog.tsx",
  "utf8",
);
const presentationSource = readFileSync(
  "src/views/evaluation/OnlineEvaluationPresentation.tsx",
  "utf8",
);
const pageSource = readFileSync(
  "src/views/evaluation/OnlineEvaluationPage.tsx",
  "utf8",
);

assert.match(
  dialogSource,
  /<DialogContent className="[^"]*max-h-\[calc\(100dvh-2rem\)\][^"]*overflow-hidden[^"]*sm:max-w-4xl[^"]*">/,
  "the online evaluation dialog should remain bounded to the dynamic viewport",
);

assert.match(
  presentationSource,
  /<div className="flex min-h-0 flex-1 flex-col gap-4">[\s\S]*?<div className="min-h-0 flex-1 overflow-y-auto rounded-lg border">/,
  "only the selected evaluation view should own the dialog's remaining scroll area",
);

assert.match(
  presentationSource,
  /Needs attention \([\s\S]*Review queue \([\s\S]*Sources \([\s\S]*Coverage[\s\S]*All checks/,
  "actionable issue groups and coverage should precede the pass-heavy audit list",
);

assert.match(
  pageSource,
  /Number\(searchParams\.get\("days"\) \|\| "1"\)/,
  "live traffic should default to a 24-hour window",
);

assert.match(
  presentationSource,
  /Investigate with agent/,
  "representative cases should provide a bounded agent-investigation handoff",
);

assert.doesNotMatch(
  `${dialogSource}\n${presentationSource}\n${pageSource}`,
  /eligible runtime events do not have a durable assessment yet/,
  "coverage health must not be presented as an unexplained user action queue",
);

console.log("online-evaluation-dialog-layout.test.ts: all assertions passed");
