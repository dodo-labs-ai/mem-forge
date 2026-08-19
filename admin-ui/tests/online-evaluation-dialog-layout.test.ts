import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const dialogSource = readFileSync(
  "src/views/sources/SourceOnlineEvaluationDialog.tsx",
  "utf8",
);

assert.match(
  dialogSource,
  /<DialogContent className="[^"]*max-h-\[calc\(100dvh-2rem\)\][^"]*overflow-hidden[^"]*sm:max-w-4xl[^"]*">/,
  "the online evaluation dialog should remain bounded to the dynamic viewport",
);

assert.match(
  dialogSource,
  /<div className="flex min-h-0 flex-col gap-4">[\s\S]*?<div className="min-h-0 flex-1 overflow-y-auto rounded-lg border">/,
  "only the selected evaluation view should own the dialog's remaining scroll area",
);

assert.match(
  dialogSource,
  /Needs attention \([\s\S]*Review queue \([\s\S]*Coverage[\s\S]*All checks/,
  "actionable issue groups and coverage should precede the pass-heavy audit list",
);

assert.match(
  dialogSource,
  /const \[days, setDays\] = useState\(1\)/,
  "live traffic should default to a 24-hour window",
);

assert.match(
  dialogSource,
  /Investigate with agent/,
  "representative cases should provide a bounded agent-investigation handoff",
);

assert.doesNotMatch(
  dialogSource,
  /eligible runtime events do not have a durable assessment yet/,
  "coverage health must not be presented as an unexplained user action queue",
);

console.log("online-evaluation-dialog-layout.test.ts: all assertions passed");
