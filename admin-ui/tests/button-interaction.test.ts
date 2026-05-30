import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const buttonSource = readFileSync("src/components/ui/button.tsx", "utf8");
const globalStyles = readFileSync("src/index.css", "utf8");

assert.match(
  buttonSource,
  /cursor-pointer/,
  "button primitive should show a pointer cursor for enabled buttons",
);

assert.match(
  buttonSource,
  /default: "bg-primary text-primary-foreground hover:bg-primary\/90 hover:shadow-sm"/,
  "primary buttons should have visible button hover feedback",
);

assert.match(
  buttonSource,
  /outline:\s+"border-border bg-background hover:border-foreground\/20 hover:bg-muted hover:text-foreground hover:shadow-xs/,
  "outline buttons should visibly respond on hover",
);

assert.match(
  globalStyles,
  /button:not\(:disabled\)\s*{\s*cursor: pointer;\s*}/,
  "enabled native buttons should use a pointer cursor even outside the Button primitive",
);

assert.match(
  globalStyles,
  /button:disabled\s*{\s*cursor: not-allowed;\s*}/,
  "disabled native buttons should keep not-allowed cursor feedback",
);
