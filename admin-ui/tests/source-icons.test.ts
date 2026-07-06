import assert from "node:assert/strict";

import {
  AGENT_SESSION_CLIENT_MARK,
  BRAND_MARKS,
  SOURCE_TYPE_MARKS,
  sourceBrandKeysFor,
} from "../src/views/sources/sourceBrand.js";

// Every brand mark must carry an accessible label and a complete SVG path:
// a self-closed <path> with a non-empty `d`, so a truncated copy/paste fails.
for (const [key, mark] of Object.entries(BRAND_MARKS)) {
  assert.ok(mark.label.length > 0, `${key} mark should have a label`);
  assert.match(mark.markup, /<path\b[^>]*\bd="[^"]+"\s*\/>/, `${key} mark should embed a complete SVG path`);
}

// Source-type-keyed marks. agent_session resolves per-client; see below.
const SOURCE_TYPES = ["confluence", "jira", "github_pages", "github_repo", "teams", "local_markdown"];
for (const type of SOURCE_TYPES) {
  const keys = SOURCE_TYPE_MARKS[type];
  assert.ok(keys && keys.length > 0, `${type} should have at least one brand mark`);
  for (const key of keys) {
    assert.ok(key in BRAND_MARKS, `${type} references unknown mark "${key}"`);
  }
}

// Source-specific routing the UI depends on.
assert.equal(
  SOURCE_TYPE_MARKS.agent_session,
  undefined,
  "agent_session brand is resolved via AGENT_SESSION_CLIENT_MARK, not SOURCE_TYPE_MARKS",
);
assert.equal(AGENT_SESSION_CLIENT_MARK.codex, "codex");
assert.equal(AGENT_SESSION_CLIENT_MARK["claude-code"], "claude");
assert.deepEqual(sourceBrandKeysFor("agent_session", "codex"), ["codex"]);
assert.deepEqual(sourceBrandKeysFor("agent_session", "claude-code"), ["claude"]);
assert.deepEqual(sourceBrandKeysFor("user_memory", "codex"), ["codex"]);
assert.deepEqual(sourceBrandKeysFor("user_memory", "claude-code"), ["claude"]);
assert.equal(sourceBrandKeysFor("user_memory", null), undefined);
assert.deepEqual(SOURCE_TYPE_MARKS.local_markdown, ["obsidian"], "local markdown should show the Obsidian mark");
assert.deepEqual(SOURCE_TYPE_MARKS.github_pages, ["github"], "GitHub Pages should show the GitHub mark");
assert.deepEqual(SOURCE_TYPE_MARKS.github_repo, ["github"], "GitHub repositories should show the GitHub mark");

// Brand colors are pinned so they can't silently drift. These four match the
// exact hexes the README uses for the same logos.
assert.equal(BRAND_MARKS.confluence.color, "#172B4D");
assert.equal(BRAND_MARKS.jira.color, "#0052CC");
assert.equal(BRAND_MARKS.teams.color, "#6264A7");
assert.equal(BRAND_MARKS.claude.color, "#D97757");
assert.equal(BRAND_MARKS.obsidian.color, "#7C3AED");

// GitHub and Codex (OpenAI mark) are near-black in the README; we pin them to
// null so they inherit the theme foreground and stay visible in dark mode.
assert.equal(BRAND_MARKS.github.color, null, "GitHub mark should inherit the theme foreground");
assert.equal(BRAND_MARKS.codex.color, null, "Codex mark should inherit the theme foreground");

console.log("source-icons.test.ts passed");
