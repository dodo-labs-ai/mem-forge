// Verifies that the package metadata declares @clack/prompts as a dependency
// and that the entry script is loadable.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import url from "node:url";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const packageJson = JSON.parse(readFileSync(path.join(here, "..", "package.json"), "utf-8"));

assert.equal(packageJson.name, "memforge-cli");
assert.equal(packageJson.type, "module");
assert.ok(packageJson.dependencies?.["@clack/prompts"], "@clack/prompts must be declared as a dependency");
assert.match(packageJson.main, /index\.mjs$/);
assert.ok(packageJson.bin?.["memforge-interactive"], "expected a memforge-interactive bin entry");
