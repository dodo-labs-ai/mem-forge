import assert from "node:assert/strict";

import { MEMORY_TYPES, MEMORY_TYPE_ICON_COLOR, MEMORY_TYPE_LABEL } from "../src/views/memories/memoryTypeMeta.js";

// All four memory types are covered.
assert.deepEqual(
  [...MEMORY_TYPES].sort(),
  ["convention", "decision", "fact", "procedure"],
  "memory type list should cover every memory_type the API returns",
);

// Every type has a tint and a label.
for (const type of MEMORY_TYPES) {
  assert.match(MEMORY_TYPE_ICON_COLOR[type], /^text-/, `${type} should have a text-color tint`);
  assert.ok(MEMORY_TYPE_LABEL[type]?.length > 0, `${type} should have a display label`);
}

// Tints are distinct so types are visually separable at a glance.
const colors = MEMORY_TYPES.map((type) => MEMORY_TYPE_ICON_COLOR[type]);
assert.equal(new Set(colors).size, colors.length, "each memory type should have a distinct icon color");

console.log("memory-type-icons.test.ts passed");
