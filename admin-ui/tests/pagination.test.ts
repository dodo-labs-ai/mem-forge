import assert from "node:assert/strict";

import { getPageMeta } from "../src/lib/pagination.js";

// Empty result set: no pages, no navigation.
assert.deepEqual(getPageMeta(0, 50, 0), {
  totalPages: 0,
  pageStart: 0,
  pageEnd: 0,
  hasPrev: false,
  hasNext: false,
});

// Single partial page (30 of 50): one page, no navigation either way.
assert.deepEqual(getPageMeta(0, 50, 30), {
  totalPages: 1,
  pageStart: 1,
  pageEnd: 30,
  hasPrev: false,
  hasNext: false,
});

// First page of 249 at size 50: 5 pages, next only.
assert.deepEqual(getPageMeta(0, 50, 249), {
  totalPages: 5,
  pageStart: 1,
  pageEnd: 50,
  hasPrev: false,
  hasNext: true,
});

// Middle page: both directions available, range is interior.
assert.deepEqual(getPageMeta(2, 50, 249), {
  totalPages: 5,
  pageStart: 101,
  pageEnd: 150,
  hasPrev: true,
  hasNext: true,
});

// Last (partial) page: prev only, range ends at the total.
assert.deepEqual(getPageMeta(4, 50, 249), {
  totalPages: 5,
  pageStart: 201,
  pageEnd: 249,
  hasPrev: true,
  hasNext: false,
});

// Exact multiple: final full page reports no next.
assert.deepEqual(getPageMeta(1, 50, 100), {
  totalPages: 2,
  pageStart: 51,
  pageEnd: 100,
  hasPrev: true,
  hasNext: false,
});

// Out-of-range page never reports rows past the total.
assert.deepEqual(getPageMeta(9, 50, 120), {
  totalPages: 3,
  pageStart: 120,
  pageEnd: 120,
  hasPrev: true,
  hasNext: false,
});

console.log("pagination.test.ts: all assertions passed");
