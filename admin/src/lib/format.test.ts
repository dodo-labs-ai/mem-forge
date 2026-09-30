import { expect, test } from "vitest";
import { formatInterval, formatRelative, formatUntil, pluralize } from "./format";

const NOW = new Date("2026-09-29T12:00:00Z");

test("pluralize", () => {
  expect(pluralize(1, "page")).toBe("1 page");
  expect(pluralize(1200, "memory", "memories")).toBe("1,200 memories");
});

test("formatRelative", () => {
  expect(formatRelative("2026-09-29T11:59:50Z", NOW)).toBe("just now");
  expect(formatRelative("2026-09-29T11:55:00Z", NOW)).toBe("5 min ago");
  expect(formatRelative("2026-09-29T09:00:00Z", NOW)).toBe("3 h ago");
  expect(formatRelative("2026-09-28T12:00:00Z", NOW)).toBe("1 day ago");
});

test("formatUntil", () => {
  expect(formatUntil("2026-09-29T11:00:00Z", NOW)).toBe("now");
  expect(formatUntil("2026-09-29T12:20:00Z", NOW)).toBe("in 20 min");
  expect(formatUntil("2026-09-29T15:00:00Z", NOW)).toBe("in 3 h");
});

test("formatInterval", () => {
  expect(formatInterval(30)).toBe("Every 30 min");
  expect(formatInterval(360)).toBe("Every 6 h");
  expect(formatInterval(1440)).toBe("Daily");
  expect(formatInterval(4320)).toBe("Every 3 days");
});
