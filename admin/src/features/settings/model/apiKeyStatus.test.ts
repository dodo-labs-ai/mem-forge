import { expect, test } from "vitest";
import { apiKeyStatus } from "./apiKeyStatus";

const SAVED = { set: true, last4: "a91f" };
const NONE = { set: false, last4: null };

test("a saved key shows only its last four characters and can be removed", () => {
  expect(apiKeyStatus(SAVED, "keep")).toEqual({
    message: "Saved key (****a91f) in use. Leave blank to keep it.",
    warning: false,
    action: "remove",
  });
});

test("typing over a saved key says it will be replaced", () => {
  expect(apiKeyStatus(SAVED, "replace").message).toBe("Will replace saved key (****a91f) on save.");
});

test("a pending removal warns and can be undone", () => {
  expect(apiKeyStatus(SAVED, "remove")).toEqual({
    message: "Saved key will be removed on save.",
    warning: true,
    action: "undo",
  });
});

test("without a saved key", () => {
  expect(apiKeyStatus(NONE, "replace").message).toBe("New key will be saved.");
  expect(apiKeyStatus(NONE, "keep")).toEqual({
    message: "Leave blank only for local endpoints that do not require auth.",
    warning: false,
    action: null,
  });
});

test("a saved key without its last characters is still named", () => {
  expect(apiKeyStatus({ set: true, last4: null }, "keep").message).toBe("Saved key in use. Leave blank to keep it.");
});
