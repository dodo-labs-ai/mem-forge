import { expect, test } from "vitest";
import { ApiError } from "@/api";
import { aliasOrigin, canonicalNameHint, entityName, isCurationForbidden } from "./entityPresentation";

const ENTITY = { id: 1, canonical_name: "cut-off", display_name: "Cut-off", created_at: null };

test("names an entity by its display name and hints the canonical name only when it differs", () => {
  expect(entityName(ENTITY)).toBe("Cut-off");
  expect(canonicalNameHint(ENTITY)).toBe("cut-off");
  expect(entityName({ ...ENTITY, display_name: "" })).toBe("cut-off");
  expect(canonicalNameHint({ ...ENTITY, display_name: "" })).toBeNull();
  expect(canonicalNameHint({ ...ENTITY, display_name: "cut-off" })).toBeNull();
});

test("tells an admin's alias from one MemForge found", () => {
  const alias = { alias: "Cutoff", alias_normalized: "cutoff", created_at: null };
  expect(aliasOrigin({ ...alias, source: "admin_manual" })).toBe("Added by an admin");
  expect(aliasOrigin({ ...alias, source: "deterministic" })).toBe("Found by MemForge");
});

test("recognises only the curation refusal", () => {
  expect(isCurationForbidden(new ApiError(403, "Only a workspace admin…", "entity_curation_forbidden"))).toBe(true);
  expect(isCurationForbidden(new ApiError(403, "Forbidden"))).toBe(false);
  expect(isCurationForbidden(new Error("entity_curation_forbidden"))).toBe(false);
});
