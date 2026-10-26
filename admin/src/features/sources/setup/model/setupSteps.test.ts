import { expect, test } from "vitest";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import { firstGroupWithPurpose } from "../../model/configGroups";
import {
  checkSetupGroup,
  groupComplete,
  groupFields,
  groupSummary,
  issueCountsByGroup,
  setupSteps,
} from "./setupSteps";

const stepKeys = (...args: Parameters<typeof setupSteps>) => setupSteps(...args).map((step) => step.key);

test("the product steps surround the type's groups, and Access appears only when creating in a shared workspace", () => {
  const teams = sourceTypeSchema("teams");

  expect(stepKeys(teams, { mode: "create", sharedWorkspace: true })).toEqual([
    "basics",
    "connection",
    "scope",
    "sync",
    "access",
    "project",
    "schedule",
  ]);
  expect(stepKeys(teams, { mode: "create", sharedWorkspace: false })).not.toContain("access");
  expect(stepKeys(teams, { mode: "edit", sharedWorkspace: true })).not.toContain("access");
  expect(setupSteps(teams, { mode: "edit", sharedWorkspace: false })[1]).toMatchObject({ label: "Teams session", kind: "group" });
});

test("a group's fields are drawn in their declared order", () => {
  expect(groupFields(sourceTypeSchema("jira"), "connection").map((field) => field.key)).toEqual(["base_url", "sync_mode", "auth_mode", "pat"]);
});

test("Configure opens the connection group and Check setup follows the last content group", () => {
  const teams = sourceTypeSchema("teams");

  expect(firstGroupWithPurpose(teams, "connection")).toBe("connection");
  expect(firstGroupWithPurpose(teams, "content")).toBe("scope");
  expect(checkSetupGroup(teams)).toBe("scope");
});

test("an options group is complete with its working defaults, and a content group needs its required fields", () => {
  const teams = sourceTypeSchema("teams");

  expect(groupComplete(teams, "sync", { values: {} })).toBe(true);
  expect(groupComplete(teams, "scope", { values: {} })).toBe(false);
  expect(groupComplete(teams, "scope", { values: { conversation_ids: ["19:a"] } })).toBe(true);
});

test("field issues count against their group", () => {
  expect(issueCountsByGroup(sourceTypeSchema("confluence"), ["base_url", "pat", "spaces", null, "vault_id"])).toEqual({
    connection: 2,
    scope: 1,
  });
});

test("a completed step's summary names its visible values: a URL's host, a choice's label and a list's count", () => {
  const confluence = sourceTypeSchema("confluence");
  const state = {
    values: { base_url: "https://wiki.example.com/confluence", pat: "secret", sync_mode: "space", spaces: ["ENG", "OPS"], exclude_labels: ["draft"] },
  };

  expect(groupSummary(confluence, "connection", state)).toBe("wiki.example.com");
  expect(groupSummary(confluence, "scope", state)).toBe("Whole space · Spaces to Sync: 2");
});
