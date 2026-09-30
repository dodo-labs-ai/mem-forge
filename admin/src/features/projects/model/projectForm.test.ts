import { describe, expect, test } from "vitest";
import { createProjectBody, createProjectSchema, editProjectSchema, RESERVED_CODE_MESSAGE } from "./projectForm";

describe("createProjectSchema", () => {
  test("trims the name and code", () => {
    expect(createProjectSchema.parse({ name: "  Payments platform ", code: " PAY " })).toEqual({
      name: "Payments platform",
      code: "PAY",
    });
  });

  test("requires a name", () => {
    const result = createProjectSchema.safeParse({ name: "   ", code: "" });
    expect(result.success).toBe(false);
    expect(result.error?.issues[0]?.path).toEqual(["name"]);
  });

  test("refuses the codes of built-in projects in any case", () => {
    for (const code of ["SHARED", "unsorted"]) {
      const result = createProjectSchema.safeParse({ name: "Team", code });
      expect(result.error?.issues[0]?.message).toBe(RESERVED_CODE_MESSAGE);
    }
  });
});

test("editing requires a name", () => {
  expect(editProjectSchema.safeParse({ name: "" }).success).toBe(false);
  expect(editProjectSchema.parse({ name: " Payroll " })).toEqual({ name: "Payroll" });
});

test("sends a code only when one was given, so the server derives it otherwise", () => {
  expect(createProjectBody({ name: "Payments platform", code: "" })).toEqual({ name: "Payments platform" });
  expect(createProjectBody({ name: "Payments platform", code: "PAY" })).toEqual({ name: "Payments platform", key: "PAY" });
});
