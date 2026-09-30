import { describe, expect, it } from "vitest";
import { apiErrorMessage } from "@/api/errors";

function apiError(detail: unknown) {
  return { response: { data: { detail } } };
}

describe("apiErrorMessage", () => {
  it("returns a plain string detail", () => {
    expect(apiErrorMessage(apiError("project key 'PAY' already exists"))).toBe(
      "project key 'PAY' already exists",
    );
  });

  it("returns the message of a structured detail", () => {
    const error = apiError({
      error: "project_management_forbidden",
      message: "Only a workspace admin can create, change, or delete projects.",
    });
    expect(apiErrorMessage(error)).toBe(
      "Only a workspace admin can create, change, or delete projects.",
    );
  });

  it("returns null when the response carries no readable detail", () => {
    expect(apiErrorMessage(new Error("Network Error"))).toBeNull();
    expect(apiErrorMessage(apiError([{ msg: "field required" }]))).toBeNull();
    expect(apiErrorMessage(apiError({ error: "reserved_project" }))).toBeNull();
    expect(apiErrorMessage(null)).toBeNull();
  });
});
