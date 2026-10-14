import { expect, test } from "vitest";
import { ApiError } from "@/api";
import { actionErrorMessage } from "./actionErrors";

test("translates the reason codes memory actions return", () => {
  expect(actionErrorMessage(new ApiError(409, "content_hash_mismatch"))).toBe(
    "Someone changed this memory after you opened it. Reload to see the latest version before you try again.",
  );
  expect(actionErrorMessage(new ApiError(409, "source_backed_memory_requires_lifecycle_review"))).toBe(
    "This memory is backed by a source. Remove it from the source, or propose a correction.",
  );
});

test("keeps the explanation of a change that waits for the search index", () => {
  const waiting = new ApiError(409, "The search index is being rebuilt, so this change has to wait.", "reindex_in_progress");

  expect(actionErrorMessage(waiting)).toBe("The search index is being rebuilt, so this change has to wait.");
});

test("keeps any other message the server gives", () => {
  expect(actionErrorMessage(new ApiError(500, "The request failed with status 500."))).toBe("The request failed with status 500.");
});
