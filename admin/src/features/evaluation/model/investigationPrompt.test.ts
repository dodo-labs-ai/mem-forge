import { expect, test } from "vitest";
import { makeCase, makeIssueGroup } from "@/test/evaluationFixtures";
import { buildInvestigationPrompt } from "./investigationPrompt";

test("the prompt carries lineage and versions and marks missing IDs as none", () => {
  const prompt = buildInvestigationPrompt("Architecture docs", makeIssueGroup(), makeCase());
  const lines = prompt.split("\n");
  expect(lines[0]).toBe("Investigate this MemForge online-evaluation case read-only.");
  expect(lines).toContain("Source: Architecture docs (src-docs, github_pages)");
  expect(lines).toContain("Signal: evidence_reference_validity / unknown_evidence_block_id / fail");
  expect(lines).toContain("Observation: none");
  expect(lines).toContain("Evaluator: memforge.deterministic.runtime_contract:3");
  expect(lines).toContain("Runtime contract: extraction-v7");
  expect(lines.at(-1)).toBe("Deployment: 2026.09.28-1");
});
