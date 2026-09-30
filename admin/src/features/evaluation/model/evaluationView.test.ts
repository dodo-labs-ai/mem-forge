import { expect, test } from "vitest";
import { makeOverview } from "@/test/evaluationFixtures";
import { buildEvaluationView, criteriaIn } from "./evaluationView";

test("splits issue groups into failures and patterns to check with their occurrences", () => {
  const view = buildEvaluationView(makeOverview(), { criterion: null, label: null });
  expect(view.failGroups.map((group) => group.group_id)).toEqual(["aeg-fail", "aeg-contract"]);
  expect(view.reviewGroups.map((group) => group.group_id)).toEqual(["aeg-review"]);
  expect(view.failOccurrences).toBe(17);
  expect(view.reviewOccurrences).toBe(9);
  expect(view.assessments).toHaveLength(3);
});

test("the criterion and result filters narrow groups, checks and counts together", () => {
  const byCriterion = buildEvaluationView(makeOverview(), { criterion: "evidence_reference_validity", label: null });
  expect(byCriterion.failGroups.map((group) => group.group_id)).toEqual(["aeg-fail"]);
  expect(byCriterion.failOccurrences).toBe(14);
  expect(byCriterion.reviewGroups).toEqual([]);
  expect(byCriterion.assessments.map((assessment) => assessment.assessment_id)).toEqual(["asm-1"]);

  const passing = buildEvaluationView(makeOverview(), { criterion: null, label: "pass" });
  expect(passing.failGroups).toEqual([]);
  expect(passing.assessments.map((assessment) => assessment.assessment_id)).toEqual(["asm-3"]);
});

test("offers every criterion of the loaded scope once, sorted", () => {
  expect(criteriaIn(makeOverview())).toEqual([
    "evidence_localization",
    "evidence_reference_validity",
    "extraction_completion",
    "structured_output_contract",
  ]);
});
