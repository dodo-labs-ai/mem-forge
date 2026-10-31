"""Every rubric of the public quality cohort takes the forms its case kind allows.

The expectation judge reads each case kind through a closed set of sentence
forms (``memforge.evals.expectation_judge``). A contributor's new case fills
one of them; this test refuses, without a model, an expectation in another
form or about an old memory its reconciliation case does not hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memforge.evals.expectation_judge import validate_expectations

QUICK = Path(__file__).resolve().parents[1] / "evals" / "quality" / "quick"
_COHORT = json.loads((QUICK / "cohort.json").read_text(encoding="utf-8"))
_RUBRICS = [
    *(
        pytest.param("source_unit_derivation_v1", case["rubric"], (), id=f"derivation: {case['title']}")
        for case in _COHORT["cases"]
    ),
    *(
        pytest.param("source_unit_support_v1", case["rubric"], (), id=f"support: {case['title']}")
        for case in _COHORT["support_cases"]
    ),
    *(
        pytest.param(
            "source_unit_reconciliation_v1",
            payload["rubric"],
            tuple(str(item["id"]) for item in payload["input"]["incumbents"]),
            id=f"reconciliation: {path.stem}",
        )
        for path in sorted((QUICK / "reconciliation").glob("*.json"))
        for payload in [json.loads(path.read_text(encoding="utf-8"))]
    ),
]


@pytest.mark.parametrize(("case_kind", "rubric", "incumbent_ids"), _RUBRICS)
def test_a_public_rubric_takes_the_forms_its_case_kind_allows(case_kind, rubric, incumbent_ids) -> None:
    validate_expectations(case_kind, rubric, incumbent_ids=incumbent_ids)
