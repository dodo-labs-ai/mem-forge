"""Calibrate the expectation judge against expectations a person labeled.

A calibration file holds one case: its kind, the document it is about, its
expectations and one or more outputs exactly as the judge reads them, each with
a person's label for every expectation, ``met`` or ``not_met``. An output can be
one a model gave in a quality run or one edited to be wrong, so both labels
occur. Each file belongs to the ``dev`` split, which a maintainer reads while
changing how the judge reads a case kind, or the ``test`` split, which measures
that change once it is made. Every case about one document sits in the same
split, so the test split never holds an output the change was fitted to, even
under another case kind.

The report gives, per split, case kind and expectation form, the judge's true
positive rate (labeled met, judged met) and true negative rate (labeled
not_met, judged not_met), and lists every expectation where the judge and the
label differ. A judge decision of unclear, or none, differs from either label.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from memforge.evals.expectation_judge import (
    CaseOutputView,
    ExpectationJudge,
    ExpectationJudgeSpec,
    expectation_form,
    validate_expectations,
)

CALIBRATION_SPLITS = ("dev", "test")
_LABELS = ("met", "not_met")
# How a judge decision is named against a person's label.
_DECISIONS = {"pass": "met", "fail": "not_met", "needs_review": "unclear", None: "undecided"}


@dataclass(frozen=True, slots=True)
class CalibrationOutput:
    """One output of a calibration case as the judge reads it, with a person's label for each expectation."""

    note: str
    items: tuple[Mapping[str, object], ...]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    name: str
    case_kind: str
    document: str
    split: str
    expectations: tuple[str, ...]
    outputs: tuple[CalibrationOutput, ...]


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    case: str
    split: str
    case_kind: str
    form: str
    note: str
    index: int
    label: str
    decision: str


def load_calibration(directory: Path) -> tuple[CalibrationCase, ...]:
    """Every calibration case under ``directory``, by file name; a malformed file raises ``ValueError`` naming it."""

    cases = []
    for path in sorted(directory.glob("*.json")):
        try:
            cases.append(_case(path.stem, json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    if not cases:
        raise ValueError(f"{directory} holds no calibration case")
    splits: dict[str, set[str]] = {}
    for case in cases:
        splits.setdefault(case.document, set()).add(case.split)
    if crossing := sorted(document for document, named in splits.items() if len(named) > 1):
        raise ValueError(f"cases about {', '.join(crossing)} sit in both splits; keep each document in one")
    return tuple(cases)


def _case(name: str, payload: Mapping[str, object]) -> CalibrationCase:
    case_kind = str(payload["case_kind"])
    document = str(payload["document"]).strip()
    if not document:
        raise ValueError("document must name what the case is about")
    split = str(payload["split"])
    if split not in CALIBRATION_SPLITS:
        raise ValueError(f"split must be one of {', '.join(CALIBRATION_SPLITS)}")
    expectations = payload["expectations"]
    validate_expectations(case_kind, {"expectations": expectations}, incumbent_ids={
        str(item["id"]) for output in payload["outputs"] for item in output["items"] if "old_memory" in item
    })
    outputs = []
    for number, output in enumerate(payload["outputs"], start=1):
        items = tuple(dict(item) for item in output["items"])
        ids = [str(item["id"]) for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"output {number} repeats an item id")
        labels = tuple(str(label) for label in output["labels"])
        if len(labels) != len(expectations) or any(label not in _LABELS for label in labels):
            raise ValueError(f"output {number} needs one label, met or not_met, per expectation")
        note = str(output.get("note") or f"output {number}")
        if "|" in note:
            raise ValueError(f"output {number}'s note holds |, which the report's table cannot show")
        outputs.append(CalibrationOutput(note=note, items=items, labels=labels))
    if not outputs:
        raise ValueError("a calibration case needs at least one output")
    return CalibrationCase(name, case_kind, document, split, tuple(str(text) for text in expectations), tuple(outputs))


async def calibrate(
    judge: ExpectationJudge, cases: Sequence[CalibrationCase], *, splits: Sequence[str] = CALIBRATION_SPLITS,
) -> tuple[CalibrationDecision, ...]:
    """Judge every output of the cases in ``splits`` and pair each decision with its label."""

    jobs = [(case, output) for case in cases if case.split in splits for output in case.outputs]
    verdicts = await asyncio.gather(*(
        judge.judge(
            case.expectations,
            CaseOutputView(case.case_kind, output.items, {str(item["id"]): str(item["id"]) for item in output.items}),
        )
        for case, output in jobs
    ))
    decisions = []
    for (case, output), judged in zip(jobs, verdicts):
        for index, (expectation, label, verdict) in enumerate(
            zip(case.expectations, output.labels, judged, strict=True), start=1,
        ):
            # A loaded case's expectations each take a form.
            form, _ = expectation_form(case.case_kind, expectation)
            decisions.append(CalibrationDecision(
                case=case.name, split=case.split, case_kind=case.case_kind,
                form=form.example, note=output.note, index=index,
                label=label, decision=_DECISIONS[verdict.label],
            ))
    return tuple(decisions)


def render_calibration_markdown(spec: ExpectationJudgeSpec, decisions: Sequence[CalibrationDecision]) -> str:
    """The judge's agreement with the labels, per split, case kind and form, then every disagreement."""

    lines = [
        "### Expectation judge calibration",
        "",
        f"- Judge: `{spec.model}` (`{spec.provider}`), prompt `{spec.prompt_hash[:12]}`",
        "",
        "| Split | Case kind | Form | Met | TPR | Not met | TNR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups: dict[tuple[str, str, str], Counter[str]] = {}
    for decision in decisions:
        counts = groups.setdefault((decision.split, decision.case_kind, decision.form), Counter())
        counts[decision.label] += 1
        counts[f"{decision.label}_agreed"] += decision.decision == decision.label
    for split in CALIBRATION_SPLITS:
        keys = [key for key in sorted(groups) if key[0] == split]
        if not keys:
            continue
        total = sum((groups[key] for key in keys), Counter())
        for (row_split, kind, form), counts in [((split, "all", "all"), total), *((key, groups[key]) for key in keys)]:
            lines.append(
                f"| {row_split} | {kind} | {form} | {counts['met']} | {_rate(counts, 'met')} "
                f"| {counts['not_met']} | {_rate(counts, 'not_met')} |"
            )
    differing = [decision for decision in decisions if decision.decision != decision.label]
    if not differing:
        lines += ["", "The judge agreed with every label."]
        return "\n".join(lines) + "\n"
    lines += ["", "| Split | Case | Output | Expectation | Label | Judge |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [
        f"| {item.split} | {item.case} | {item.note} | E{item.index} | {item.label} | {item.decision} |"
        for item in sorted(differing, key=lambda item: (item.split, item.case, item.note, item.index))
    ]
    return "\n".join(lines) + "\n"


def _rate(counts: Counter[str], label: str) -> str:
    return f"{counts[f'{label}_agreed'] / counts[label]:.0%}" if counts[label] else "–"
