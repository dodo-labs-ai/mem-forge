"""Wait for an admitted offline evaluation run and summarize its report for a pull request.

The report is the service's content-free view: result states, the code
checks of every result and the expectation judge's decisions. A run that pins
a judge is summarized by its quality: the count of the judge's decisions, and
each expectation not met, unclear or not judged, with the IDs of the output
items it rests on. These are advisory. Its code checks ask only whether the
replay gave a valid result, so the summary shows them, with the verdict that
decides the exit status of ``memforge eval offline-run``, only when one did not
pass. A run without a judge is summarized by its verdict and code checks.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from memforge.evals.expectation_judge import EXPECTATION_CRITERION

# The execution states after which a run's report no longer changes.
_FINISHED_EXECUTION_STATES = frozenset({"completed", "failed"})


def wait_for_report(
    read_report: Callable[[], Mapping[str, Any]],
    *,
    timeout_s: float,
    poll_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Mapping[str, Any] | None:
    """Read the report until its run has finished; ``None`` when ``timeout_s`` passes first.

    A read that answers with an error is returned at once.
    """

    deadline = clock() + timeout_s
    while True:
        report = read_report()
        if report.get("error") or _finished(report):
            return report
        if clock() + poll_s > deadline:
            return None
        sleep(poll_s)


def report_verdict(report: Mapping[str, Any]) -> str:
    """The verdict of a finished run; a run whose execution failed without every result is ``fail``."""

    verdict = str((report.get("summary") or {}).get("verdict") or "pending")
    execution = report.get("execution") or {}
    if verdict == "pending" and execution.get("state") == "failed":
        return "fail"
    return verdict


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """A Markdown summary of one finished run, without case content."""

    run = report.get("run") or {}
    summary = report.get("summary") or {}
    results = report.get("results") or []
    assessments = [item for item in report.get("assessments") or [] if item.get("annotator_kind") == "code"]
    case_by_result = {str(result.get("result_id")): str(result.get("case_id")) for result in results}
    labels: dict[str, Counter[str]] = {}
    attention: list[tuple[str, str, str]] = []
    for assessment in assessments:
        criterion = str(assessment.get("criterion"))
        # A check that waits for a person's review has not decided either way.
        label = str(assessment.get("label") or "unknown")
        label = "unknown" if label == "needs_review" else label
        labels.setdefault(criterion, Counter())[label] += 1
        if label != "pass":
            attention.append(
                (
                    case_by_result.get(str(assessment.get("target_result_id")), "?"),
                    criterion,
                    f"{label}: {assessment.get('reason_code') or ''}",
                )
            )
    for result in results:
        if result.get("status") != "completed":
            attention.append(
                (str(result.get("case_id")), "result", f"{result.get('status')}: {result.get('error_code') or ''}")
            )
    manifest = run.get("candidate_manifest") or {}
    judge = run.get("expectation_judge_manifest")
    verdict = report_verdict(report)
    lines = [
        "### Offline quality evaluation",
        "",
        f"- Run: `{run.get('run_id')}` on cohort `{run.get('cohort_id')}`",
        f"- Code: `{manifest.get('code_revision')}`, model `{manifest.get('model')}`",
    ]
    validity = [
        f"- Verdict: **{verdict}**",
        (
            f"- Results: {summary.get('completed_result_count', 0)} completed, "
            f"{summary.get('error_result_count', 0)} not completed"
        ),
        "",
        "| Check | Pass | Fail | Unknown |",
        "| --- | --- | --- | --- |",
    ]
    for criterion in sorted(labels):
        counts = labels[criterion]
        validity.append(f"| {criterion} | {counts['pass']} | {counts['fail']} | {counts['unknown']} |")
    if attention:
        validity += ["", "| Case | Check | Outcome |", "| --- | --- | --- |"]
        validity += [f"| `{case}` | {check} | {outcome} |" for case, check, outcome in sorted(attention)]
    if not judge:
        return "\n".join(lines + validity) + "\n"
    lines.append(f"- Judge: `{judge.get('model')}`")
    lines += _expectation_lines(report.get("assessments") or [], case_by_result)
    if verdict != "pass" or attention:
        lines += ["", "#### Replay problems", "", *validity]
    return "\n".join(lines) + "\n"


# How a report names the judge's decision on one expectation.
_EXPECTATION_OUTCOMES = {"pass": "met", "fail": "not met", "needs_review": "unclear"}


def _expectation_lines(
    assessments: list[Mapping[str, Any]], case_by_result: Mapping[str, str],
) -> list[str]:
    """The count of the judge's decisions and each expectation it did not find met."""

    counts: Counter[str] = Counter()
    listed: list[tuple[str, int, str, str]] = []
    for assessment in assessments:
        if assessment.get("criterion") != EXPECTATION_CRITERION:
            continue
        code, _, items = str(assessment.get("reason_code") or "").partition("/")
        expectation, _, detail = code.partition(":")
        outcome = (
            _EXPECTATION_OUTCOMES.get(str(assessment.get("label")), "unclear")
            if assessment.get("status") == "completed"
            else "not judged"
        )
        counts[outcome] += 1
        if outcome != "met":
            listed.append((
                case_by_result.get(str(assessment.get("target_result_id")), "?"),
                int(expectation[1:]) if expectation[1:].isdigit() else 0,
                outcome if outcome != "not judged" else f"not judged: {detail}",
                ", ".join(f"`{item}`" for item in items.split("/") if item),
            ))
    lines = [
        "",
        "| Met | Not met | Unclear | Not judged |",
        "| --- | --- | --- | --- |",
        f"| {counts['met']} | {counts['not met']} | {counts['unclear']} | {counts['not judged']} |",
    ]
    if listed:
        lines += ["", "| Case | Expectation | Outcome | Items |", "| --- | --- | --- | --- |"]
        lines += [
            f"| `{case}` | E{index} | {outcome} | {items} |" for case, index, outcome, items in sorted(listed)
        ]
    return lines


def _finished(report: Mapping[str, Any]) -> bool:
    """Whether the run's execution has ended, so its report no longer changes.

    The verdict is decided once every result is in, before the judges' decisions
    on the last result are recorded, so the execution's state decides; a run
    without one has ended once its verdict is decided.
    """

    execution = report.get("execution")
    if execution:
        return execution.get("state") in _FINISHED_EXECUTION_STATES
    return (report.get("summary") or {}).get("verdict", "pending") != "pending"
