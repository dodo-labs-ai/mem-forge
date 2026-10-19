"""Wait for an admitted offline evaluation run and summarize its report for a pull request.

The report is the service's content-free view: result states and the code
checks of every result. The summary counts each check by label, lists every
case whose result errored or whose check did not pass, and states the run's
verdict, which decides the exit status of ``memforge eval offline-run``.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

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
    lines = [
        "### Offline quality evaluation",
        "",
        f"- Verdict: **{report_verdict(report)}**",
        f"- Run: `{run.get('run_id')}` on cohort `{run.get('cohort_id')}`",
        f"- Code: `{manifest.get('code_revision')}`, model `{manifest.get('model')}`",
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
        lines.append(f"| {criterion} | {counts['pass']} | {counts['fail']} | {counts['unknown']} |")
    if attention:
        lines += ["", "| Case | Check | Outcome |", "| --- | --- | --- |"]
        lines += [f"| `{case}` | {check} | {outcome} |" for case, check, outcome in sorted(attention)]
    return "\n".join(lines) + "\n"


def _finished(report: Mapping[str, Any]) -> bool:
    if (report.get("summary") or {}).get("verdict", "pending") != "pending":
        return True
    return (report.get("execution") or {}).get("state") in _FINISHED_EXECUTION_STATES
