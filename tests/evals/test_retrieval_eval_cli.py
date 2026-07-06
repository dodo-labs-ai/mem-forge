from __future__ import annotations

import json

from click.testing import CliRunner

from memforge.main import cli


def test_retrieval_eval_cli_runs_packaged_case_set_as_json() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "retrieval",
            "--case-set",
            "retrieval-core-v1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == {
        "case_count": 4,
        "hard_failures": 0,
    }
    assert payload["hard_failures"] == []
    assert payload["qrels"]["exact_external_id_lookup"] == {"mem-blocker-hint": 3}


def test_retrieval_eval_cli_text_output_is_quiet_summary() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "retrieval",
            "--case-set",
            "retrieval-core-v1",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "retrieval eval retrieval-core-v1: 4 cases, 0 hard failures"


def test_retrieval_eval_cli_can_fail_on_hard_failures(monkeypatch) -> None:
    from memforge.evals.retrieval.runner import HardFailure, RetrievalEvalReport

    async def fake_run_sqlite_case_set(*args, **kwargs) -> RetrievalEvalReport:
        return RetrievalEvalReport(
            case_results={},
            hard_failures=(HardFailure(case_id="broken", message="expected rank <= 1, got 3"),),
            qrels={},
            run={},
        )

    monkeypatch.setattr(
        "memforge.evals.retrieval.runner.run_sqlite_case_set",
        fake_run_sqlite_case_set,
    )

    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "retrieval",
            "--format",
            "json",
            "--fail-on-hard-failure",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["hard_failures"] == 1
    assert payload["hard_failures"][0]["case_id"] == "broken"
