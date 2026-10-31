"""The expectation judge is calibrated against labeled outputs, apart from any run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memforge.evals.expectation_judge import ExpectationJudgeSpec, ExpectationVerdict
from memforge.evals import offline_runtime
from memforge.evals.judge_calibration import (
    CALIBRATION_SPLITS,
    calibrate,
    load_calibration,
    render_calibration_markdown,
)
from memforge.main import cli

CALIBRATION = Path(__file__).resolve().parents[1] / "evals" / "quality" / "judge-calibration"
SPEC = ExpectationJudgeSpec(provider="openai_compatible", model="openai/gpt-6-luna")
ACK = "If a memory states that a page must be acknowledged within 5 minutes, it is judged unsupported."
NONE = "No memory is judged unsupported."


def _case(split: str = "test", **changes) -> dict:
    return {
        "case_kind": "source_unit_support_v1",
        "document": "on-call",
        "split": split,
        "expectations": [ACK, NONE],
        "outputs": [
            {
                "note": "as replayed",
                "items": [{"id": "M1", "memory": "Acknowledge a page within 5 minutes.", "verdict": "unsupported"}],
                "labels": ["met", "not_met"],
            },
            {
                "note": "verdict swapped",
                "items": [{"id": "M1", "memory": "Acknowledge a page within 5 minutes.", "verdict": "supported"}],
                "labels": ["not_met", "met"],
            },
        ],
        **changes,
    }


def _write(directory: Path, name: str, payload: dict) -> None:
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


class ByVerdictJudge:
    """Answers from the single item's verdict: the first expectation met when unsupported, the second never."""

    async def judge(self, expectations, view):
        unsupported = view.items[0]["verdict"] == "unsupported"
        return (
            ExpectationVerdict("pass" if unsupported else "fail", ("M1",)),
            ExpectationVerdict("needs_review", ()),
        )


def test_a_calibration_case_is_refused_unless_every_expectation_is_labeled_in_an_allowed_form(tmp_path) -> None:
    _write(tmp_path, "ack", _case())
    [case] = load_calibration(tmp_path)
    assert (case.name, case.split, len(case.outputs)) == ("ack", "test", 2)

    for name, payload, message in [
        ("split", _case(split="train"), "split must be one of dev, test"),
        ("labels", _case(outputs=[{"items": [{"id": "M1"}], "labels": ["met"]}]), "one label, met or not_met"),
        ("unclear", _case(outputs=[{"items": [{"id": "M1"}], "labels": ["met", "unclear"]}]), "met or not_met"),
        ("form", _case(expectations=["Treat paging as supported.", NONE]), "takes none of the forms"),
        ("kind", _case(case_kind="memory_relation_v1"), "has no expectations; it is one of"),
        ("note", _case(outputs=[{"note": "a | b", "items": [{"id": "M1"}], "labels": ["met", "met"]}]), "holds \\|"),
    ]:
        bad = tmp_path / name
        bad.mkdir()
        _write(bad, name, payload)
        with pytest.raises(ValueError, match=rf"{name}\.json: .*{message}"):
            load_calibration(bad)


def test_every_case_about_one_document_sits_in_one_split(tmp_path) -> None:
    _write(tmp_path, "support-on-call", _case(split="test"))
    _write(tmp_path, "derivation-on-call", _case(split="dev"))

    with pytest.raises(ValueError, match="cases about on-call sit in both splits"):
        load_calibration(tmp_path)


async def test_the_report_rates_agreement_per_split_and_form_and_lists_each_disagreement(tmp_path) -> None:
    _write(tmp_path, "ack-test", _case())
    _write(tmp_path, "ack-dev", _case(split="dev", document="escalation"))
    cases = load_calibration(tmp_path)

    decisions = await calibrate(ByVerdictJudge(), cases, splits=("test",))
    summary = render_calibration_markdown(SPEC, decisions)

    assert {decision.case for decision in decisions} == {"ack-test"}
    assert "| test | all | all | 2 | 50% | 2 | 50% |" in summary
    assert "| test | source_unit_support_v1 | If a memory states that …, it is judged unsupported. | 1 | 100% | 1 | 100% |" in summary
    assert "| test | source_unit_support_v1 | No memory is judged unsupported. | 1 | 0% | 1 | 0% |" in summary
    assert "| test | ack-test | as replayed | E2 | not_met | unclear |" in summary
    assert "| dev |" not in summary


def test_the_public_calibration_set_holds_both_splits_and_both_labels() -> None:
    cases = load_calibration(CALIBRATION)

    assert {case.split for case in cases} == set(CALIBRATION_SPLITS)
    assert {case.case_kind for case in cases} == {
        "source_unit_derivation_v1", "source_unit_support_v1", "source_unit_reconciliation_v1",
    }
    for split in CALIBRATION_SPLITS:
        labels = {label for case in cases if case.split == split for output in case.outputs for label in output.labels}
        assert labels == {"met", "not_met"}, split


def test_the_command_judges_the_chosen_split_with_the_configured_judge(tmp_path, monkeypatch) -> None:
    _write(tmp_path, "ack-test", _case())
    _write(tmp_path, "ack-dev", _case(split="dev", document="escalation"))
    for name in ("PROVIDER", "MODEL", "ENDPOINT", "API_KEY"):
        monkeypatch.delenv(f"MEMFORGE_AGENT_EVALUATION_JUDGE_{name}", raising=False)

    unconfigured = CliRunner().invoke(cli, ["eval", "judge-calibrate", "--dir", str(tmp_path)])

    assert unconfigured.exit_code == 2
    assert "No expectation judge is configured" in unconfigured.output

    async def built(config, runtime_provider):
        return SPEC, ByVerdictJudge()

    monkeypatch.setattr(offline_runtime, "build_expectation_judge", built)
    judged = CliRunner().invoke(cli, ["eval", "judge-calibrate", "--dir", str(tmp_path), "--split", "dev"])

    assert judged.exit_code == 0, judged.output
    assert "| dev | all | all | 2 | 50% | 2 | 50% |" in judged.output
    assert "| test |" not in judged.output
