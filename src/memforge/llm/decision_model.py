"""Which model answers a Decision task.

A Decision task answers one fixed question about one item the program supplied
in full, with closed options (ADR 0043). It moves to the decision model only
after its contract version passes its evaluation: on held-out labeled cases,
against the main model, its safe-answer recall and its precision on every other
option are no lower. Passed tasks are registered here with that contract
version. A new contract version is not registered until it passes on its own.

The decision model is one setting (``MEMFORGE_DECISION_MODEL``) applied to every
registered task. When it is not set, every task runs on the main model. There is
no per-task setting, no per-item routing and no fallback between models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class DecisionTask:
    """One Decision task at the contract version its work identity carries."""

    name: str
    contract_version: str


# Task name -> the contract version that passed its decision evaluation.
EVALUATED_DECISION_TASKS: Mapping[str, str] = MappingProxyType({})


def decision_task_model(client: Any, task: DecisionTask, main_model: str | None) -> str | None:
    """The model that answers ``task`` on ``client``.

    ``main_model`` is what the caller passes for every other task; ``None``
    means the client's own model. A client without a decision model answers
    every task on the main model.
    """

    decision_model = getattr(client, "decision_model", None)
    if decision_model and EVALUATED_DECISION_TASKS.get(task.name) == task.contract_version:
        return decision_model
    return main_model
