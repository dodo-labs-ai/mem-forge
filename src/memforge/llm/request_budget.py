"""Immutable request capacity resolved through LiteLLM and operator caps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging

import litellm

logger = logging.getLogger(__name__)


def metadata_model(model: str) -> str:
    """Route aliases identify models; all capacity numbers remain SDK-owned."""
    return {
        "sap/anthropic--claude-4.6-sonnet": "bedrock/anthropic.claude-sonnet-4-6",
    }.get(model, model)



@dataclass(frozen=True)
class RequestBudget:
    model: str
    input_limit: int
    context_limit: int
    output_limit: int
    fraction: float
    metadata_source: str
    correction_reserve: int = 1024

    @classmethod
    def resolve(cls, model, config):
        try:
            info = litellm.get_model_info(metadata_model(model))
        except Exception:
            info = {}

        def limit(name, metadata):
            configured = getattr(config, name)
            values = [int(value) for value in (configured, metadata) if value is not None]
            if not values:
                raise ValueError(
                    f"Model {model} has no {name} metadata; configure MEMFORGE_LLM_{name.upper()} for this route"
                )
            return min(values)

        known = bool(info.get("max_input_tokens"))
        if not known:
            logger.warning("request_budget_metadata_unavailable model=%s using_explicit_operator_caps=true", model)
        input_limit = limit("max_input_tokens", info.get("max_input_tokens"))
        context_limit = limit("context_window_tokens", info.get("context_window") or info.get("max_input_tokens"))
        output_limit = limit("max_output_tokens", info.get("max_output_tokens"))
        if min(input_limit, context_limit, output_limit) < 1 or not 0 < config.input_budget_fraction <= 1:
            raise ValueError("invalid structured request capacity")
        return cls(
            model,
            input_limit,
            context_limit,
            output_limit,
            config.input_budget_fraction,
            "litellm_and_operator_caps" if known else "operator_caps",
        )

    @property
    def identity(self):
        payload = {"version": "request-budget-v3", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def output_reserve(self, requested: int) -> int:
        """Bound variable output while leaving space for the request itself."""
        return min(requested, self.output_limit, max(1, self.context_limit // 4))

    def available_input(self, output_tokens: int, *, reserve_correction: bool = True) -> int:
        if not 0 < output_tokens <= self.output_limit:
            return -1
        return int(min(self.input_limit, self.context_limit - output_tokens) * self.fraction) - (
            self.correction_reserve if reserve_correction else 0
        )

    def fits(self, estimated_tokens: int, output_tokens: int, *, reserve_correction: bool = True) -> bool:
        return estimated_tokens <= self.available_input(output_tokens, reserve_correction=reserve_correction)
