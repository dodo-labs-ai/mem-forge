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
        "litellm_proxy/claude-sonnet": "bedrock/anthropic.claude-sonnet-4-6",
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
    def resolve(cls, model, config, *, catalog_window: bool = True):
        """The capacity of ``model``: LiteLLM's limits for it, lowered by the operator's.

        Without ``catalog_window`` the context window is only the one the
        operator states, for a provider that sizes the window per request: it
        is what the operator's machine holds, which the model's own window
        does not tell.
        """
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

        metadata = {
            "max_input_tokens": info.get("max_input_tokens"),
            "context_window_tokens": info.get("context_window") or info.get("max_input_tokens"),
            "max_output_tokens": info.get("max_output_tokens"),
        }
        if not catalog_window:
            metadata["context_window_tokens"] = None
        known = bool(metadata["max_input_tokens"])
        if not known:
            # Limits the operator states for a model LiteLLM does not know are
            # the expected setup for local and new models; only a limit that
            # neither of them gives is worth a warning.
            unstated = [name for name, value in metadata.items() if value is None and getattr(config, name) is None]
            if unstated:
                logger.warning(
                    "request_budget_metadata_unavailable model=%s unconfigured_limits=%s",
                    model,
                    ",".join(unstated),
                )
            else:
                logger.debug("request_budget_metadata_unavailable model=%s using_explicit_operator_caps=true", model)
        input_limit = limit("max_input_tokens", metadata["max_input_tokens"])
        context_limit = limit("context_window_tokens", metadata["context_window_tokens"])
        output_limit = limit("max_output_tokens", metadata["max_output_tokens"])
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
        payload = {"version": "request-budget-v4", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def output_tokens(self) -> int:
        """The output every request on the route asks for: the route's output capacity.

        It is the model's output limit, but no more than a quarter of the
        context window, so that the input always keeps the rest. A provider
        bills only the output it generates. Some, such as AWS Bedrock and Azure
        OpenAI, count the requested output against the rate quota while a
        request runs; a lower maximum output for the route lowers what each
        request holds there.
        """
        return min(self.output_limit, max(1, self.context_limit // 4))

    def available_input(self, *, reserve_correction: bool = True) -> int:
        """The input a request may hold beside the route's output capacity."""
        return int(min(self.input_limit, self.context_limit - self.output_tokens) * self.fraction) - (
            self.correction_reserve if reserve_correction else 0
        )

    def fits(self, estimated_tokens: int, *, reserve_correction: bool = True) -> bool:
        return estimated_tokens <= self.available_input(reserve_correction=reserve_correction)
