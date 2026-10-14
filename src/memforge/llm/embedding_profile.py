"""Embedding Profile: the identity of one vector space.

Vectors from two embedding models cannot be compared, and neither can vectors
of one model asked for two output sizes. An Embedding Profile names the space
a set of vectors belongs to: the provider, the route and the endpoint they
were requested from, the model, the output size that was requested and the
size that came back. Its fingerprint is a hash of those fields.

A vector index records the fingerprint it was built with, once, and an
Embedding Runtime knows the fingerprint it embeds in. Vectors move between the
two only when the fingerprints are equal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "EmbeddingIdentity",
    "EmbeddingProfile",
]

# The models whose vectors the similarity thresholds were tuned on, by model
# name and vector length.
_CALIBRATED_MODELS: frozenset[tuple[str, int]] = frozenset({("text-embedding-3-small", 1536)})


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The settings that decide which vector space an embedding model answers in."""

    provider: str
    # The LiteLLM route requests are made with.
    route: str
    # The normalized base URL; ``None`` leaves the endpoint to the route.
    endpoint: str | None
    model_id: str
    # The output size the requests ask for; ``None`` takes the model's own.
    requested_dimensions: int | None = None

    def profile(self, dimension: int) -> EmbeddingProfile:
        """The profile of this model once it has answered with vectors of ``dimension``."""
        return EmbeddingProfile(identity=self, dimension=dimension)


@dataclass(frozen=True)
class EmbeddingProfile:
    """One vector space: the settings of an embedding model and the vector length it returned."""

    identity: EmbeddingIdentity
    # The vector length observed in the model's answers.
    dimension: int

    @property
    def fingerprint(self) -> str:
        """Identify the vector space. Two profiles share vectors only when this is equal."""
        encoded = json.dumps(self.fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def thresholds_calibrated(self) -> bool:
        """Whether the similarity thresholds were tuned on this model's vectors.

        The thresholds for duplicates and near matches are distances measured
        with one model. With any other model they still apply, as approximate
        values that nobody has measured.
        """
        model = self.identity.route.rpartition("/")[2] or self.identity.model_id
        return (model, self.dimension) in _CALIBRATED_MODELS

    def fields(self) -> dict[str, Any]:
        """The fields the fingerprint is a hash of."""
        return {**asdict(self.identity), "dimension": self.dimension}

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> EmbeddingProfile:
        """Read a profile back from ``fields()``."""
        return cls(
            identity=EmbeddingIdentity(
                provider=fields["provider"],
                route=fields["route"],
                endpoint=fields.get("endpoint"),
                model_id=fields["model_id"],
                requested_dimensions=fields.get("requested_dimensions"),
            ),
            dimension=int(fields["dimension"]),
        )
