"""Whole-claim validity against the supplied immutable current source reading."""

from __future__ import annotations

__all__ = ["COMPLETE_SUPPORT_DEFINITION", "DIRECTIONAL_SUPPORT_RULE"]

COMPLETE_SUPPORT_DEFINITION = """Complete support: assess the fixed claim against the supplied current source
reading and its authenticated interpretation. Preserve material conditions,
exceptions, scope, modality, attribution and effective boundaries. A material
contradiction or unsupported central assertion means the claim is unsupported.
Selected references locate the prior evidence; optional supplementary-reference
omission alone does not prove the knowledge unsupported. Read governing context
and changes outside selected refs before deciding whole-claim validity.
Use no external facts. The mutable unit_title is display-only and supplies no
claim evidence. A changed display alias alone does not invalidate the claim;
a changed authenticated source fact or referent must be assessed."""


# Whether a later statement keeps an earlier claim true. Support Assessment and
# claim revision state it in these same words.
DIRECTIONAL_SUPPORT_RULE = """An added requirement over the same cases does not revoke an older one."""
