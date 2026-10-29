"""Canonical semantic contract shared by Memory extraction entrypoints."""

from __future__ import annotations

from memforge.pipeline.memory_value import MEMORY_VALUE_DEFINITION

__all__ = [
    "CONTRACT_SUPERSEDED",
    "DURABLE_MEMORY_QUALITY_RULES",
    "MEMORY_CLAIM_EVIDENCE_RULES",
    "PROJECTION_EXTRACTION_CONTRACT_VERSION",
    "PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION",
]


# Recorded on every Source derivation and hashed into its batch identities.
# Stored derivations with any other value are superseded, never resumed.
PROJECTION_EXTRACTION_CONTRACT_VERSION = "projection-extraction-v25"
PROJECTION_FRAGMENT_MODEL_PRESENTATION_POLICY_VERSION = 11
CONTRACT_SUPERSEDED = "CONTRACT_SUPERSEDED"

EVIDENCE_DISPLAY_RULES = """For each selected Primary and Required ref, return one evidence_displays entry
with that ref and a focused faithful restatement of its supplied source view.
Keep the wording and language as close to the source as practical. Preserve every
material condition, negation, exception, scope, order, modality and uncertainty
needed for that ref's contribution. Omit unrelated material and presentation
syntax; do not replace specifics with an overview, explain them, add conclusions,
or import content from another ref. Use only this ref and its declared source
interpretation; separately selected Required content stays in its own display.
Format relationships clearly. No fixed length target overrides source fidelity.
The application retains the complete selected source block for provenance and
revision correspondence; evidence_displays are only readable presentation.
"""


MEMORY_CLAIM_EVIDENCE_RULES = """Extract independently useful claims from the authorized source material.
Preserve material scope, conditions, exceptions, order and modality: proposal,
requirement, expectation, observation or completed change. State uncertainty as
the source states it; do not invent resolutions, attribution or effective dates.
Split by assertion, not by ref: assertions that could change independently, such
as a storage choice and a separate retention period, are separate claims even
when one ref states them together.
Keep together what changes together: a decision with its reason, a rule with
its condition or exception, the steps of one procedure, the items of one list or
schedule, a definition with its scope, and the values of one table row. A
qualification limits when or to what an assertion applies; a further
requirement about the same subject is another assertion, whose Memory restates
the subject it needs to stand alone. Keep known material qualifications in each
claim even when a supplementary citation is omitted.

Choose one eligible Primary directly supporting the central conclusion. It need
not alone prove every supporting detail. Among refs eligible as Primary, a
signpost that only introduces an assertion elsewhere is insufficient: select the
eligible ref that states the assertion itself. Prefer a focused complete source
view when available; a complete broader view is valid.
A new reply that agrees with, answers or decides on earlier content states its
own stance: select the reply as Primary and the earlier content it settles as
Required. Only a pure repeat that adds no stance yields no Memory.
Select Required only when it confidently adds support to a specific part of the
final claim or metadata beyond Primary and source-bound interpretation already
supplied. Do not repeat included labels, duplicate evidence or merely related
background. Empty Required is valid; supplementary recall is best effort. No
ref-count target or exact dependency closure applies.
Effective boundaries must come from source statements that date the claim's
meaning, rather than timestamps attached to updates, events or references.
"""



DURABLE_MEMORY_QUALITY_RULES = MEMORY_VALUE_DEFINITION + """

Cover the independently useful knowledge throughout the authorized source. Do not sample representative claims or replace useful specifics with an overview. Similar wording is not duplication when scope, conditions, branches, outcomes or reasons differ. Exact repetitions and details that establish no lasting knowledge may be omitted.
Distinct defensive conditions, rejections and no-op behavior are useful knowledge even when their verification outcome is unknown. Preserve the expected rule and any material uncertainty; lack of a completed observation is not a reason to omit the rule.

ONE CLAIM, ONE MEMORY. A claim is one assertion that could be revised on its own, not a ref, sentence or paragraph: a ref that states several such assertions yields one Memory for each, all citing that primary_ref. Preserve each independently useful conclusion without reworded duplicates. Fold a rejected alternative into the chosen decision when it explains that same decision.

NO META-MEMORIES. Preserve durable system/domain knowledge rather than commit structure, diff splitting, tool use, validation output or whether a project rule was followed.

THE PRIMARY SETS THE LANGUAGE. Write memory.content in the language of its primary_ref: when that ref is primarily Chinese, write Chinese, even when its Required evidence is English. A primary_ref that mixes languages sets its predominant one. Keep the Primary's own wording, and write any meaning taken from Required evidence or context in the Primary's language. Keep technical identifiers, names, commands and quoted values exactly as the source writes them. Required evidence and structural context may resolve meaning but never change the language.

OPERATIONAL DOES NOT MEAN TRANSIENT. Keep an explicitly stated, repeatable procedure when it remains useful beyond the immediate event. Skip one-off observations, current status, and instance-specific details that establish no lasting knowledge.

"""
