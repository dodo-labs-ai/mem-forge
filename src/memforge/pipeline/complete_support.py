"""What complete Support means, defined once for every model request that judges it.

Support Assessment asks whether current Evidence still supports a fixed claim;
Candidate Admission asks whether a Candidate's selected Evidence supports its
new claim. Both judge the same relation, so both prompts embed this text, and a
change to it raises the contract identity of each.
"""

from __future__ import annotations

__all__ = ["COMPLETE_SUPPORT_DEFINITION"]

COMPLETE_SUPPORT_DEFINITION = """Complete support: the selected Evidence completely supports a claim only when every specific
the claim states appears in that Evidence or follows directly from it. Specifics include names
of people, systems and things, identifiers, quantities, dates and times, statuses, conditions
and scope. A claim that states any specific the Evidence contradicts or does not contain is not
supported, even when the rest of the claim matches. Use no knowledge outside the Evidence."""
