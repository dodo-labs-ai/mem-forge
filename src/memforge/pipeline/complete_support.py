"""What complete Support means, defined once for every model request that judges it.

Support Assessment asks whether current Evidence still supports a fixed claim;
Candidate Admission asks whether a Candidate's selected Evidence supports its
new claim. Both judge the same relation, with the Unit's current Unit Title as
context, so both prompts embed this text, and a change to it raises the contract
identity of each.
"""

from __future__ import annotations

__all__ = ["COMPLETE_SUPPORT_DEFINITION"]

COMPLETE_SUPPORT_DEFINITION = """Complete support: the selected Evidence completely supports a claim only when every specific
the claim states appears in that Evidence or in the unit_title, or follows directly from them.
Specifics include names of people, systems and things, identifiers, quantities, dates and times,
statuses, conditions and scope. A claim that states any specific the Evidence contradicts, or one
that neither the Evidence nor the unit_title contains, is not supported, even when the rest of the
claim matches. Use no knowledge outside the Evidence and the unit_title. The unit_title is never
Evidence, but every value it shows, such as the Unit's key, type, summary, title or path, is a fact
about the Unit that a claim may state without Evidence for it. A claim that names its Unit by an
earlier name, such as a former title or path, still speaks of that Unit; that the name differs from
the unit_title is never alone a reason for unsupported. Apart from the Unit's own earlier names, an
identifier that neither the unit_title nor the Evidence contains, such as another Unit's key, is not
supported."""
