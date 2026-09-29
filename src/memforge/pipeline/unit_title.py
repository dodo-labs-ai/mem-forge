"""The Unit Title as every model reading of a Source Unit shows it.

Claim Extraction, Candidate Admission, Support Assessment and Change Impact read
a Unit's text with the Unit's current Unit Title. It is rendered here, once, from
the values its adapter supplied, and every prompt that shows it embeds
``UNIT_TITLE_DEFINITION``. A change to either changes what each of those
readings shows, so it raises the contract identity of each.
"""

from __future__ import annotations

from memforge.source_projection import UnitTitle

__all__ = ["UNIT_TITLE_DEFINITION", "unit_title_block"]

UNIT_TITLE_DEFINITION = """unit_title names the Source Unit this text belongs to by its kind and values, such as its
key, type, title or path. It is the Unit's current name, not source text: it states no claim and is
never Evidence."""


def unit_title_block(title: UnitTitle | None) -> str:
    """The ``<unit_title>`` block of a prompt; empty for a reading whose projection names no Unit."""
    if title is None:
        return ""
    return f"<unit_title>\n{title.text}\n</unit_title>\n"
