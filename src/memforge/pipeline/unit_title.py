"""The Unit Title as every model reading of a Source Unit shows it.

Claim Extraction, Candidate Admission, Support Assessment and Change Impact read
a Unit's text with the Unit's current Unit Title. Its rendering and its definition
live here, once, and every one of those prompts shows them through
``unit_title_block``. A change to either changes what each of those readings
shows, so it raises the contract identity of each.
"""

from __future__ import annotations

from memforge.source_projection import UnitTitle

__all__ = ["UNIT_TITLE_DEFINITION", "render_unit_title", "unit_title_block"]

UNIT_TITLE_DEFINITION = """unit_title is a mutable display label used to navigate the Source Unit.
It is not source Evidence and cannot substantiate claim facts. Use authenticated
revision-pinned source framing for names, identifiers, attribution and time."""


def render_unit_title(title: UnitTitle) -> str:
    """The kind, then one ``name: value`` line per value."""
    return "\n".join((title.kind, *(f"{name}: {value}" for name, value in title.fields)))


def unit_title_block(title: UnitTitle | None, *, guidance: str = "") -> str:
    """The ``<unit_title>`` block, its definition and a reading's own ``guidance`` on it.

    Empty for a reading whose projection names no Unit, so no prompt describes a
    title it does not show.
    """
    if title is None:
        return ""
    lines = (f"<unit_title>\n{render_unit_title(title)}\n</unit_title>", UNIT_TITLE_DEFINITION, guidance)
    return "\n".join(line for line in lines if line) + "\n"
