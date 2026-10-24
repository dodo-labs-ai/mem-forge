"""Render a Notion row's property values as text, one record per property.

A database row is a page whose properties the user defines per database, so
they are kept as a list of records with a fixed shape, never as fields named
after the user's columns::

    render_properties(page["properties"], file_text=str)
    # [{"name": "Owner", "type": "people", "value": "Ada Lovelace"},
    #  {"name": "Status", "type": "status", "value": "Done"}]

Every value is a string, ``""`` when the property is empty. Records are
sorted by name, so the same row always renders the same. A property type
this module does not know, such as ``button`` or one Notion adds later, keeps
its name and type with the value ``""``.

``file_text`` renders one entry of a ``files`` property; the Gene passes a
function that turns a Notion-hosted file into its stable reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from memforge.genes.notion_api import notion_id

__all__ = ["render_properties", "render_value"]

FileText = Callable[[Mapping[str, Any]], str]

_LIST_SEPARATOR = ", "


def render_properties(properties: Mapping[str, Any], *, file_text: FileText) -> list[dict[str, str]]:
    """Each property of a row as ``{"name", "type", "value"}``, sorted by name; see the module docstring."""
    records = []
    for name, value in properties.items():
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Notion row property {name!r} is not an object")
        records.append({"name": str(name), "type": str(value.get("type") or ""), "value": render_value(value, file_text=file_text)})
    return sorted(records, key=lambda record: record["name"])


def render_value(value: Mapping[str, Any], *, file_text: FileText) -> str:
    """One property value, or one element of a rollup array, as text by its ``type``::

        render_value({"type": "number", "number": 0}, file_text=str)                  # "0"
        render_value({"type": "multi_select", "multi_select": [{"name": "A"}, {"name": "B"}]},
                     file_text=str)                                                   # "A, B"
        render_value({"type": "unique_id", "unique_id": {"prefix": "BUG", "number": 7}},
                     file_text=str)                                                   # "BUG-7"
    """
    value_type = str(value.get("type") or "")
    content = value.get(value_type)
    if value_type in {"title", "rich_text"}:
        return _rich_text(content)
    if value_type == "number":
        return _number(content)
    if value_type == "checkbox":
        return "true" if content is True else "false"
    if value_type in {"url", "email", "phone_number", "created_time", "last_edited_time"}:
        return str(content or "")
    if value_type in {"select", "status"}:
        return _name(content)
    if value_type == "multi_select":
        return _joined(_name(option) for option in _list(content))
    if value_type == "date":
        return _date(content)
    if value_type == "people":
        return _joined(_user(user) for user in _list(content))
    if value_type in {"created_by", "last_edited_by"}:
        return _user(content)
    if value_type == "relation":
        return _joined(notion_id(related.get("id")) or "" for related in _list(content) if isinstance(related, Mapping))
    if value_type == "files":
        return _joined(file_text(entry) for entry in _list(content) if isinstance(entry, Mapping))
    if value_type in {"formula", "rollup"} and isinstance(content, Mapping):
        if content.get("type") == "array":
            return _joined(
                render_value(element, file_text=file_text)
                for element in _list(content.get("array"))
                if isinstance(element, Mapping)
            )
        return _scalar(content)
    if value_type == "unique_id" and isinstance(content, Mapping):
        number = _number(content.get("number"))
        return f"{content['prefix']}-{number}" if content.get("prefix") and number else number
    if value_type == "verification" and isinstance(content, Mapping):
        return str(content.get("state") or "")
    return ""


def _scalar(content: Mapping[str, Any]) -> str:
    """A formula result or a rollup that is a string, number, boolean or date."""
    result_type = str(content.get("type") or "")
    result = content.get(result_type)
    if result_type == "string":
        return str(result or "")
    if result_type == "number":
        return _number(result)
    if result_type == "boolean":
        return "" if result is None else ("true" if result is True else "false")
    if result_type == "date":
        return _date(result)
    return ""


def _rich_text(content: object) -> str:
    """Rich text segments joined by their ``plain_text``, which also spells out mentions and equations."""
    return "".join(str(segment.get("plain_text") or "") for segment in _list(content) if isinstance(segment, Mapping))


def _number(value: object) -> str:
    """A number as plain decimal text: ``0`` is ``"0"`` and ``1e21`` is ``"1000000000000000000000"``."""
    if value is None or isinstance(value, bool):
        return ""
    try:
        return format(Decimal(str(value)), "f")
    except InvalidOperation:
        return str(value)


def _date(content: object) -> str:
    """``start``, or ``start/end`` for a range, followed by the time zone when Notion names one."""
    if not isinstance(content, Mapping) or not content.get("start"):
        return ""
    text = str(content["start"])
    if content.get("end"):
        text += f"/{content['end']}"
    if content.get("time_zone"):
        text += f" {content['time_zone']}"
    return text


def _user(user: object) -> str:
    """A user's name, or the user's id when Notion withholds the name (a deleted user has none)."""
    if not isinstance(user, Mapping):
        return ""
    return str(user.get("name") or "").strip() or notion_id(user.get("id")) or ""


def _name(option: object) -> str:
    return str(option.get("name") or "") if isinstance(option, Mapping) else ""


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _joined(parts: Any) -> str:
    return _LIST_SEPARATOR.join(part for part in parts if part)
