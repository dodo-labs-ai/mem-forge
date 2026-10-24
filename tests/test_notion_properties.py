"""A Notion row's property values render as text by type."""

from __future__ import annotations

from typing import Any

import pytest

from memforge.genes.notion_properties import render_properties, render_value

ADA = {"object": "user", "id": "11111111-2222-4333-8444-555555555555", "name": "Ada"}
DELETED = {"object": "user", "id": "66666666-7777-4888-8999-000000000000", "name": None}
WITHHELD = {"object": "user", "id": "66666666-7777-4888-8999-000000000000"}


def file_text(entry: Any) -> str:
    return f"file:{entry['name']}"


def rich(*parts: str) -> list[dict]:
    return [{"type": "text", "plain_text": part} for part in parts]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        # text
        ({"type": "title", "title": rich("Launch ", "plan")}, "Launch plan"),
        (
            {"type": "rich_text", "rich_text": [*rich("See "), {"type": "mention", "plain_text": "@Ada"}, {"type": "equation", "plain_text": "E=mc^2"}]},
            "See @AdaE=mc^2",
        ),
        ({"type": "rich_text", "rich_text": []}, ""),
        ({"type": "url", "url": "https://example.com"}, "https://example.com"),
        ({"type": "email", "email": "ada@example.com"}, "ada@example.com"),
        ({"type": "phone_number", "phone_number": "+1 555"}, "+1 555"),
        ({"type": "url", "url": None}, ""),
        # number and checkbox
        ({"type": "number", "number": 0}, "0"),
        ({"type": "number", "number": 3.25}, "3.25"),
        ({"type": "number", "number": 1e21}, "1000000000000000000000"),
        ({"type": "number", "number": None}, ""),
        ({"type": "checkbox", "checkbox": False}, "false"),
        ({"type": "checkbox", "checkbox": True}, "true"),
        # select
        ({"type": "select", "select": {"name": "High"}}, "High"),
        ({"type": "select", "select": None}, ""),
        ({"type": "status", "status": {"name": "Done"}}, "Done"),
        ({"type": "multi_select", "multi_select": [{"name": "A"}, {"name": "B"}]}, "A, B"),
        # dates
        ({"type": "date", "date": {"start": "2026-03-01"}}, "2026-03-01"),
        ({"type": "date", "date": {"start": "2026-03-01", "end": "2026-03-05"}}, "2026-03-01/2026-03-05"),
        (
            {"type": "date", "date": {"start": "2026-03-01T09:30:00.000+01:00", "end": None, "time_zone": "Europe/Berlin"}},
            "2026-03-01T09:30:00.000+01:00 Europe/Berlin",
        ),
        ({"type": "date", "date": None}, ""),
        ({"type": "created_time", "created_time": "2026-03-01T09:00:00.000Z"}, "2026-03-01T09:00:00.000Z"),
        ({"type": "last_edited_time", "last_edited_time": "2026-03-02T10:00:00.000Z"}, "2026-03-02T10:00:00.000Z"),
        # people
        ({"type": "people", "people": [ADA, DELETED]}, "Ada, 66666666777748888999000000000000"),
        ({"type": "created_by", "created_by": ADA}, "Ada"),
        ({"type": "last_edited_by", "last_edited_by": WITHHELD}, "66666666777748888999000000000000"),
        # relation
        (
            {"type": "relation", "relation": [{"id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"}, {"id": "11111111-2222-4333-8444-555555555555"}]},
            "aaaaaaaabbbb4ccc8dddeeeeeeeeeeee, 11111111222243338444555555555555",
        ),
        # files
        ({"type": "files", "files": [{"name": "a.png"}, {"name": "b.pdf"}]}, "file:a.png, file:b.pdf"),
        # formula
        ({"type": "formula", "formula": {"type": "string", "string": "ok"}}, "ok"),
        ({"type": "formula", "formula": {"type": "number", "number": 0}}, "0"),
        ({"type": "formula", "formula": {"type": "boolean", "boolean": False}}, "false"),
        ({"type": "formula", "formula": {"type": "date", "date": {"start": "2026-03-01"}}}, "2026-03-01"),
        # rollup
        ({"type": "rollup", "rollup": {"type": "number", "number": 12, "function": "sum"}}, "12"),
        ({"type": "rollup", "rollup": {"type": "date", "date": {"start": "2026-03-01"}}}, "2026-03-01"),
        (
            {"type": "rollup", "rollup": {"type": "array", "array": [{"type": "title", "title": rich("One")}, {"type": "number", "number": 2}]}},
            "One, 2",
        ),
        # unique id and verification
        ({"type": "unique_id", "unique_id": {"prefix": "BUG", "number": 7}}, "BUG-7"),
        ({"type": "unique_id", "unique_id": {"prefix": None, "number": 7}}, "7"),
        ({"type": "verification", "verification": {"state": "verified"}}, "verified"),
        # types this renderer does not know
        ({"type": "button", "button": {}}, ""),
        ({"type": "place", "place": {"lat": 1}}, ""),
        ({"type": "something_new", "something_new": {"x": 1}}, ""),
    ],
)
def test_property_values_render_as_text(value: dict, text: str) -> None:
    assert render_value(value, file_text=file_text) == text


def test_records_keep_name_and_type_and_sort_by_name() -> None:
    properties = {
        "Status": {"id": "s", "type": "status", "status": {"name": "Done"}},
        "Action": {"id": "b", "type": "button", "button": {}},
    }

    assert render_properties(properties, file_text=file_text) == [
        {"name": "Action", "type": "button", "value": ""},
        {"name": "Status", "type": "status", "value": "Done"},
    ]
