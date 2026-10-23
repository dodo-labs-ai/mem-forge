"""The Teams Gene: its configuration, its selection, and Units stored before Teams joined the conversation layer."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.source_adapters.conversation import project_native
from memforge.conversations.records import record_revision
from memforge.conversations.units import UnitKind
from memforge.genes.teams_gene import TeamsGene
from memforge.models import ConfigFieldType, ContentItem
from memforge.source_adapters.teams import message_record, teams_unit_shape


def test_metadata_fields():
    meta = TeamsGene.metadata()
    assert meta.name == "teams"
    assert meta.display_name == "Microsoft Teams"
    assert meta.data_shape == "message"
    assert meta.auth_method == "browser_cookie"
    assert meta.default_sync_interval_minutes == 60


def test_config_schema_groups():
    schema = TeamsGene.config_schema()
    group_keys = [g.key for g in schema.groups]
    assert group_keys == ["connection", "scope", "sync"]


def test_config_schema_fields():
    schema = TeamsGene.config_schema()
    field_keys = [f.key for f in schema.fields]
    assert "region" in field_keys
    assert "conversation_ids" in field_keys
    assert "channels" not in field_keys
    assert "group_chats" not in field_keys
    assert "individual_chats" not in field_keys
    assert "initial_history_days" in field_keys
    assert "rolling_retention_days" in field_keys
    assert "max_age_days" not in field_keys
    assert "conversation_gap_minutes" not in field_keys
    assert "max_block_messages" not in field_keys
    retention = next(f for f in schema.fields if f.key == "rolling_retention_days")
    assert [(option.value, option.label) for option in retention.choices] == [
        ("forever", "Forever"),
        ("365", "1 year"),
        ("730", "2 years"),
        ("1095", "3 years"),
    ]


@pytest.mark.parametrize("retention_days", [365, 730, 1095])
def test_normalize_config_accepts_product_retention_presets(retention_days: int):
    config = {
        "conversation_ids": ["19:chat@example.test"],
        "rolling_retention_days": retention_days,
    }

    TeamsGene.normalize_config(config)

    assert config["rolling_retention_days"] == retention_days


@pytest.mark.parametrize("initial_history_days", [0, 366])
def test_normalize_config_bounds_initial_history_to_a_year(initial_history_days: int):
    config = {"conversation_ids": ["19:chat@example.test"], "initial_history_days": initial_history_days}

    with pytest.raises(ValueError, match="between 1 and 365 days"):
        TeamsGene.normalize_config(config)


def test_normalize_config_collapses_legacy_direct_selectors():
    config = {
        "conversation_ids": ["19:chat@example.test"],
        "channels": ["19:channel@thread.tacv2"],
        "group_chats": ["19:chat@example.test"],
        "individual_chats": ["19:person@example.test"],
    }

    TeamsGene.normalize_config(config)

    assert config == {
        "conversation_ids": [
            "19:chat@example.test",
            "19:channel@thread.tacv2",
            "19:person@example.test",
        ],
        "initial_history_days": 14,
        "rolling_retention_days": None,
    }


def test_normalize_config_rejects_display_name_selectors():
    with pytest.raises(
        ValueError,
        match="teams_sync_requires_direct_conversation_ids",
    ):
        TeamsGene.normalize_config({"channels": ["Team Name/General"]})


def test_legacy_max_age_migrates_to_initial_history_without_retention():
    config = {
        "conversation_ids": ["19:chat@example.test"],
        "max_age_days": 45,
        "conversation_gap_minutes": 30,
        "max_block_messages": 50,
    }

    TeamsGene.normalize_config(config)

    assert config == {
        "conversation_ids": ["19:chat@example.test"],
        "initial_history_days": 45,
        "rolling_retention_days": None,
    }


@pytest.mark.parametrize("selector", ["Payroll: General", "alice@example.com"])
def test_normalize_config_rejects_punctuation_that_is_not_provider_identity(
    selector: str,
):
    with pytest.raises(
        ValueError,
        match="teams_sync_requires_direct_conversation_ids",
    ):
        TeamsGene.normalize_config({"conversation_ids": [selector]})


def test_numeric_fields_use_integer_type():
    schema = TeamsGene.config_schema()
    numeric_fields = {"initial_history_days"}
    for f in schema.fields:
        if f.key in numeric_fields:
            assert f.field_type == ConfigFieldType.INTEGER, f"Field {f.key} should be INTEGER"


def test_config_validation_rejects_empty_scope():
    with pytest.raises(ValueError, match="teams_sync_requires_direct_conversation_ids"):
        TeamsGene(config={"region": "emea"}, source_id="test")


def test_config_accepts_channels_only():
    gene = TeamsGene(
        config={"channels": "19:channel@thread.tacv2"},
        source_id="test",
    )
    assert gene.source_id == "test"


def test_config_accepts_direct_conversation_ids():
    gene = TeamsGene(
        config={"conversation_ids": "19:group@thread.v2"},
        source_id="test",
    )

    assert gene.source_id == "test"


def test_the_selection_is_the_configured_conversations_and_time_scope():
    gene = TeamsGene(
        config={"conversation_ids": ["19:a@thread.v2", "19:b@thread.v2"], "rolling_retention_days": 365},
        source_id="src",
    )

    selection = gene.conversation_selection()

    assert selection.conversation_ids == ("19:a@thread.v2", "19:b@thread.v2")
    assert selection.initial_history_days == 14
    assert selection.rolling_retention_days == 365
    assert gene.selected_conversation_ids({"conversation_ids": ["19:a@thread.v2"]}) == ("19:a@thread.v2",)


def _window_package(window_id: str, messages: list[dict]) -> bytes:
    """A window package as a local agent uploaded it before Teams joined the conversation layer."""

    return json.dumps(
        {
            "package_kind": "teams_window_document",
            "conversation_id": "19:chat@thread.v2",
            "window_id": window_id,
            "raw_payload": {
                "conversation_type": "group_chat",
                "conversation_name": "Payroll",
                "channel_name": "Payroll",
                "messages": messages,
            },
        }
    ).encode()


THREAD_KEY = (
    "teams-thread:v1:eyJjb252ZXJzYXRpb25faWQiOiIxOTpjaGF0QHRocmVhZC52MiIsInJvb3Rfb3JfYW5jaG9yX21lc3NhZ2VfaWQiOiJyb290"
    "Iiwic291cmNlX2lkIjoic3JjIiwid2luZG93X3R5cGUiOiJ0aHJlYWQifQ"
)


def test_a_stored_window_package_is_read_as_the_same_records():
    gene = TeamsGene(config={"conversation_ids": ["19:chat@thread.v2"]}, source_id="src")
    messages = [
        {"id": "root", "from": "Ada", "content": "Question", "time": "2026-07-14T10:00:00+00:00", "is_root": True},
        {
            "id": "reply",
            "from": "Ben",
            "content": "Answer",
            "time": "2026-07-14T10:05:00+00:00",
            "edited_time": "2026-07-14T11:00:00+00:00",
            "rootMessageId": "root",
            "is_root": False,
        },
    ]
    payload = gene.stored_unit_payload(_window_package(THREAD_KEY, messages))

    assert payload is not None
    assert payload["unit"] == {"kind": UnitKind.THREAD.value, "provider_key": THREAD_KEY, "thread_root_id": "root"}
    # Each member is the record the window recorded, so its Observation revision stays the same.
    assert [member["record"] for member in payload["members"]] == [
        message_record(message, conversation_id="19:chat@thread.v2", conversation_name="Payroll") for message in messages
    ]
    item = ContentItem(item_id="doc-1", title="t", source_url="", last_modified=datetime(2026, 7, 14, tzinfo=timezone.utc))
    projection = project_native(item=item, native=payload)
    assert projection.unit_type == UnitKind.THREAD.value
    assert projection.locator["thread_root_id"] == "root"
    assert [value.semantic_value for value in projection.observations] == [
        member["record"] for member in payload["members"]
    ]


def test_a_stored_conversation_unit_is_read_as_it_is():
    gene = TeamsGene(config={"conversation_ids": ["19:chat@thread.v2"]}, source_id="src")

    assert gene.stored_unit_payload(json.dumps({"payload_kind": "conversation_unit"}).encode()) is None


def test_teams_unit_keys_name_their_kind_and_thread_root():
    assert teams_unit_shape(THREAD_KEY) == (UnitKind.THREAD, "root")
    assert teams_unit_shape("teams-block:v1:abc") == (UnitKind.WINDOW, None)
    with pytest.raises(ValueError, match="not a Teams Unit key"):
        teams_unit_shape("conv-window:v1:abc")


def test_a_record_revision_is_its_canonical_json_hash():
    record = {"content": "x", "time": "2026-07-14T10:00:00+00:00"}

    assert record_revision(record) == record_revision(dict(reversed(list(record.items()))))
