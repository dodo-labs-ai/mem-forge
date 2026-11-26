"""The provider credentials a device holds for its device Sources, and the report that names them (ADR 0058)."""

from __future__ import annotations

import json

import pytest

import memforge.main as main
from memforge.local_agent import device_credentials, device_identity
from memforge.local_agent.device_credentials import DEVICE_CREDENTIALS_KEYRING_SERVICE, DeviceCredentialStore
from memforge.local_agent.device_identity import DeviceIdentityStore
from memforge.local_agent.device_report import DeviceReport, declared_acquisition_contracts
from memforge.release import installed_version
from tests.test_daemon_service import FakeKeyring


def test_secrets_live_in_the_keyring_per_source_and_field_and_the_index_names_only_fields(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", keyring_store=keyring)

    store.set("src-wiki", {"token": "wiki-token-secret"})
    store.set("src-notes", {})

    assert store.source_ids() == frozenset({"src-wiki", "src-notes"})
    assert store.get("src-wiki") == {"token": "wiki-token-secret"}
    assert store.get("src-notes") == {}
    assert store.get("src-other") is None
    assert keyring.values == {(DEVICE_CREDENTIALS_KEYRING_SERVICE, "src-wiki/token"): "wiki-token-secret"}
    assert "wiki-token-secret" not in (tmp_path / "device-credentials.json").read_text(encoding="utf-8")


def test_setting_a_sources_credentials_again_replaces_them_and_deleting_forgets_them(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", keyring_store=keyring)
    store.set("src-wiki", {"token": "old", "cookie": "session"})

    store.set("src-wiki", {"token": "new"})
    assert store.get("src-wiki") == {"token": "new"}
    assert keyring.values == {(DEVICE_CREDENTIALS_KEYRING_SERVICE, "src-wiki/token"): "new"}

    store.delete("src-wiki")
    assert store.source_ids() == frozenset()
    assert keyring.values == {}


def test_a_secret_missing_from_the_keyring_is_an_error(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", keyring_store=keyring)
    store.set("src-wiki", {"token": "secret"})
    keyring.values.clear()

    with pytest.raises(ValueError, match="token of Source src-wiki"):
        store.get("src-wiki")


def test_the_daemons_report_names_this_device_its_version_contracts_and_held_sources() -> None:
    store = DeviceCredentialStore(device_credentials.device_credentials_path(), keyring_store=FakeKeyring())
    store.set("src-wiki", {"token": "secret"})

    first = DeviceReport.from_payload(main._this_device_report())
    second = DeviceReport.from_payload(main._this_device_report())

    identity = DeviceIdentityStore(device_identity.device_identity_path()).load()
    assert identity is not None
    assert first == second
    assert (first.device_id, first.name) == (identity.device_id, identity.name)
    assert first.version == installed_version()
    assert dict(first.acquisition_contracts) == declared_acquisition_contracts()
    assert first.source_ids == frozenset({"src-wiki"})
    assert "secret" not in json.dumps(main._this_device_report())
