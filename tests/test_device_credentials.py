"""The provider credentials a device holds for its device Sources, and the report that names them (ADR 0058)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

import memforge.main as main
from memforge.local_agent import device_credentials, device_identity
from memforge.local_agent.credential_check import run_credential_check
from memforge.local_agent.device_credentials import (
    DEVICE_CREDENTIALS_KEYRING_SERVICE,
    DeviceCredentialsMissing,
    DeviceCredentialStore,
    check_browser_session,
)
from memforge.local_agent.device_lease import DeviceLeaseAnswers
from memforge.local_agent.device_identity import DeviceIdentityStore
from memforge.local_agent.device_report import DeviceReport, declared_acquisition_contracts
from memforge.release import installed_version
from tests.test_daemon_service import FakeKeyring

# The server the installed daemon service serves in these tests (FakeServiceManager's environment).
SERVER = "https://stored.example"
OTHER_SERVER = "https://other.example"


def test_secrets_live_in_the_keyring_per_source_and_field_and_the_index_names_only_fields(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", server=SERVER, keyring_store=keyring)

    store.set("src-wiki", {"token": "wiki-token-secret"})
    store.set("src-notes", {})

    assert store.source_ids() == frozenset({"src-wiki", "src-notes"})
    assert store.get("src-wiki") == {"token": "wiki-token-secret"}
    assert store.get("src-notes") == {}
    assert store.get("src-other") is None
    assert keyring.values == {(DEVICE_CREDENTIALS_KEYRING_SERVICE, f"{SERVER}#src-wiki/token"): "wiki-token-secret"}
    assert "wiki-token-secret" not in (tmp_path / "device-credentials.json").read_text(encoding="utf-8")


def test_setting_a_sources_credentials_again_replaces_them_and_deleting_forgets_them(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", server=SERVER, keyring_store=keyring)
    store.set("src-wiki", {"token": "old", "cookie": "session"})

    store.set("src-wiki", {"token": "new"})
    assert store.get("src-wiki") == {"token": "new"}
    assert keyring.values == {(DEVICE_CREDENTIALS_KEYRING_SERVICE, f"{SERVER}#src-wiki/token"): "new"}

    store.delete("src-wiki")
    assert store.source_ids() == frozenset()
    assert keyring.values == {}


def test_an_origins_session_is_deleted_with_the_last_held_source_that_signs_in_with_it(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", server=SERVER, keyring_store=keyring)
    store.hold_session("src-issues", "jira", "https://jira.example.test", "SESSION=one")
    store.hold_session("src-bugs", "jira", "https://jira.example.test", "SESSION=two")

    store.delete("src-issues")
    assert store.session("jira", "https://jira.example.test") == "SESSION=two"

    store.delete("src-bugs")
    assert store.session("jira", "https://jira.example.test") is None
    assert keyring.values == {}
    assert "SESSION" not in (tmp_path / "device-credentials.json").read_text(encoding="utf-8")


def test_concurrent_changes_to_the_index_each_keep_their_source(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    keyring = FakeKeyring()
    path = tmp_path / "device-credentials.json"
    source_ids = [f"src-{number}" for number in range(24)]

    def hold(source_id: str) -> None:
        DeviceCredentialStore(path, server=SERVER, keyring_store=keyring).set(source_id, {"token": source_id})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hold, source_ids))

    assert DeviceCredentialStore(path, server=SERVER, keyring_store=keyring).source_ids() == frozenset(source_ids)


def test_what_the_device_holds_for_one_server_is_apart_from_another_servers(tmp_path) -> None:
    keyring = FakeKeyring()
    path = tmp_path / "device-credentials.json"
    first = DeviceCredentialStore(path, server=SERVER, keyring_store=keyring)
    second = DeviceCredentialStore(path, server=OTHER_SERVER, keyring_store=keyring)
    first.set("src-wiki", {"token": "first-token"})
    first.hold_session("src-issues", "jira", "https://jira.example.test", "SESSION=first")
    second.set("src-wiki", {"token": "second-token"})
    second.hold_session("src-bugs", "jira", "https://jira.example.test", "SESSION=second")

    second.delete("src-wiki")
    second.delete("src-bugs")

    assert second.source_ids() == frozenset()
    assert first.source_ids() == frozenset({"src-wiki", "src-issues"})
    assert first.get("src-wiki") == {"token": "first-token"}
    assert first.session("jira", "https://jira.example.test") == "SESSION=first"


def test_a_daemon_leasing_from_another_server_keeps_the_first_servers_credentials(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    device_credentials.device_credentials(SERVER).set("src-wiki", {"token": "first-token"})
    other = device_credentials.device_credentials(OTHER_SERVER)
    report = DeviceReport.from_payload(main._this_device_report(OTHER_SERVER))
    assert report.source_ids == frozenset()

    # The other server releases every Source the report named, and none of the first server's.
    DeviceLeaseAnswers(other, log=lambda message: None).observe(
        {"jobs": [], "released_source_ids": sorted(report.source_ids) + ["src-wiki"]}
    )

    assert device_credentials.device_credentials(SERVER).get("src-wiki") == {"token": "first-token"}
    assert DeviceReport.from_payload(main._this_device_report(SERVER)).source_ids == frozenset({"src-wiki"})


def test_a_secret_missing_from_the_keyring_is_an_error(tmp_path) -> None:
    keyring = FakeKeyring()
    store = DeviceCredentialStore(tmp_path / "device-credentials.json", server=SERVER, keyring_store=keyring)
    store.set("src-wiki", {"token": "secret"})
    keyring.values.clear()

    with pytest.raises(ValueError, match="token of Source src-wiki"):
        store.get("src-wiki")


def test_the_daemons_report_names_this_device_its_version_contracts_and_held_sources() -> None:
    store = DeviceCredentialStore(device_credentials.device_credentials_path(), server=SERVER, keyring_store=FakeKeyring())
    store.set("src-wiki", {"token": "secret"})

    first = DeviceReport.from_payload(main._this_device_report(SERVER))
    second = DeviceReport.from_payload(main._this_device_report(SERVER))

    identity = DeviceIdentityStore(device_identity.device_identity_path()).load()
    assert identity is not None
    assert first == second
    assert (first.device_id, first.name) == (identity.device_id, identity.name)
    assert first.version == installed_version()
    assert dict(first.acquisition_contracts) == declared_acquisition_contracts()
    assert first.source_ids == frozenset({"src-wiki"})
    assert "secret" not in json.dumps(main._this_device_report(SERVER))


# -- What the device runs a Source's Gene with --

JIRA_ORIGIN = "https://jira.example.test"
JIRA_TOKEN_CONFIG = {"base_url": JIRA_ORIGIN, "auth_mode": "pat", "jql": "project = ACME"}
JIRA_COOKIE_CONFIG = {"base_url": JIRA_ORIGIN, "auth_mode": "browser_cookie", "jql": "project = ACME"}


@pytest.fixture
def keyring(monkeypatch) -> FakeKeyring:
    """The OS keyring this device's credential store reaches."""

    import sys

    fake = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return fake


def _store() -> DeviceCredentialStore:
    return device_credentials.device_credentials(SERVER)


def test_the_device_gene_runs_with_the_sources_secrets_and_its_origins_browser_session(keyring) -> None:
    store = _store()
    store.set("src-token", {"pat": "device-pat"})
    store.hold_session("src-cookie", "jira", JIRA_ORIGIN, "SESSION=device")

    assert store.gene_config("src-token", "jira", JIRA_TOKEN_CONFIG) == {**JIRA_TOKEN_CONFIG, "pat": "device-pat"}
    assert store.gene_config("src-cookie", "jira", JIRA_COOKIE_CONFIG) == {
        **JIRA_COOKIE_CONFIG,
        "jira_cookie": "SESSION=device",
    }
    with pytest.raises(DeviceCredentialsMissing, match="memforge source credentials set src-other"):
        store.gene_config("src-other", "jira", JIRA_TOKEN_CONFIG)
    keyring.values.clear()
    with pytest.raises(DeviceCredentialsMissing, match="sign in to the Source again"):
        store.gene_config("src-cookie", "jira", JIRA_COOKIE_CONFIG)


def test_the_browser_session_is_checked_against_the_provider_on_the_device(monkeypatch) -> None:
    from memforge.auth import browser_session

    checked: list[tuple] = []

    async def accept(origin, cookie_header, tls_config):
        checked.append((origin, cookie_header))
        return {"name": "ada"}

    async def refuse(origin, cookie_header, tls_config):
        raise RuntimeError("401 from the provider")

    provider = browser_session.get_provider("jira")
    config = {**JIRA_COOKIE_CONFIG, "jira_cookie": "SESSION=device"}
    monkeypatch.setitem(browser_session._PROVIDERS, "jira", replace(provider, validate_session=accept))
    asyncio.run(check_browser_session("jira", config))
    asyncio.run(check_browser_session("jira", JIRA_TOKEN_CONFIG))
    monkeypatch.setitem(browser_session._PROVIDERS, "jira", replace(provider, validate_session=refuse))

    assert checked == [(JIRA_ORIGIN, "SESSION=device")]
    with pytest.raises(DeviceCredentialsMissing, match="has expired") as refused:
        asyncio.run(check_browser_session("jira", config))
    assert refused.value.retryable is False


# -- The lease answer --


def test_the_device_deletes_released_credentials_and_logs_only_when_contract_agreement_changes(keyring) -> None:
    store = _store()
    store.set("src-kept", {"pat": "kept"})
    store.set("src-moved", {"pat": "moved"})
    logged: list[str] = []
    answers = DeviceLeaseAnswers(store, log=logged.append)
    ours = declared_acquisition_contracts()
    other = {**ours, "jira": ours["jira"] + 1}

    def answer(contracts, released=()):
        return {
            "jobs": [],
            "server": {"version": "9.9.9", "acquisition_contracts": contracts},
            "released_source_ids": list(released),
        }

    answers.observe(answer(ours, released=["src-moved"]))
    answers.observe(answer(ours))
    answers.observe(answer(other))
    answers.observe(answer(other))
    answers.observe(answer(ours))

    assert store.source_ids() == frozenset({"src-kept"})
    assert (DEVICE_CREDENTIALS_KEYRING_SERVICE, f"{SERVER}#src-moved/pat") not in keyring.values
    assert len(logged) == 3
    assert "src-moved" in logged[0]
    assert "9.9.9" in logged[1] and "memforge upgrade" in logged[1]
    assert "match the server's again" in logged[2]


# -- credential_check --


def _capture_provider(monkeypatch):
    from memforge.auth import browser_session

    async def capture(base_url, *, browser, tls_config):
        return browser_session.CapturedBrowserSession(
            origin=JIRA_ORIGIN, cookie_header="SESSION=captured", browser=browser, principal={"name": "ada"}
        )

    monkeypatch.setitem(
        browser_session._PROVIDERS, "jira", replace(browser_session.get_provider("jira"), capture_session=capture)
    )


def _check(payload, *, uploaded=None, opened=None):
    uploads: list[tuple] = []

    def upload(provider, captured):
        uploads.append((provider, captured.origin, captured.cookie_header))
        return uploaded or {"status": "active"}

    result = asyncio.run(
        run_credential_check(
            payload,
            credentials=_store(),
            open_gene=opened or (lambda source_id, config: pytest.fail("no Gene is opened")),
            upload_session=upload,
            browser="chrome",
        )
    )
    return result, uploads


def test_a_device_sources_browser_session_stays_in_the_devices_keyring(monkeypatch, keyring) -> None:
    _capture_provider(monkeypatch)

    result, uploads = _check(
        {"source_id": "src-cookie", "source_type": "jira", "executor": "device", "config": JIRA_COOKIE_CONFIG}
    )

    assert (result.authenticated, result.origin, result.error) == (True, JIRA_ORIGIN, None)
    assert uploads == []
    assert _store().session("jira", JIRA_ORIGIN) == "SESSION=captured"
    # The device now holds the Source, so its report names it.
    assert _store().source_ids() == frozenset({"src-cookie"})


def test_a_server_sources_browser_session_is_stored_on_the_server(monkeypatch, keyring) -> None:
    _capture_provider(monkeypatch)

    result, uploads = _check(
        {"source_id": "src-cookie", "source_type": "jira", "executor": "server", "config": JIRA_COOKIE_CONFIG}
    )
    changed, _ = _check(
        {"source_id": "src-cookie", "source_type": "jira", "executor": "server", "config": JIRA_COOKIE_CONFIG},
        uploaded={"code": "jira_principal_changed", "detail": {"origin": JIRA_ORIGIN, "new_principal_id": "bob"}},
    )

    assert result.authenticated is True
    assert uploads == [("jira", JIRA_ORIGIN, "SESSION=captured")]
    assert keyring.values == {}
    assert (changed.error, changed.new_principal_id, changed.retryable) == ("principal_changed", "bob", False)


def test_a_device_sources_token_is_checked_by_the_types_authenticate(keyring) -> None:
    _store().set("src-token", {"pat": "device-pat"})
    payload = {"source_id": "src-token", "source_type": "jira", "executor": "device", "config": JIRA_TOKEN_CONFIG}
    opened: list[dict] = []

    class Gene:
        def __init__(self, config):
            opened.append(config)

        async def authenticate(self):
            if self_refuses:
                raise RuntimeError("401 Unauthorized")

        async def close(self):
            return None

    self_refuses = False
    accepted, _ = _check(payload, opened=lambda source_id, config: Gene(config))
    self_refuses = True
    refused, _ = _check(payload, opened=lambda source_id, config: Gene(config))
    missing, _ = _check({**payload, "source_id": "src-unknown"})

    assert accepted.authenticated is True
    assert opened[0] == {**JIRA_TOKEN_CONFIG, "pat": "device-pat"}
    assert (refused.error, refused.retryable) == ("401 Unauthorized", False)
    assert "memforge source credentials set src-unknown" in (missing.error or "")


# -- The device's Sources on the command line --


class _StatusClient:
    status: dict = {}

    def __init__(self, *, target, api_token):
        self.target = target

    def get_local_agent_status(self):
        return self.status


def _status(sources, contracts=None) -> dict:
    return {
        "status": "online",
        "last_seen_at": None,
        "devices": [],
        "sources": sources,
        "server": {"version": installed_version(), "acquisition_contracts": contracts or declared_acquisition_contracts()},
        "checked_at": "2026-11-26T09:00:00+00:00",
        "stale_after_seconds": 90,
    }


@pytest.fixture
def daemon(monkeypatch):
    from tests.test_daemon_service import FakeServiceManager

    monkeypatch.setattr(main, "_daemon_service_manager", FakeServiceManager)
    monkeypatch.setattr(main, "ToolClient", _StatusClient)
    monkeypatch.setattr(main, "_local_agent_status_payload", lambda ctx, verbose: {"status": "running"})
    return _StatusClient


def test_credentials_set_prompts_for_the_secret_fields_and_keeps_them_in_the_keyring(daemon, keyring, tmp_path):
    from click.testing import CliRunner

    daemon.status = _status([{"id": "src-token", "name": "Tracker", "type": "jira", "config": JIRA_TOKEN_CONFIG}])
    env = {"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")}

    stored = CliRunner().invoke(main.cli, ["source", "credentials", "set", "src-token"], input="device-pat\n", env=env)
    unknown = CliRunner().invoke(main.cli, ["source", "credentials", "set", "src-server"], env=env)

    assert stored.exit_code == 0, stored.output
    assert json.loads(stored.output[stored.output.index("{") :]) == {"ok": True, "source_id": "src-token", "fields": ["pat"]}
    assert "device-pat" not in stored.output
    assert _store().get("src-token") == {"pat": "device-pat"}
    assert unknown.exit_code == 1
    assert "not one of your Sources that run on your computer" in unknown.output


def test_credentials_set_sends_a_browser_session_source_to_sign_in(daemon, keyring, tmp_path):
    from click.testing import CliRunner

    daemon.status = _status([{"id": "src-cookie", "name": "Tracker", "type": "jira", "config": JIRA_COOKIE_CONFIG}])

    result = CliRunner().invoke(
        main.cli, ["source", "credentials", "set", "src-cookie"], env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "c.toml")}
    )

    assert result.exit_code == 1
    assert "signs in with a browser session" in result.output
    assert _store().source_ids() == frozenset()


def test_daemon_status_lists_each_device_source_as_ready_or_waiting_with_its_command(daemon, keyring, tmp_path):
    from click.testing import CliRunner

    _store().set("src-ready", {"pat": "device-pat"})
    _store().set("src-notion", {"token": "device-token"})
    ours = declared_acquisition_contracts()
    daemon.status = _status(
        [
            {"id": "src-ready", "name": "Ready", "type": "jira", "config": JIRA_TOKEN_CONFIG},
            {"id": "src-token", "name": "Token", "type": "jira", "config": JIRA_TOKEN_CONFIG},
            {"id": "src-cookie", "name": "Cookie", "type": "jira", "config": JIRA_COOKIE_CONFIG},
            {"id": "src-notion", "name": "Notes", "type": "notion", "config": {}},
        ],
        contracts={**ours, "notion": ours["notion"] + 1},
    )

    result = CliRunner().invoke(main.cli, ["daemon", "status"], env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "c.toml")})

    assert result.exit_code == 0, result.output
    states = {entry["id"]: entry for entry in json.loads(result.output)["sources"]}
    assert states["src-ready"]["state"] == "ready"
    assert (states["src-token"]["state"], states["src-token"]["command"]) == (
        "waits_for_credential",
        "memforge source credentials set src-token",
    )
    assert states["src-cookie"]["state"] == "waits_for_credential"
    assert "Sign in" in states["src-cookie"]["instruction"]
    assert (states["src-notion"]["state"], states["src-notion"]["command"]) == ("waits_for_update", "memforge upgrade")
