"""Naming each Jira Source's authentication method (migration 133): every stored Source authenticates as before."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from memforge.local_agent.source_contract import jira_config_with_auth_mode
from memforge.storage.database import Database, MigrationFailed, latest_schema_version

NAMING = 133
JIRA = "https://jira.example.test"


@pytest.mark.parametrize(
    ("config", "auth_mode"),
    [
        ({"base_url": JIRA}, "browser_cookie"),
        ({"base_url": JIRA, "auth_mode": None}, "browser_cookie"),
        ({"base_url": JIRA, "auth_mode": ""}, "browser_cookie"),
        ({"base_url": JIRA, "auth_mode": "", "pat_encrypted": "ciphertext"}, "browser_cookie"),
        ({"base_url": JIRA, "sync_mode": "local_agent"}, "browser_cookie"),
        ({"base_url": JIRA, "sync_mode": "local_agent", "pat_encrypted": "ciphertext"}, "browser_cookie"),
        ({"base_url": JIRA, "pat_encrypted": "ciphertext"}, "pat"),
        ({"base_url": JIRA, "auth_mode": None, "pat_encrypted": "ciphertext"}, "pat"),
        ({"base_url": JIRA, "pat": "jira-pat"}, "pat"),
    ],
)
def test_a_configuration_without_a_method_names_the_one_it_used(config, auth_mode) -> None:
    assert jira_config_with_auth_mode(config) == {**config, "auth_mode": auth_mode}


@pytest.mark.parametrize("auth_mode", ["browser_cookie", "pat"])
def test_a_configuration_that_names_its_method_is_kept(auth_mode) -> None:
    assert jira_config_with_auth_mode({"base_url": JIRA, "auth_mode": auth_mode, "pat_encrypted": "c"}) is None


async def _store_and_rewind(path: Path, sources: dict[str, tuple[str, object]]) -> None:
    db = Database(str(path))
    await db.connect()
    try:
        for source_id, (source_type, config) in sources.items():
            await db.upsert_source(
                id=source_id,
                type=source_type,
                name=source_id,
                config_json=json.dumps(config),
                access_policy="workspace",
                owner_user_id="owner-user",
            )
    finally:
        await db.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= ?", (NAMING,))
        connection.commit()


def _configs(path: Path) -> dict[str, dict]:
    with closing(sqlite3.connect(path)) as connection:
        return {row[0]: json.loads(row[1]) for row in connection.execute("SELECT id, config FROM sources")}


async def test_upgrade_writes_the_method_each_jira_source_used(tmp_path) -> None:
    path = tmp_path / "prior.db"
    session = {"base_url": JIRA, "projects": ["PAY"]}
    token = {"base_url": JIRA, "projects": ["OPS"], "pat_encrypted": "ciphertext"}
    named = {"base_url": JIRA, "projects": ["ARCH"], "auth_mode": "browser_cookie"}
    local = {"base_url": JIRA, "projects": ["CORE"], "sync_mode": "local_agent", "pat_encrypted": "ciphertext"}
    pages = {"auth_mode": "none", "page_url": "https://pages.example.test/org/site/"}
    await _store_and_rewind(
        path,
        {
            "src-session": ("jira", session),
            "src-token": ("jira", token),
            "src-named": ("jira", named),
            "src-local": ("jira", local),
            "src-listed": ("jira", ["not", "a", "mapping"]),
            "src-pages": ("github_pages", pages),
        },
    )

    db = Database(str(path))
    await db.connect()
    await db.close()

    configs = _configs(path)
    assert configs["src-session"] == {**session, "auth_mode": "browser_cookie"}
    assert configs["src-token"] == {**token, "auth_mode": "pat"}
    assert configs["src-named"] == named
    assert configs["src-local"] == {**local, "auth_mode": "browser_cookie"}
    assert configs["src-listed"] == ["not", "a", "mapping"]
    assert configs["src-pages"] == pages
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == latest_schema_version()


async def test_an_unreadable_jira_configuration_stops_the_upgrade_and_changes_nothing(tmp_path) -> None:
    path = tmp_path / "prior.db"
    session = {"base_url": JIRA, "projects": ["PAY"]}
    await _store_and_rewind(path, {"src-session": ("jira", session), "src-broken": ("jira", {})})
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE sources SET config = ? WHERE id = ?", ("{not json", "src-broken"))
        connection.commit()

    db = Database(str(path))
    with pytest.raises(MigrationFailed, match="Jira source src-broken has invalid JSON config"):
        await db.connect()

    assert _configs_without("src-broken", path) == {"src-session": session}


def _configs_without(broken: str, path: Path) -> dict[str, dict]:
    with closing(sqlite3.connect(path)) as connection:
        return {
            row[0]: json.loads(row[1])
            for row in connection.execute("SELECT id, config FROM sources WHERE id != ?", (broken,))
        }
