"""Model bindings: how settings name a model, where they are stored and how they resolve."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memforge.config import AppConfig
from memforge.llm import bindings
from memforge.llm.bindings import (
    ValidationResult,
    binding_from_record,
    binding_from_settings,
    binding_record,
    build_binding,
    environment_binding,
    key_last4,
    new_stored_secret,
    resolve_active_binding,
    resolve_active_invocation,
    resolve_credential,
    state_after_change,
    state_after_check,
    validation_record,
)
from memforge.llm.providers import ModelConfigError, ResolvedInvocation
from memforge.runtime import resolve_models
from memforge.source_secrets import decrypt_secret
from memforge.storage.database import Database

SECRET_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
GENERATION_KEY = "generation-secret-key-1111"
EMBEDDING_KEY = "embedding-secret-key-2222"


@pytest.fixture(autouse=True)
def secret_key(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", SECRET_KEY)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "bindings.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_model = "claude-sonnet-4"
    config.llm.enrichment_base_url = "https://api.anthropic.com"
    config.llm.enrichment_api_key = ""
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_base_url = "https://api.openai.com/v1"
    config.llm.embedding_api_key = ""
    config.llm.decision_model = ""
    return config


class UnreadableStore:
    """A database that must not be touched."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the database was used: {name}")


async def _store_active(db: Database, kind: str, *, api_key: str | None = None, **settings):
    binding = build_binding(kind=kind, config_source="settings", **settings)
    secret = None
    if api_key is not None:
        secret_ref, secret = new_stored_secret(api_key, origin=binding.origin)
        binding = build_binding(kind=kind, config_source="settings", secret_ref=secret_ref, **settings)
    await db.put_model_binding(kind, "active", binding_record(binding), secret=secret)
    return binding


# -- How model name, base URL and key name a provider ---------------------------------


@pytest.mark.parametrize(
    ("kind", "model", "base_url", "has_key", "expected"),
    [
        (
            "generation",
            "claude-sonnet-4",
            "https://api.anthropic.com",
            True,
            ("anthropic", "anthropic/claude-sonnet-4", "https://api.anthropic.com", "api_key"),
        ),
        (
            "generation",
            "anthropic--claude-sonnet-latest",
            "http://localhost:6655/anthropic",
            True,
            ("anthropic", "anthropic/anthropic--claude-sonnet-latest", "http://localhost:6655/anthropic", "api_key"),
        ),
        ("generation", "provider/chat-model", "", False, ("litellm", "provider/chat-model", None, "environment")),
        (
            "generation",
            "openai/gpt-5-mini",
            "https://proxy.example.test/v1",
            True,
            ("litellm", "openai/gpt-5-mini", "https://proxy.example.test/v1", "api_key"),
        ),
        (
            "embedding",
            "text-embedding-3-small",
            "https://api.openai.com/v1",
            True,
            ("openai", "openai/text-embedding-3-small", "https://api.openai.com/v1", "api_key"),
        ),
        (
            "embedding",
            "BAAI/bge-m3",
            "https://embed.example.test/v1",
            True,
            ("openai_compatible", "openai/BAAI/bge-m3", "https://embed.example.test/v1", "api_key"),
        ),
        (
            "embedding",
            "local-embedding",
            "http://localhost:11434/v1",
            False,
            ("openai_compatible", "openai/local-embedding", "http://localhost:11434/v1", "none"),
        ),
        ("embedding", "ollama/nomic-embed-text", "", False, ("litellm", "ollama/nomic-embed-text", None, "environment")),
        (
            "embedding",
            "sap/text-embedding-3-small",
            "",
            False,
            ("litellm", "sap/text-embedding-3-small", None, "environment"),
        ),
    ],
)
def test_a_model_name_a_base_url_and_a_key_name_a_binding(kind, model, base_url, has_key, expected) -> None:
    binding = binding_from_settings(
        kind,
        model=model,
        base_url=base_url,
        secret_ref="secret:one" if has_key else None,
        config_source="environment",
    )

    assert binding is not None
    assert (binding.provider, binding.route, binding.endpoint, binding.auth_strategy) == expected


@pytest.mark.parametrize(
    ("kind", "model", "base_url"),
    [
        ("generation", "", "https://api.anthropic.com"),
        ("generation", "claude-sonnet-4", "https://api.anthropic.com"),
        ("embedding", "text-embedding-3-small", "https://api.openai.com/v1"),
        ("embedding", "bare-model", ""),
        ("embedding", "BAAI/bge-m3", ""),
        ("embedding", "bare-model", "not-a-url"),
    ],
    ids=[
        "no_model",
        "anthropic_without_key",
        "openai_without_key",
        "model_id_without_endpoint",
        "slashed_model_id_without_endpoint",
        "unusable_endpoint",
    ],
)
def test_settings_that_describe_no_usable_model_name_no_binding(kind, model, base_url) -> None:
    binding = binding_from_settings(
        kind,
        model=model,
        base_url=base_url,
        secret_ref=None,
        config_source="environment",
    )

    assert binding is None


def test_a_fresh_installation_has_no_model_of_either_kind(tmp_path) -> None:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = ""
    config.llm.embedding_api_key = ""

    assert environment_binding(config, "generation", config_source="environment") is None
    assert environment_binding(config, "embedding", config_source="environment") is None


# -- Resolution ---------------------------------------------------------------------


async def test_environment_values_describe_a_kind_with_nothing_stored(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = GENERATION_KEY

    binding = await resolve_active_binding(db, config, "generation")

    assert binding is not None
    assert binding.config_source == "environment"
    assert binding.state == "not_validated"
    assert binding.secret_ref == "env:MEMFORGE_ENRICHMENT_API_KEY"
    assert binding.saved_at is None
    assert await db.get_model_binding("generation", "active") is None
    assert await resolve_active_invocation(db, config, "generation") == ResolvedInvocation(
        route="anthropic/claude-sonnet-4",
        api_base="https://api.anthropic.com",
        api_key=GENERATION_KEY,
        provider="anthropic",
    )


async def test_a_stored_binding_wins_over_the_environment(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = GENERATION_KEY
    await _store_active(
        db,
        "generation",
        provider="openai",
        model_id="gpt-5-mini",
        api_key="stored-openai-key-3333",
    )

    binding = await resolve_active_binding(db, config, "generation")
    invocation = await resolve_active_invocation(db, config, "generation")

    assert binding.config_source == "settings"
    assert binding.saved_at is not None
    assert invocation == ResolvedInvocation(
        route="openai/gpt-5-mini",
        api_base="https://api.openai.com/v1",
        api_key="stored-openai-key-3333",
        provider="openai",
    )


async def test_each_kind_resolves_on_its_own(db, tmp_path) -> None:
    config = _config(tmp_path)
    config.llm.enrichment_api_key = GENERATION_KEY
    await _store_active(db, "embedding", provider="ollama", model_id="nomic-embed-text")

    models = await resolve_models(db, config)

    assert models.generation_route == "anthropic/claude-sonnet-4"
    assert models.embedding == ResolvedInvocation(
        route="openai/nomic-embed-text",
        api_base="http://localhost:11434/v1",
        api_key="no-api-key",
        provider="ollama",
    )


async def test_two_databases_resolve_their_own_bindings(tmp_path) -> None:
    config = _config(tmp_path)
    first = Database(str(tmp_path / "first.db"))
    second = Database(str(tmp_path / "second.db"))
    await first.connect()
    await second.connect()
    try:
        await _store_active(first, "embedding", provider="ollama", model_id="first-model")
        await _store_active(second, "embedding", provider="ollama", model_id="second-model")

        routes = [
            (await resolve_active_invocation(database, config, "embedding")).route
            for database in (first, second, first)
        ]
    finally:
        await first.close()
        await second.close()

    assert routes == ["openai/first-model", "openai/second-model", "openai/first-model"]


async def test_a_deployment_managed_installation_compiles_from_the_environment_and_stores_nothing(
    db, tmp_path
) -> None:
    config = _config(tmp_path)
    config.server.llm_config_writable = False
    config.llm.enrichment_model = "sap/anthropic--claude-4.6-sonnet"
    config.llm.enrichment_base_url = ""
    config.llm.embedding_model = "sap/text-embedding-3-small"
    config.llm.embedding_base_url = ""
    await _store_active(db, "generation", provider="ollama", model_id="stored-and-ignored")
    stored_before = sqlite3.connect(db.db_path).execute("SELECT kind, slot FROM model_bindings").fetchall()

    binding = await resolve_active_binding(UnreadableStore(), config, "generation")
    models = await resolve_models(UnreadableStore(), config)

    assert binding.config_source == "deployment"
    assert binding.provider == "litellm"
    assert binding.auth_strategy == "environment"
    assert models.generation == ResolvedInvocation(route="sap/anthropic--claude-4.6-sonnet", provider="litellm")
    assert models.embedding == ResolvedInvocation(route="sap/text-embedding-3-small", provider="litellm")
    assert sqlite3.connect(db.db_path).execute("SELECT kind, slot FROM model_bindings").fetchall() == stored_before


async def test_a_deployment_reads_the_environment_on_every_resolution(tmp_path) -> None:
    config = _config(tmp_path)
    config.server.llm_config_writable = False
    config.llm.enrichment_model = "sap/first-model"
    config.llm.enrichment_base_url = ""

    first = await resolve_active_invocation(UnreadableStore(), config, "generation")
    config.llm.enrichment_model = "sap/second-model"
    second = await resolve_active_invocation(UnreadableStore(), config, "generation")

    assert (first.route, second.route) == ("sap/first-model", "sap/second-model")


async def test_a_binding_whose_key_is_gone_resolves_to_no_invocation(db, tmp_path, caplog) -> None:
    config = _config(tmp_path)
    binding = build_binding(
        kind="generation",
        provider="openai",
        model_id="gpt-5-mini",
        secret_ref="env:MEMFORGE_ENRICHMENT_API_KEY",
        config_source="settings",
    )
    await db.put_model_binding("generation", "active", binding_record(binding))

    with caplog.at_level("WARNING"):
        invocation = await resolve_active_invocation(db, config, "generation")

    assert invocation is None
    assert "credential_unavailable" in caplog.text


# -- Credentials --------------------------------------------------------------------


async def test_an_api_key_is_stored_encrypted_and_the_binding_holds_only_a_reference(db) -> None:
    binding = await _store_active(
        db,
        "generation",
        provider="openai",
        model_id="gpt-5-mini",
        api_key=GENERATION_KEY,
    )

    with sqlite3.connect(db.db_path) as raw:
        stored_binding = raw.execute("SELECT * FROM model_bindings").fetchall()
        secret_id, ciphertext, origin, last4 = raw.execute(
            "SELECT id, ciphertext, origin, last4 FROM model_secrets"
        ).fetchone()

    assert binding.secret_ref == f"secret:{secret_id}"
    assert GENERATION_KEY not in repr(stored_binding)
    assert ciphertext.startswith("enc:v1:")
    assert GENERATION_KEY not in ciphertext
    assert decrypt_secret(ciphertext) == GENERATION_KEY
    assert origin == "https://api.openai.com"
    assert last4 == "1111"


async def test_a_stored_key_is_released_only_to_the_origin_it_was_saved_for(db, tmp_path) -> None:
    saved = await _store_active(
        db,
        "generation",
        provider="openai_compatible",
        endpoint="https://saved.example.test/v1",
        model_id="model-a",
        api_key=GENERATION_KEY,
    )
    moved = build_binding(
        kind="generation",
        provider="openai_compatible",
        endpoint="https://other.example.test/v1",
        model_id="model-a",
        secret_ref=saved.secret_ref,
        config_source="settings",
    )

    assert await resolve_credential(db, _config(tmp_path), saved) == GENERATION_KEY
    with pytest.raises(ModelConfigError) as refused:
        await resolve_credential(db, _config(tmp_path), moved)
    assert refused.value.code == "api_key_origin_changed"
    assert GENERATION_KEY not in refused.value.message


async def test_a_key_no_binding_refers_to_is_deleted(db) -> None:
    await _store_active(db, "generation", provider="openai", model_id="gpt-5-mini", api_key=GENERATION_KEY)
    await _store_active(db, "generation", provider="openai", model_id="gpt-5-mini", api_key="replacement-key-4444")

    with sqlite3.connect(db.db_path) as raw:
        last4 = [row[0] for row in raw.execute("SELECT last4 FROM model_secrets")]

    assert last4 == ["4444"]


async def test_a_key_stays_while_either_slot_refers_to_it(db) -> None:
    active = await _store_active(db, "generation", provider="openai", model_id="gpt-5-mini", api_key=GENERATION_KEY)
    candidate = build_binding(
        kind="generation",
        provider="openai",
        model_id="gpt-5",
        secret_ref=active.secret_ref,
        config_source="settings",
    )
    await db.put_model_binding("generation", "candidate", binding_record(candidate))
    await db.delete_model_binding("generation", "active")

    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM model_secrets").fetchone() == (1,)
    await db.delete_model_binding("generation", "candidate")
    with sqlite3.connect(db.db_path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM model_secrets").fetchone() == (0,)


@pytest.mark.parametrize(("api_key", "shown"), [("sk-0123456789abcd", "abcd"), ("short-key", ""), ("", "")])
def test_only_a_long_key_shows_its_last_four_characters(api_key, shown) -> None:
    assert key_last4(api_key) == shown


# -- States -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "changed", "expected"),
    [
        ("ready", True, "needs_revalidation"),
        ("ready", False, "ready"),
        ("not_validated", True, "not_validated"),
        ("needs_revalidation", True, "needs_revalidation"),
        ("needs_revalidation", False, "needs_revalidation"),
    ],
)
def test_a_change_to_what_a_check_vouched_for_leaves_a_ready_binding_to_be_checked_again(
    state, changed, expected
) -> None:
    assert state_after_change(state, previous_digest="a", digest="b" if changed else "a") == expected


@pytest.mark.parametrize(
    ("state", "passed", "expected"),
    [
        ("not_validated", True, "ready"),
        ("needs_revalidation", True, "ready"),
        ("ready", True, "ready"),
        ("ready", False, "needs_revalidation"),
        ("not_validated", False, "not_validated"),
        ("needs_revalidation", False, "needs_revalidation"),
    ],
)
def test_a_check_makes_a_binding_ready_and_a_failed_one_unsettles_a_ready_binding(
    state, passed, expected
) -> None:
    assert state_after_check(state, passed=passed) == expected


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint": "https://other.example.test/v1"},
        {"provider": "openai"},
        {"auth_strategy": "none"},
        {"model_id": "model-b"},
        {"parameters": {"dimensions": 256}},
    ],
    ids=["endpoint", "provider", "auth_strategy", "model", "dimensions"],
)
def test_the_digest_changes_with_each_setting_a_check_vouches_for(change) -> None:
    settings = {
        "kind": "embedding",
        "provider": "openai_compatible",
        "endpoint": "https://embed.example.test/v1",
        "auth_strategy": "api_key",
        "model_id": "model-a",
        "config_source": "settings",
    }

    assert build_binding(**settings).digest != build_binding(**{**settings, **change}).digest


def test_the_digest_does_not_change_with_where_the_settings_came_from() -> None:
    settings = {"kind": "generation", "provider": "openai", "model_id": "gpt-5-mini", "secret_ref": "secret:one"}

    first = build_binding(**settings, config_source="settings")
    second = build_binding(**settings, config_source="environment")

    assert first.digest == second.digest


def test_the_digest_changes_with_the_key_the_binding_refers_to() -> None:
    settings = {"kind": "generation", "provider": "openai", "model_id": "gpt-5-mini", "config_source": "settings"}

    saved = build_binding(**settings, secret_ref="secret:one")
    replaced = build_binding(**settings, secret_ref="secret:two")
    from_environment = build_binding(**settings, secret_ref="env:MEMFORGE_ENRICHMENT_API_KEY")

    assert len({saved.digest, replaced.digest, from_environment.digest}) == 3


def test_the_digest_changes_with_the_token_limits_of_a_generation_binding() -> None:
    settings = {"kind": "generation", "provider": "ollama", "model_id": "qwen3", "config_source": "settings"}

    plain = build_binding(**settings)
    limited = build_binding(**settings, token_limits={"context_window_tokens": 32_768, "max_output_tokens": 4_096})

    assert plain.token_limits == {}
    assert limited.token_limits == {"context_window_tokens": 32_768, "max_output_tokens": 4_096}
    assert plain.digest != limited.digest


@pytest.mark.parametrize(
    ("kind", "token_limits", "error_code"),
    [
        ("generation", {"context_window_tokens": 0}, "token_limit_invalid"),
        ("generation", {"max_output_tokens": "4096"}, "token_limit_invalid"),
        ("generation", {"max_input_tokens": 8_000}, "token_limit_not_supported"),
        ("embedding", {"context_window_tokens": 8_000}, "token_limit_not_supported"),
    ],
)
def test_a_token_limit_a_binding_cannot_state_is_refused(kind, token_limits, error_code) -> None:
    with pytest.raises(ModelConfigError) as refusal:
        build_binding(
            kind=kind,
            provider="ollama",
            model_id="local-model",
            token_limits=token_limits,
            config_source="settings",
        )

    assert refusal.value.code == error_code


async def test_a_ready_binding_needs_revalidation_once_the_installed_litellm_changes(db, monkeypatch) -> None:
    binding = build_binding(kind="embedding", provider="ollama", model_id="nomic-embed-text", config_source="settings")
    validation = ValidationResult(checked_at="2026-10-14T00:00:00+00:00", digest=binding.digest, facts={"dimension": 3})
    await db.put_model_binding(
        "embedding",
        "active",
        {**binding_record(binding), "state": "ready", "validation": validation_record(validation)},
    )
    assert binding_from_record(await db.get_model_binding("embedding", "active")).state == "ready"

    monkeypatch.setattr(bindings, "installed_litellm_version", lambda: "9.9.9")
    upgraded = binding_from_record(await db.get_model_binding("embedding", "active"))

    assert upgraded.state == "needs_revalidation"
    assert upgraded.validation.digest != upgraded.digest
    assert upgraded.validation.facts == {"dimension": 3}


async def test_the_latest_validation_is_stored_on_the_binding_without_content(db) -> None:
    binding = build_binding(kind="embedding", provider="ollama", model_id="nomic-embed-text", config_source="settings")
    stored = await db.put_model_binding("embedding", "candidate", binding_record(binding))
    first = ValidationResult(checked_at="2026-10-14T00:00:00+00:00", digest=binding.digest, error_code="timeout")
    second = ValidationResult(
        checked_at="2026-10-14T00:01:00+00:00",
        digest=binding.digest,
        facts={"dimension": 3, "latency_ms": 12},
    )

    for validation, state in ((first, "not_validated"), (second, "ready")):
        assert await db.record_model_binding_validation(
            "embedding",
            "candidate",
            updated_at=stored["updated_at"],
            state=state,
            validation=validation_record(validation),
        )

    latest = binding_from_record(await db.get_model_binding("embedding", "candidate"))
    with sqlite3.connect(db.db_path) as raw:
        columns = [row[1] for row in raw.execute("PRAGMA table_info(model_bindings)")]
    assert latest.state == "ready"
    assert latest.validation == second
    assert {"validated_at", "validation_digest", "validation_facts_json", "validation_error_code"} <= set(columns)
    assert not [column for column in columns if "prompt" in column or "response" in column]


async def test_a_validation_is_not_recorded_on_a_binding_saved_again_since(db) -> None:
    binding = build_binding(kind="embedding", provider="ollama", model_id="nomic-embed-text", config_source="settings")
    checked = await db.put_model_binding("embedding", "candidate", binding_record(binding))
    saved_again = await db.put_model_binding("embedding", "candidate", binding_record(binding))
    validation = validation_record(ValidationResult(checked_at="2026-10-14T00:00:00+00:00", digest=binding.digest))

    recorded = await db.record_model_binding_validation(
        "embedding", "candidate", updated_at=checked["updated_at"], state="ready", validation=validation
    )
    promoted = await db.promote_model_candidate(
        "embedding", updated_at=checked["updated_at"], state="ready", validation=validation
    )

    assert checked["updated_at"] != saved_again["updated_at"]
    assert (recorded, promoted) == (False, False)
    assert (await db.get_model_binding("embedding", "candidate"))["state"] == "not_validated"
    assert await db.get_model_binding("embedding", "active") is None


# -- Settings stored by an earlier version ------------------------------------------


async def _database_with_llm_config(path: Path, row: dict[str, str | None] | None) -> None:
    """Prepare a database as the version before model bindings left it."""
    database = Database(str(path))
    await database.connect()
    await database.close()
    with sqlite3.connect(path) as raw:
        raw.execute("DELETE FROM schema_migrations WHERE version = 113")
        raw.execute("DROP TABLE model_bindings")
        raw.execute("DROP TABLE model_secrets")
        raw.execute(
            """CREATE TABLE llm_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enrichment_model TEXT, enrichment_base_url TEXT, enrichment_api_key TEXT,
                embedding_model TEXT, embedding_base_url TEXT, embedding_api_key TEXT
            )"""
        )
        if row is not None:
            columns = ", ".join(row)
            raw.execute(
                f"INSERT INTO llm_config (id, {columns}) VALUES (1, {', '.join('?' for _ in row)})",
                tuple(row.values()),
            )


@pytest.fixture
def default_environment(monkeypatch, tmp_path) -> None:
    """An installation whose model variables are unset and that has no configuration file."""
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    for name in (
        "MEMFORGE_ENRICHMENT_MODEL",
        "MEMFORGE_ENRICHMENT_BASE_URL",
        "MEMFORGE_ENRICHMENT_API_KEY",
        "MEMFORGE_EMBEDDING_MODEL",
        "MEMFORGE_EMBEDDING_BASE_URL",
        "MEMFORGE_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


async def _opened_and_translated(path: Path, config: AppConfig) -> Database:
    """Open a database the way a process does: connect, then translate with its configuration."""
    database = Database(str(path))
    await database.connect()
    await database.translate_llm_config(config.llm)
    return database


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as raw:
        return {row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _database_bytes(path: Path) -> bytes:
    return b"".join(file.read_bytes() for file in path.parent.glob(f"{path.name}*"))


async def test_stored_settings_with_plaintext_keys_become_active_bindings_with_encrypted_keys(
    tmp_path, default_environment
) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {
            "enrichment_model": "claude-sonnet-4",
            "enrichment_base_url": "http://localhost:6655/anthropic",
            "enrichment_api_key": GENERATION_KEY,
            "embedding_model": "BAAI/bge-m3",
            "embedding_base_url": "https://embed.example.test/v1",
            "embedding_api_key": EMBEDDING_KEY,
        },
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    database = await _opened_and_translated(path, config)
    try:
        generation = await resolve_active_binding(database, config, "generation")
        embedding = await resolve_active_binding(database, config, "embedding")
        models = await resolve_models(database, config)
        candidates = [await database.get_model_binding(kind, "candidate") for kind in ("generation", "embedding")]
    finally:
        await database.close()

    assert (generation.provider, generation.route, generation.endpoint) == (
        "anthropic",
        "anthropic/claude-sonnet-4",
        "http://localhost:6655/anthropic",
    )
    assert (embedding.provider, embedding.route, embedding.endpoint) == (
        "openai_compatible",
        "openai/BAAI/bge-m3",
        "https://embed.example.test/v1",
    )
    assert {generation.state, embedding.state} == {"not_validated"}
    assert {generation.config_source, embedding.config_source} == {"settings"}
    assert generation.validation is None and embedding.validation is None
    assert candidates == [None, None]
    assert models.generation.api_key == GENERATION_KEY
    assert models.embedding.api_key == EMBEDDING_KEY
    assert GENERATION_KEY.encode() not in _database_bytes(path)
    assert EMBEDDING_KEY.encode() not in _database_bytes(path)
    with sqlite3.connect(path) as raw:
        origins = {row[0] for row in raw.execute("SELECT origin FROM model_secrets")}
    assert "llm_config" not in _tables(path)
    assert origins == {"http://localhost:6655", "https://embed.example.test"}


async def test_an_empty_stored_row_leaves_both_kinds_to_the_environment(tmp_path, default_environment) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(path, {"enrichment_model": None})

    database = await _opened_and_translated(path, AppConfig(base_dir=tmp_path / "mem"))
    try:
        stored = [
            await database.get_model_binding(kind, slot)
            for kind in ("generation", "embedding")
            for slot in ("active", "candidate")
        ]
    finally:
        await database.close()

    assert stored == [None, None, None, None]
    with sqlite3.connect(path) as raw:
        secrets = raw.execute("SELECT COUNT(*) FROM model_secrets").fetchone()
    assert "llm_config" not in _tables(path)
    assert secrets == (0,)


async def test_a_database_without_a_stored_row_translates_to_no_bindings(tmp_path, default_environment) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(path, None)

    database = await _opened_and_translated(path, AppConfig(base_dir=tmp_path / "mem"))
    try:
        assert await database.get_model_binding("generation", "active") is None
        assert await database.get_model_binding("embedding", "active") is None
    finally:
        await database.close()

    assert "llm_config" not in _tables(path)


async def test_a_stored_column_left_empty_takes_its_environment_value(tmp_path, default_environment, monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_API_KEY", GENERATION_KEY)
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", "https://embed.environment.example.test/v1")
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"enrichment_model": "claude-opus-4", "embedding_api_key": EMBEDDING_KEY},
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    database = await _opened_and_translated(path, config)
    try:
        generation = await resolve_active_binding(database, config, "generation")
        embedding = await resolve_active_binding(database, config, "embedding")
        generation_key = await resolve_credential(database, config, generation)
    finally:
        await database.close()

    assert generation.route == "anthropic/claude-opus-4"
    assert generation.endpoint == "https://api.anthropic.com"
    assert generation.secret_ref == "env:MEMFORGE_ENRICHMENT_API_KEY"
    assert generation_key == GENERATION_KEY
    assert embedding.route == "openai/text-embedding-3-small"
    assert embedding.endpoint == "https://embed.environment.example.test/v1"
    assert embedding.secret_ref.startswith("secret:")


async def test_empty_columns_take_the_values_of_the_configuration_the_process_runs_with(
    tmp_path, default_environment
) -> None:
    """The settings of a configuration file given to the process complete the row, not the defaults."""
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {
            "enrichment_model": "claude-opus-ui",
            "enrichment_base_url": "http://proxy.internal:6655/anthropic",
            "embedding_model": "text-embedding-3-large",
        },
    )
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_model = "claude-from-the-file"
    config.llm.enrichment_api_key = GENERATION_KEY
    config.llm.embedding_api_key = EMBEDDING_KEY

    database = await _opened_and_translated(path, config)
    try:
        models = await resolve_models(database, config)
    finally:
        await database.close()

    assert (models.generation.route, models.generation.api_base, models.generation.api_key) == (
        "anthropic/claude-opus-ui",
        "http://proxy.internal:6655/anthropic",
        GENERATION_KEY,
    )
    assert (models.embedding.route, models.embedding.api_base, models.embedding.api_key) == (
        "openai/text-embedding-3-large",
        "https://api.openai.com/v1",
        EMBEDDING_KEY,
    )


async def test_a_stored_model_without_any_key_stays_the_model_and_waits_for_the_environment_key(
    tmp_path, default_environment, caplog
) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"enrichment_model": "claude-opus-ui", "enrichment_base_url": "http://proxy.internal:6655/anthropic"},
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    database = await _opened_and_translated(path, config)
    try:
        generation = await resolve_active_binding(database, config, "generation")
        without_key = await resolve_models(database, config)
        config.llm.enrichment_api_key = GENERATION_KEY
        with_key = await resolve_models(database, config)
    finally:
        await database.close()

    assert (generation.route, generation.endpoint, generation.secret_ref) == (
        "anthropic/claude-opus-ui",
        "http://proxy.internal:6655/anthropic",
        "env:MEMFORGE_ENRICHMENT_API_KEY",
    )
    assert without_key.generation is None
    assert (with_key.generation.route, with_key.generation.api_base, with_key.generation.api_key) == (
        "anthropic/claude-opus-ui",
        "http://proxy.internal:6655/anthropic",
        GENERATION_KEY,
    )


async def test_a_stored_keyless_embedding_endpoint_becomes_a_binding_with_no_auth(
    tmp_path, default_environment
) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"embedding_model": "local-embedding", "embedding_base_url": "http://localhost:11434/v1"},
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    database = await _opened_and_translated(path, config)
    try:
        embedding = await resolve_active_binding(database, config, "embedding")
    finally:
        await database.close()

    assert (embedding.provider, embedding.auth_strategy, embedding.secret_ref) == ("openai_compatible", "none", None)


async def test_stored_settings_no_binding_accepts_are_dropped_with_a_warning(
    tmp_path, default_environment, caplog
) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {
            "enrichment_model": "claude-sonnet-4",
            "enrichment_api_key": GENERATION_KEY,
            "embedding_model": "local-embedding",
            "embedding_base_url": "not-a-url",
            "embedding_api_key": EMBEDDING_KEY,
        },
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    with caplog.at_level("WARNING", logger="memforge.llm.bindings"):
        database = await _opened_and_translated(path, config)
    try:
        generation = await database.get_model_binding("generation", "active")
        embedding = await database.get_model_binding("embedding", "active")
    finally:
        await database.close()

    assert generation["route"] == "anthropic/claude-sonnet-4"
    assert embedding is None
    assert "stored embedding model settings cannot be kept" in caplog.text
    assert "endpoint_invalid" in caplog.text
    assert EMBEDDING_KEY not in caplog.text
    assert EMBEDDING_KEY.encode() not in _database_bytes(path)
    assert "llm_config" not in _tables(path)


async def test_a_key_that_cannot_be_encrypted_stops_the_translation_and_keeps_the_stored_settings(
    tmp_path, default_environment, monkeypatch
) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"enrichment_model": "claude-sonnet-4", "enrichment_api_key": GENERATION_KEY},
    )
    config = AppConfig(base_dir=tmp_path / "mem")

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", "not-a-usable-secret-key")
    database = Database(str(path))
    await database.connect()
    try:
        with pytest.raises(ModelConfigError) as refused:
            await database.translate_llm_config(config.llm)
        kept = _tables(path)
        monkeypatch.setenv("MEMFORGE_SECRET_KEY", SECRET_KEY)
        translated = await database.translate_llm_config(config.llm)
        models = await resolve_models(database, config)
    finally:
        await database.close()

    assert refused.value.code == "secret_storage_unavailable"
    assert "llm_config" in kept
    assert translated == 1
    assert models.generation.api_key == GENERATION_KEY
    assert "llm_config" not in _tables(path)


async def test_no_model_binding_is_read_before_the_stored_settings_are_translated(
    tmp_path, default_environment
) -> None:
    """A database opened without the translation step never falls back to the environment's model."""
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(path, {"enrichment_model": "claude-opus-ui"})
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = GENERATION_KEY

    database = Database(str(path))
    await database.connect()
    try:
        with pytest.raises(RuntimeError, match="translate_llm_config"):
            await resolve_models(database, config)
        assert await database.translate_llm_config(config.llm) == 1
        assert await database.translate_llm_config(config.llm) == 0
        models = await resolve_models(database, config)
    finally:
        await database.close()

    assert models.generation.route == "anthropic/claude-opus-ui"


async def test_translation_by_another_process_leaves_nothing_to_translate(tmp_path, default_environment) -> None:
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(path, {"enrichment_model": "claude-opus-ui"})
    config = AppConfig(base_dir=tmp_path / "mem")
    config.llm.enrichment_api_key = GENERATION_KEY

    second = Database(str(path))
    await second.connect()
    first = await _opened_and_translated(path, config)
    try:
        assert await second.translate_llm_config(config.llm) == 0
        binding = await second.get_model_binding("generation", "active")
    finally:
        await first.close()
        await second.close()

    assert binding["route"] == "anthropic/claude-opus-ui"


async def test_plaintext_keys_leave_the_write_ahead_log_once_no_other_connection_reads(
    tmp_path, default_environment, caplog, monkeypatch
) -> None:
    monkeypatch.setattr("memforge.storage.database._BUSY_TIMEOUT_MS", 50)
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"enrichment_model": "claude-sonnet-4", "enrichment_api_key": GENERATION_KEY},
    )
    config = AppConfig(base_dir=tmp_path / "mem")
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()

    with caplog.at_level("WARNING", logger="memforge.storage.database"):
        database = await _opened_and_translated(path, config)
    try:
        held = GENERATION_KEY.encode() in _database_bytes(path)
        still_reading = await database.erase_plaintext_keys()
        reader.execute("COMMIT")
        reader.close()
        erased = await database.erase_plaintext_keys()
        left = GENERATION_KEY.encode() in _database_bytes(path)
        binding = await database.get_model_binding("generation", "active")
    finally:
        await database.close()

    assert held is True
    assert "write-ahead log still holds the model API keys" in caplog.text
    assert (still_reading, erased, left) == (False, True, False)
    assert binding["route"] == "anthropic/claude-sonnet-4"


async def test_closing_the_database_erases_plaintext_keys_a_reader_held_back(
    tmp_path, default_environment, monkeypatch
) -> None:
    monkeypatch.setattr("memforge.storage.database._BUSY_TIMEOUT_MS", 50)
    path = tmp_path / "earlier.db"
    await _database_with_llm_config(
        path,
        {"enrichment_model": "claude-sonnet-4", "enrichment_api_key": GENERATION_KEY},
    )
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()

    database = await _opened_and_translated(path, AppConfig(base_dir=tmp_path / "mem"))
    reader.execute("COMMIT")
    await database.close()
    reader.close()

    assert GENERATION_KEY.encode() not in _database_bytes(path)


async def test_a_new_database_stores_model_settings_as_bindings_and_secrets_only(db) -> None:
    with sqlite3.connect(db.db_path) as raw:
        tables = {row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert {"model_bindings", "model_secrets"} <= tables
    assert "llm_config" not in tables
