"""The command line shows the models in use, never their keys."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from click.testing import CliRunner

from memforge.config import load_config
from memforge.llm.bindings import binding_record, build_binding, new_stored_secret
from memforge.main import cli
from memforge.storage.database import Database
from tests.test_model_bindings import _database_with_llm_config

ENVIRONMENT_KEY = "environment-secret-key-3333"
SAVED_KEY = "saved-secret-key-1111"


@pytest.fixture(autouse=True)
def installation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30=")
    monkeypatch.setenv("COLUMNS", "200")
    for name in ("MEMFORGE_ENRICHMENT_MODEL", "MEMFORGE_ENRICHMENT_BASE_URL", "MEMFORGE_ENRICHMENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "")


def _show() -> str:
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0, result.output
    return result.output


def test_config_show_reports_a_first_run_as_not_configured(tmp_path) -> None:
    output = _show()

    assert "Generation model" in output and "NOT CONFIGURED" in output
    assert "Embedding model" in output
    assert not (tmp_path / "mem" / "db").exists()


def test_config_show_describes_the_model_the_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_MODEL", "claude-sonnet-4")
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_API_KEY", ENVIRONMENT_KEY)

    output = _show()

    assert "claude-sonnet-4 (Anthropic)" in output
    assert "https://api.anthropic.com" in output
    assert "API key from the environment" in output
    assert "not validated, from environment" in output
    assert ENVIRONMENT_KEY not in output
    assert ENVIRONMENT_KEY[-4:] not in output


def test_config_show_describes_the_model_saved_in_settings(monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_API_KEY", ENVIRONMENT_KEY)

    async def save() -> None:
        database = Database(load_config().storage.db_path)
        await database.connect()
        binding = build_binding(
            kind="generation", provider="openai", model_id="gpt-5-mini", config_source="settings"
        )
        secret_ref, secret = new_stored_secret(SAVED_KEY, origin=binding.origin)
        binding = build_binding(
            kind="generation",
            provider="openai",
            model_id="gpt-5-mini",
            secret_ref=secret_ref,
            config_source="settings",
        )
        await database.put_model_binding("generation", "active", binding_record(binding), secret=secret)
        await database.close()

    asyncio.run(save())

    output = _show()

    assert "gpt-5-mini (OpenAI)" in output
    assert "https://api.openai.com/v1" in output
    assert "API key saved in Settings" in output
    assert "not validated, from settings" in output
    assert "claude" not in output
    assert SAVED_KEY not in output


def test_a_command_given_a_configuration_file_completes_earlier_settings_from_that_file(tmp_path) -> None:
    """Settings stored before model bindings take their missing values from --config, not from the default file."""
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '[llm]\nenrichment_model = "claude-from-the-file"\n'
        f'enrichment_api_key = "{ENVIRONMENT_KEY}"\nembedding_api_key = "{ENVIRONMENT_KEY}"\n'
    )
    config = load_config(config_path=custom)
    asyncio.run(
        _database_with_llm_config(
            Path(config.storage.db_path),
            {
                "enrichment_model": "claude-opus-ui",
                "enrichment_base_url": "http://proxy.internal:6655/anthropic",
                "embedding_model": "text-embedding-3-large",
                "embedding_base_url": "https://api.openai.com/v1",
            },
        )
    )

    result = CliRunner().invoke(cli, ["--config", str(custom), "config", "show"])

    assert result.exit_code == 0, result.output
    assert "claude-opus-ui (Anthropic)" in result.output
    assert "http://proxy.internal:6655/anthropic" in result.output
    assert "text-embedding-3-large (OpenAI)" in result.output
    assert "claude-from-the-file" not in result.output
    assert result.output.count("API key from the environment") == 2
    assert ENVIRONMENT_KEY not in result.output
