from __future__ import annotations

import stat

import pytest

from memforge.config import AppConfig
from memforge.source_secrets import (
    SecretConfigurationError,
    decrypt_source_config_for_runtime,
    prepare_source_config_for_storage,
    redact_source_config,
)

TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
OTHER_SOURCE_KEY = "GFdRS9_z07biLN73Vrh9gBEl7nHhsp2zaLDSbYiaKSM="


def _config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "stable-test-secret"
    return cfg


def test_source_pat_is_encrypted_redacted_and_recovered(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch)

    stored = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
    )

    assert "pat" not in stored
    assert stored["pat_encrypted"] != "pat-secret"
    assert stored["pat_configured"] is True
    assert "pat_hint" not in stored

    redacted = redact_source_config(stored)
    assert "pat" not in redacted
    assert "pat_encrypted" not in redacted
    assert redacted["pat_configured"] is True
    assert "pat_hint" not in redacted

    runtime = decrypt_source_config_for_runtime(stored)
    assert runtime["pat"] == "pat-secret"
    assert "pat_encrypted" not in runtime


def test_redaction_marks_undecryptable_secret_for_reentry(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch)
    stored = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
    )

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", OTHER_SOURCE_KEY)
    redacted = redact_source_config(stored, validate_encryption=True)

    assert "pat" not in redacted
    assert "pat_encrypted" not in redacted
    assert redacted["pat_configured"] is False
    assert redacted["pat_decrypt_failed"] is True
    assert "pat_hint" not in redacted


def test_blank_pat_update_preserves_existing_encrypted_secret(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch)
    existing = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "old-secret"},
    )

    updated = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test/wiki", "pat": ""},
        existing_config=existing,
    )

    assert updated["base_url"] == "https://wiki.example.test/wiki"
    assert updated["pat_encrypted"] == existing["pat_encrypted"]
    assert decrypt_source_config_for_runtime(updated)["pat"] == "old-secret"


def test_same_pat_update_preserves_existing_encrypted_secret(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch)
    existing = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "same-secret"},
    )

    updated = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "same-secret"},
        existing_config=existing,
    )

    assert updated["pat_encrypted"] == existing["pat_encrypted"]
    assert decrypt_source_config_for_runtime(updated)["pat"] == "same-secret"


def test_blank_pat_update_migrates_existing_plaintext_secret(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch)
    existing = {"base_url": "https://wiki.example.test", "pat": "legacy-secret"}

    updated = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test/wiki", "pat": ""},
        existing_config=existing,
    )

    assert "pat" not in updated
    assert updated["pat_encrypted"] != "legacy-secret"
    assert decrypt_source_config_for_runtime(updated)["pat"] == "legacy-secret"


def test_missing_env_key_generates_local_source_secret_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMFORGE_SECRET_KEY", raising=False)
    monkeypatch.delenv("MEMFORGE_BASE_DIR", raising=False)
    monkeypatch.delenv("MEMFORGE_SECRET_KEY_FILE", raising=False)
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))

    stored = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
    )

    key_path = tmp_path / "mem" / "secrets" / "source-secrets.key"
    assert key_path.is_file()
    assert stat.S_IMODE(key_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert decrypt_source_config_for_runtime(stored)["pat"] == "pat-secret"


def test_decrypt_without_existing_local_key_does_not_generate_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    stored = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
    )

    monkeypatch.delenv("MEMFORGE_SECRET_KEY", raising=False)
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))

    with pytest.raises(SecretConfigurationError, match="restore"):
        decrypt_source_config_for_runtime(stored)

    assert not (tmp_path / "mem" / "secrets" / "source-secrets.key").exists()


def test_jwt_secret_is_not_used_as_source_secret_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMFORGE_SECRET_KEY", raising=False)
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "stable-test-secret"

    stored = prepare_source_config_for_storage(
        {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
    )

    assert decrypt_source_config_for_runtime(stored)["pat"] == "pat-secret"
    assert (tmp_path / "mem" / "secrets" / "source-secrets.key").read_text(
        encoding="ascii"
    ).strip() != cfg.server.jwt_secret


def test_weak_source_secret_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", "weak-passphrase")

    with pytest.raises(SecretConfigurationError, match="Fernet key"):
        prepare_source_config_for_storage(
            {"base_url": "https://wiki.example.test", "pat": "pat-secret"},
        )
