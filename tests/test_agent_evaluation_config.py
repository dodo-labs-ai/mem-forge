from memforge.config import AgentEvaluationConfig, AppConfig, load_config


def test_agent_runtime_event_retention_defaults_are_bounded() -> None:
    config = AppConfig()

    assert config.agent_evaluation == AgentEvaluationConfig(
        runtime_event_retention_days=90,
        runtime_event_purge_batch_size=1000,
    )


def test_agent_runtime_event_retention_loads_toml_then_env(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[agent_evaluation]
runtime_event_retention_days = 45
runtime_event_purge_batch_size = 250
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_AGENT_RUNTIME_EVENT_RETENTION_DAYS", "30")
    monkeypatch.setenv("MEMFORGE_AGENT_RUNTIME_EVENT_PURGE_BATCH_SIZE", "20000")

    config = load_config(config_path=config_path, base_dir=tmp_path / "memforge")

    assert config.agent_evaluation.runtime_event_retention_days == 30
    assert config.agent_evaluation.runtime_event_purge_batch_size == 10_000
