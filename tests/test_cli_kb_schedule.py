"""The server owns local source configuration; the daemon only executes jobs."""

from __future__ import annotations

from click.testing import CliRunner

from memforge.main import cli


def test_legacy_profile_cli_groups_are_removed():
    runner = CliRunner()

    kb_result = runner.invoke(cli, ["adapter", "kb"])
    github_result = runner.invoke(cli, ["adapter", "github"])
    help_result = runner.invoke(cli, ["adapter", "--help"])

    assert kb_result.exit_code != 0
    assert "No such command 'kb'" in kb_result.output
    assert github_result.exit_code != 0
    assert "No such command 'github'" in github_result.output
    assert help_result.exit_code == 0, help_result.output
    assert "Manage local repository adapter profiles" not in help_result.output
    assert "Manage local GitHub repository adapter profiles" not in help_result.output


def test_daemon_help_is_the_only_local_profile_execution_surface():
    result = CliRunner().invoke(cli, ["adapter", "daemon", "--help"])

    assert result.exit_code == 0, result.output
    assert "once" in result.output
    assert "run" in result.output
