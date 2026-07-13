import json

import pytest

from memforge.github_repo_utils import (
    github_exclude_paths,
    github_include_paths,
    github_path_in_scope,
    normalize_github_scope_paths,
)
from memforge.storage.database import Database
from memforge.server.admin_api import _validate_github_repo_config


def test_repository_exclusions_override_the_base_scope() -> None:
    config = {
        "include_paths": ["docs"],
        "exclude_paths": ["docs/archived", "docs/drafts/obsolete.md"],
    }

    included = github_include_paths(config)
    excluded = github_exclude_paths(config)

    assert github_path_in_scope("docs/current/overview.md", included, excluded) is True
    assert github_path_in_scope("docs/archived/old.md", included, excluded) is False
    assert github_path_in_scope("docs/drafts/obsolete.md", included, excluded) is False
    assert github_path_in_scope("src/main.py", included, excluded) is False


def test_whole_repository_scope_still_honors_exclusions() -> None:
    assert (
        github_path_in_scope(
            "docs/current.md",
            include_paths=[],
            exclude_paths=["docs/archived"],
        )
        is True
    )
    assert (
        github_path_in_scope(
            "docs/archived/old.md",
            include_paths=[],
            exclude_paths=["docs/archived"],
        )
        is False
    )


def test_scope_paths_are_canonical_and_collapse_descendants() -> None:
    assert normalize_github_scope_paths(
        [
            "/docs/current/",
            "docs",
            "docs/current/overview.md",
            "src/main.py",
            "src/main.py",
        ]
    ) == ["docs", "src/main.py"]


def test_source_config_rejects_local_clone_paths() -> None:
    with pytest.raises(ValueError, match="remote repository"):
        _validate_github_repo_config(
            {
                "repo_url": "https://github.example/org/repo",
                "connection_mode": "local_push",
                "repo_path": "/tmp/repo",
            }
        )


@pytest.mark.asyncio
async def test_migration_removes_obsolete_repo_path_from_stored_sources(tmp_path) -> None:
    database = Database(str(tmp_path / "github-scope.db"))
    await database.connect()
    try:
        await database.db.execute(
            """INSERT INTO sources
               (id, type, name, config, access_policy, access_state, owner_user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "github-source",
                "github_repo",
                "Architecture",
                json.dumps({"repo_url": "https://github.example/org/repo", "repo_path": "/tmp/repo"}),
                "private",
                "active",
                "dev",
            ),
        )
        await database.db.execute("DELETE FROM schema_migrations WHERE version = 41")
        await database.db.commit()

        await database._run_migrations()

        row = await (
            await database.db.execute("SELECT config FROM sources WHERE id = ?", ("github-source",))
        ).fetchone()
        assert json.loads(row["config"]) == {"repo_url": "https://github.example/org/repo"}
    finally:
        await database.close()
