from __future__ import annotations

from memforge.repo_identity import normalize_repo_identifier


def test_normalize_repo_identifier_prefers_canonical_remote_slug():
    assert (
        normalize_repo_identifier("git@github.example.com:Acme/memforge-cloud.git")
        == "github.example.com/acme/memforge-cloud"
    )
    assert (
        normalize_repo_identifier("https://github.com/dodo-labs-ai/mem-forge.git")
        == "github.com/dodo-labs-ai/mem-forge"
    )
    assert normalize_repo_identifier("HTTPS://GitHub.com/Dodo-Labs-AI/Mem-Forge.GIT?ref=main") == (
        "github.com/dodo-labs-ai/mem-forge"
    )
    assert normalize_repo_identifier("mem-inception") == "mem-inception"
    assert normalize_repo_identifier(None) is None
