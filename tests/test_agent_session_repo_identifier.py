from __future__ import annotations

from memforge.repo_identity import normalize_repo_identifier


def test_normalize_repo_identifier_prefers_canonical_remote_slug():
    assert (
        normalize_repo_identifier("git@github.tools.sap:HCM/memforge-cloud.git")
        == "github.tools.sap/hcm/memforge-cloud"
    )
    assert normalize_repo_identifier("https://github.com/shno-labs/mem-forge.git") == "github.com/shno-labs/mem-forge"
    assert normalize_repo_identifier("HTTPS://GitHub.com/Shno-Labs/Mem-Forge.GIT?ref=main") == (
        "github.com/shno-labs/mem-forge"
    )
    assert normalize_repo_identifier("mem-inception") == "mem-inception"
    assert normalize_repo_identifier(None) is None
