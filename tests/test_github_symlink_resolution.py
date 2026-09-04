from __future__ import annotations

import hashlib

import pytest

from memforge.github_repo_utils import GitHubTreeEntryResolver


def _sha(label: str) -> str:
    return hashlib.sha1(label.encode(), usedforsecurity=False).hexdigest()


def _entry(path: str, *, mode: str = "100644", entry_type: str = "blob") -> dict:
    return {"path": path, "mode": mode, "type": entry_type, "sha": _sha(path), "size": len(path)}


def _resolve(entry: dict, entries: list[dict], targets: dict[str, bytes], *, excluded=()):
    resolver = GitHubTreeEntryResolver(
        entry,
        entries_by_path={item["path"]: item for item in entries},
        exclude_paths=list(excluded),
    )
    while request := resolver.next_symlink_read():
        resolver.accept_symlink_target(targets[request.path])
    return resolver.result()


def test_relative_symlink_resolves_to_regular_blob_without_changing_logical_identity() -> None:
    link = _entry("README.md", mode="120000")
    target = _entry("docs/CONTRIBUTIONS.md")

    resolved = _resolve(link, [link, target], {"README.md": b"docs/CONTRIBUTIONS.md"})

    assert resolved.logical_path == "README.md"
    assert resolved.content_entry == target
    assert resolved.resolved_relative_path == "docs/CONTRIBUTIONS.md"
    assert resolved.symlink_chain == ({
        "path": "README.md",
        "blob_sha": link["sha"],
        "target_path": "docs/CONTRIBUTIONS.md",
    },)


def test_parent_relative_chain_can_leave_a_directory_but_not_the_repository() -> None:
    first = _entry("docs/start.md", mode="120000")
    second = _entry("shared/current", mode="120000")
    target = _entry("content/final.md")

    resolved = _resolve(
        first,
        [first, second, target],
        {
            "docs/start.md": b"../shared/current",
            "shared/current": b"../content/final.md",
        },
    )

    assert resolved.resolved_relative_path == "content/final.md"
    assert len(resolved.symlink_chain) == 2


@pytest.mark.parametrize(
    ("raw_target", "message"),
    [
        (b"/etc/passwd", "absolute target"),
        (b"../../outside.md", "escapes the repository"),
        (b"docs\\target.md", "invalid target"),
        (b"docs/target.md\n", "invalid target"),
        (b"", "invalid target"),
        (b" target.md", "invalid target"),
        (b"target.md ", "invalid target"),
    ],
)
def test_unsafe_target_text_fails_closed(raw_target: bytes, message: str) -> None:
    link = _entry("README.md", mode="120000")
    resolver = GitHubTreeEntryResolver(
        link,
        entries_by_path={"README.md": link},
        exclude_paths=[],
    )
    assert resolver.next_symlink_read() is not None

    with pytest.raises(ValueError, match=message):
        resolver.accept_symlink_target(raw_target)


def test_missing_and_explicitly_excluded_targets_fail_closed() -> None:
    link = _entry("README.md", mode="120000")
    target = _entry("private/target.md")

    with pytest.raises(ValueError, match="missing target"):
        _resolve(link, [link], {"README.md": b"missing.md"})
    with pytest.raises(ValueError, match="explicitly excluded"):
        _resolve(
            link,
            [link, target],
            {"README.md": b"private/target.md"},
            excluded=["private"],
        )


def test_cycle_and_overlong_acyclic_chain_fail_closed() -> None:
    first = _entry("a.md", mode="120000")
    second = _entry("b.md", mode="120000")
    with pytest.raises(ValueError, match="cycle"):
        _resolve(first, [first, second], {"a.md": b"b.md", "b.md": b"a.md"})

    links = [_entry(f"link-{index}.md", mode="120000") for index in range(41)]
    target = _entry("target.md")
    targets = {
        link["path"]: (
            links[index + 1]["path"].encode() if index + 1 < len(links) else target["path"].encode()
        )
        for index, link in enumerate(links)
    }
    with pytest.raises(ValueError, match="exceeds 40 resolution hops"):
        _resolve(links[0], [*links, target], targets)


@pytest.mark.parametrize(
    "target",
    [
        _entry("directory", mode="040000", entry_type="tree"),
        _entry("submodule", mode="160000", entry_type="commit"),
        _entry("unknown.md", mode="100600", entry_type="blob"),
    ],
)
def test_non_regular_targets_fail_closed(target: dict) -> None:
    link = _entry("README.md", mode="120000")
    with pytest.raises(ValueError, match="unsupported target"):
        _resolve(link, [link, target], {"README.md": target["path"].encode()})


def test_symlink_mode_must_be_represented_by_a_blob() -> None:
    link = _entry("README.md", mode="120000", entry_type="tree")
    resolver = GitHubTreeEntryResolver(
        link,
        entries_by_path={"README.md": link},
        exclude_paths=[],
    )
    with pytest.raises(ValueError, match="not represented by a blob"):
        resolver.next_symlink_read()
