from memforge.server.admin_api import _github_repo_tree_items


def test_github_repo_tree_items_synthesizes_folders_and_files():
    items = _github_repo_tree_items(
        [
            {"path": "Payroll Processing/README.md", "type": "blob", "size": 42},
            {"path": "Payroll Processing/V2/Migration.md", "type": "blob", "size": 84},
            {"path": "Flexible Payroll", "type": "tree"},
            {"path": "Flexible Payroll/Overview.md", "type": "blob", "size": 11},
        ],
        limit=20,
        include_extensions={"md"},
    )

    assert [(item.type, item.path, item.size) for item in items] == [
        ("tree", "Flexible Payroll", None),
        ("tree", "Payroll Processing", None),
        ("tree", "Payroll Processing/V2", None),
        ("blob", "Flexible Payroll/Overview.md", 11),
        ("blob", "Payroll Processing/README.md", 42),
        ("blob", "Payroll Processing/V2/Migration.md", 84),
    ]


def test_github_repo_tree_items_returns_limit_plus_one_for_truncation_detection():
    items = _github_repo_tree_items(
        [
            {"path": "a/one.md", "type": "blob"},
            {"path": "b/two.md", "type": "blob"},
            {"path": "c/three.md", "type": "blob"},
        ],
        limit=2,
        include_extensions={"md"},
    )

    assert len(items) == 3


def test_github_repo_tree_items_omit_unsupported_files_and_empty_folders():
    items = _github_repo_tree_items(
        [
            {"path": "docs/README.md", "type": "blob"},
            {"path": "assets/logo.png", "type": "blob"},
            {"path": "assets", "type": "tree"},
        ],
        limit=20,
        include_extensions={"md"},
    )

    assert [(item.type, item.path) for item in items] == [
        ("tree", "docs"),
        ("blob", "docs/README.md"),
    ]
