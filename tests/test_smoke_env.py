from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_env.py"


def _module():
    spec = importlib.util.spec_from_file_location("smoke_env", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke_env = _module()


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env.test"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_file_sets_named_values_and_skips_comments_and_empty_variables(tmp_path) -> None:
    path = _file(
        tmp_path,
        "# Provider keys\n"
        "\n"
        "OPENROUTER_API_KEY=sk-or-plain\n"
        "export OPENAI_API_KEY='sk-quoted'\n"
        'ANTHROPIC_API_KEY = "sk-spaced"\n'
        "VOYAGE_API_KEY=\n",
    )

    assert smoke_env.read_smoke_env(path) == {
        "OPENROUTER_API_KEY": "sk-or-plain",
        "OPENAI_API_KEY": "sk-quoted",
        "ANTHROPIC_API_KEY": "sk-spaced",
    }


def test_a_variable_set_in_the_environment_wins_over_the_file(tmp_path) -> None:
    path = _file(tmp_path, "OPENROUTER_API_KEY=from-file\nOPENAI_API_KEY=from-file\nANTHROPIC_API_KEY=from-file\n")
    environ = {"OPENROUTER_API_KEY": "from-shell", "OPENAI_API_KEY": ""}

    loaded = smoke_env.load_smoke_env(environ, path)

    assert environ == {
        "OPENROUTER_API_KEY": "from-shell",
        "OPENAI_API_KEY": "from-file",
        "ANTHROPIC_API_KEY": "from-file",
    }
    assert loaded == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def test_without_the_file_nothing_is_loaded(tmp_path) -> None:
    environ: dict[str, str] = {}

    assert smoke_env.load_smoke_env(environ, tmp_path / ".env.test") == []
    assert environ == {}


def test_a_line_that_is_not_an_assignment_is_reported_with_its_number(tmp_path) -> None:
    path = _file(tmp_path, "OPENROUTER_API_KEY=ok\nnot an assignment\n")

    with pytest.raises(ValueError, match=r"\.env\.test:2: expected NAME=value"):
        smoke_env.read_smoke_env(path)


def test_the_shell_loads_the_file_without_expanding_or_running_anything(tmp_path) -> None:
    path = _file(tmp_path, "OPENROUTER_API_KEY=sk-$(touch pwned) 'quoted' $HOME\nOPENAI_API_KEY=from-file\n")
    environ = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    environ["OPENAI_API_KEY"] = "from-shell"

    shown = subprocess.run(
        [
            "bash",
            "-c",
            'eval "$("$0" "$1" --file "$2")"; printf "%s\\n%s" "$OPENROUTER_API_KEY" "$OPENAI_API_KEY"',
            sys.executable,
            str(SCRIPT),
            str(path),
        ],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert shown == "sk-$(touch pwned) 'quoted' $HOME\nfrom-shell"
    assert not (tmp_path / "pwned").exists()


def test_the_template_names_every_key_the_smoke_scripts_read() -> None:
    template = SCRIPT.parents[1] / ".env.test.example"

    assert set(smoke_env.read_smoke_env(template)) == set()
    names = {line.partition("=")[0] for line in template.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#")}
    assert names == {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
