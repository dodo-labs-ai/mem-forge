"""Load the provider keys a contributor keeps in ``.env.test`` for the smoke scripts.

``scripts/provider_smoke.py`` and ``scripts/compose_smoke.sh`` read API keys
from the environment. A contributor who runs them often writes the keys once
into ``.env.test`` at the repository root (copied from ``.env.test.example``;
git ignores it), and both scripts load it at start. A variable already set in
the environment wins, so a key exported for one run overrides the file.

The file holds ``NAME=value`` lines. Blank lines, ``#`` comments, a leading
``export`` and quotes around the value are allowed; a variable left empty is
not loaded. Nothing is expanded or executed.

This module runs on the system Python ``compose_smoke.sh`` uses, so it keeps
to the standard library and Python 3.9.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence

SMOKE_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.test"

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def read_smoke_env(path: Path = SMOKE_ENV_FILE) -> Dict[str, str]:
    """Return the variables ``path`` sets, or nothing when it does not exist."""
    if not path.is_file():
        return {}
    values: Dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            raise ValueError(f"{path}:{number}: expected NAME=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            values[name] = value
    return values


def load_smoke_env(
    environ: Optional[MutableMapping[str, str]] = None,
    path: Path = SMOKE_ENV_FILE,
) -> List[str]:
    """Set each variable of ``path`` the environment leaves unset or empty; return their names."""
    target = os.environ if environ is None else environ
    loaded: List[str] = []
    for name, value in read_smoke_env(path).items():
        if not target.get(name):
            target[name] = value
            loaded.append(name)
    return loaded


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print ``export`` lines for the shell to evaluate: ``eval "$(python3 smoke_env.py)"``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=SMOKE_ENV_FILE, help="the file to load (default: .env.test)")
    args = parser.parse_args(argv)
    try:
        values = read_smoke_env(args.file)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for name, value in values.items():
        if not os.environ.get(name):
            print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
