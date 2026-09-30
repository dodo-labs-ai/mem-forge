#!/usr/bin/env python3
"""Write the Admin API OpenAPI document that the admin UI generates its types from."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app


# Routes that serve GET and HEAD share one operation id in FastAPI's document.
# HEAD carries no body the admin UI reads, so it is left out.
OMITTED_METHODS = frozenset({"head"})


def export_openapi() -> str:
    """Return the Admin API OpenAPI document as stable, sorted JSON."""
    app = create_admin_app(config=AppConfig())
    document = app.openapi()
    for operations in document["paths"].values():
        for method in OMITTED_METHODS & operations.keys():
            del operations[method]
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path of the JSON file to write.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the file differs from the current document.",
    )
    args = parser.parse_args(argv)

    document = export_openapi()
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != document:
            print(f"{args.output} is stale; run `npm run gen:api` in admin/.", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
