#!/usr/bin/env python3
"""Write the source type form fixture the admin UI's setup tests read.

The fixture holds the Admin API's real answers for ``GET /api/v1/genes`` and
``GET /api/v1/genes/{type}/config-schema`` for every registered type, so the
admin UI's renderer tests draw what the server declares rather than a
hand-written copy. ``npm run gen:api`` in ``admin/`` rewrites it with the API
types after a Gene's ``config_schema()`` or descriptor changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app


def export_source_type_forms() -> str:
    """Return ``{"genes": [...], "schemas": {type: schema}}`` as stable JSON."""
    client = TestClient(create_admin_app(config=AppConfig()))
    genes = _get(client, "/api/v1/genes")
    schemas = {gene["name"]: _get(client, f"/api/v1/genes/{gene['name']}/config-schema") for gene in genes}
    return json.dumps({"genes": genes, "schemas": schemas}, indent=2, sort_keys=True) + "\n"


def _get(client: TestClient, path: str):
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Path of the JSON file to write.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the file differs from the current answers.",
    )
    args = parser.parse_args(argv)

    fixture = export_source_type_forms()
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != fixture:
            print(f"{args.output} is stale; run `npm run gen:api` in admin/.", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
