"""Command line helper for retrieval golden case-set hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from memforge.evals.retrieval.schema import (
    compute_case_set_sha,
    load_case_set,
    validate_case_set_sha,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", default="retrieval-core-v1")
    parser.add_argument("--case-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    case_set = load_case_set(args.case_set, case_root=args.case_root, verify_sha=False)
    computed_sha = compute_case_set_sha(case_set)

    if args.check:
        if not validate_case_set_sha(case_set):
            print(
                f"{args.case_set}: mismatch expected={case_set.manifest.case_set_sha} "
                f"computed={computed_sha}"
            )
            return 1
        print(f"{args.case_set}: OK {computed_sha}")
        return 0

    if args.case_root is None:
        raise SystemExit("--write requires --case-root")

    manifest_path = args.case_root / "manifest.yaml"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("case_set_sha: "):
            lines[index] = f"case_set_sha: {computed_sha}"
            break
    else:
        raise SystemExit(f"case_set_sha missing from {manifest_path}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.case_set}: wrote {computed_sha} to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
