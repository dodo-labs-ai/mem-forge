import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_openapi.py"
HASH_SEEDS = ("1", "2", "3")


def _export(tmp_path: Path, seed: str) -> str:
    output = tmp_path / f"admin-{seed}.json"
    subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), str(output)],
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
    )
    return output.read_text()


def test_openapi_document_is_identical_across_hash_seeds(tmp_path):
    documents = {_export(tmp_path, seed) for seed in HASH_SEEDS}

    assert len(documents) == 1
