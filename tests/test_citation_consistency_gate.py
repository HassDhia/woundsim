"""Gate test — citation-consistency verifier (clean pass + self-test)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_citation_consistency.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_verifier_passes_on_clean_repo():
    r = _run()
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "FAILURES: 0" in r.stdout


def test_verifier_self_test_catches_injected_defects():
    r = _run("--self-test")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "MISSED" not in r.stdout
    assert "SELF-TEST:" in r.stdout
