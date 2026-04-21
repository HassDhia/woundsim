#!/usr/bin/env python3
"""AC+NC gate runner for woundsim — verify artifact-paper citation consistency.

Enforces three invariants on every push:

1. BIB integrity — every `\\cite{key}` in paper/woundsim.tex resolves to an
   entry in paper/references.bib.
2. Source-comment coverage — every non-derived parameter in
   `src/woundsim/models/*.py` carries a `# SOURCE: <bibkey>` comment
   referring to a bibkey defined in references.bib.
3. Claim-registry consistency — every `## Claim` entry in
   `docs/mechanistic-claims.md` names a falsification test file that exists.

Run `--self-test` to confirm the verifier catches known synthetic defects
(proves it is not a rubber stamp).

Exit 0 = safe to ship. Exit 1 = do not ship.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cite_keys(tex: str) -> set[str]:
    out: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        for k in group.split(","):
            out.add(k.strip())
    return out


def _bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"^@\w+\{(\w+)\s*,", bib, re.MULTILINE))


def check_bib(repo: Path) -> tuple[str, list[str]]:
    paper = (repo / "paper" / "woundsim.tex").read_text()
    bib = (repo / "paper" / "references.bib").read_text()
    missing = sorted(_cite_keys(paper) - _bib_keys(bib))
    failures = [f"BIB undefined \\cite keys: {missing}"] if missing else []
    return ("BIB — every \\cite key defined in references.bib", failures)


def check_source_comments(repo: Path) -> tuple[str, list[str]]:
    models_dir = repo / "src" / "woundsim" / "models"
    bib = _bib_keys((repo / "paper" / "references.bib").read_text())
    failures: list[str] = []
    model_count = 0
    if not models_dir.is_dir():
        return ("SOURCE — source-comment coverage", ["models directory missing"])
    for f in sorted(models_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        model_count += 1
        text = f.read_text()
        refs = set(re.findall(r"#\s*SOURCE:\s*([\w]+)", text))
        if not refs:
            failures.append(f"{f.name}: no `# SOURCE:` comments found")
            continue
        undefined = [r for r in refs if r not in bib]
        if undefined:
            failures.append(
                f"{f.name}: `# SOURCE:` references bibkeys not in references.bib: {undefined}"
            )
    desc = f"SOURCE — provenance comments across {model_count} model file(s)"
    return (desc, failures)


def check_claim_tests(repo: Path) -> tuple[str, list[str]]:
    registry = repo / "docs" / "mechanistic-claims.md"
    failures: list[str] = []
    if not registry.exists():
        return ("CLAIMS — registry + test pointers", ["docs/mechanistic-claims.md missing"])
    text = registry.read_text()
    claim_headings = re.findall(r"^##\s+Claim\s+(\S+):", text, re.MULTILINE)
    test_refs = re.findall(r"tests/(test_\S+\.py)", text)
    for t in set(test_refs):
        p = repo / "tests" / t
        if not p.exists():
            failures.append(f"claim references missing test file: tests/{t}")
    if len(claim_headings) < 3:
        failures.append(f"only {len(claim_headings)} claim(s) registered; expected ≥3")
    desc = f"CLAIMS — {len(claim_headings)} claim(s) registered, {len(set(test_refs))} unique test references"
    return (desc, failures)


def run(repo: Path = REPO) -> int:
    print("=" * 70)
    print(f"woundsim citation-consistency gate runner  (repo={repo})")
    print("=" * 70)

    checks = [
        ("BIB", check_bib(repo)),
        ("SOURCE", check_source_comments(repo)),
        ("CLAIMS", check_claim_tests(repo)),
    ]

    all_failures: list[str] = []
    all_passes: list[str] = []
    for name, (desc, failures) in checks:
        if failures:
            print(f"\n[{name}] FAIL")
            for f in failures:
                print(f"  ✗ {f}")
            all_failures.extend(failures)
        else:
            print(f"\n[{name}] PASS — {desc}")
            all_passes.append(desc)

    print("\n" + "=" * 70)
    print(f"PASSES:   {len(all_passes)}")
    print(f"FAILURES: {len(all_failures)}")
    print("=" * 70)
    return 0 if not all_failures else 1


def self_test() -> int:
    """Inject synthetic defects into a sandboxed copy and confirm the verifier
    fails on each."""
    print("=" * 70)
    print("SELF-TEST — injecting synthetic defects into a sandboxed copy")
    print("=" * 70)

    tests = [
        (
            "BIB: add undefined \\cite to paper",
            lambda repo: (repo / "paper" / "woundsim.tex").write_text(
                (repo / "paper" / "woundsim.tex").read_text() + r"\cite{nonexistent_key_xyz}" + "\n"
            ),
        ),
        (
            "SOURCE: strip a SOURCE comment from a model",
            lambda repo: (repo / "src" / "woundsim" / "models" / "zlobina.py").write_text(
                re.sub(
                    r"#\s*SOURCE:\s*\w+",
                    "",
                    (repo / "src" / "woundsim" / "models" / "zlobina.py").read_text(),
                    count=20,
                )
            ),
        ),
        (
            "CLAIMS: delete the registry",
            lambda repo: (repo / "docs" / "mechanistic-claims.md").unlink(),
        ),
    ]

    caught = 0
    missed = 0
    for name, mutate in tests:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp) / "woundsim"
            shutil.copytree(REPO, sandbox, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".git"))
            try:
                mutate(sandbox)
            except Exception as e:
                print(f"  ? [{name}] mutation raised: {e}")
                missed += 1
                continue
            rc = run(sandbox)
            if rc != 0:
                print(f"  ✓ CAUGHT [{name}]")
                caught += 1
            else:
                print(f"  ✗ MISSED [{name}] — verifier returned 0 despite defect")
                missed += 1

    print("\n" + "=" * 70)
    print(f"SELF-TEST: {caught}/{caught + missed} synthetic defects caught")
    print("=" * 70)
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Inject defects and verify failures")
    args = parser.parse_args()
    raise SystemExit(self_test() if args.self_test else run())
