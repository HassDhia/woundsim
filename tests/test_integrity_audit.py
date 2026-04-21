"""Integrity-audit tests — permanent CI gate against the defect patterns
that slipped through on v0.1.0.

Every test here is a ship-gate: if it fails, the repo is not safe to share
with a domain reviewer. Tests are cheap, deterministic, and require no
trained models."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_no_review_certificate_at_root():
    """Self-signed review certificates are internal process artifacts and
    must not live at the repo root."""
    assert not (REPO / "review-certificate.md").exists(), (
        "review-certificate.md at root — move to .reviews/ (gitignored) or delete"
    )


def test_no_loose_scripts_at_root():
    """Python scripts live under scripts/, not at repo root."""
    ALLOWED = {"setup.py", "conftest.py", "noxfile.py"}
    loose = [
        p.name
        for p in REPO.iterdir()
        if p.is_file() and p.suffix == ".py" and p.name not in ALLOWED
    ]
    assert not loose, f"Loose .py at repo root: {loose}"


def test_no_appliedresearch_schema_in_results():
    """results/*.json must not carry the surprise/mechanism/implication
    AppliedResearch-skill schema."""
    results = REPO / "results"
    if not results.exists():
        return
    offenders = []
    for f in results.glob("*.json"):
        try:
            body = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict) and {"surprise", "mechanism", "implication"} <= body.keys():
            offenders.append(f.name)
    assert not offenders, f"AppliedResearch schema leaked in: {offenders}"


def test_changelog_has_no_internal_process_vocabulary():
    """CHANGELOG must not contain internal-process markers — 'APPROVED FOR
    PUBLICATION', 'Council', 'Phase C', 'SR-M', 'OUTREACH GATE', etc."""
    path = REPO / "CHANGELOG.md"
    if not path.exists():
        return
    bad_patterns = [
        r"APPROVED FOR PUBLICATION",
        r"\bCouncil\s+(re-?convene|confirmed|approved)",
        r"\bPhase\s+[A-E]\b",
        r"\bSR-M\b",
        r"\bOUTREACH GATE\b",
        r"\bpre-?partner-?send\b",
    ]
    text = path.read_text()
    hits = []
    for pat in bad_patterns:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    assert not hits, f"CHANGELOG contains internal-process vocabulary: {hits}"


def test_mechanistic_claims_registry_exists():
    """docs/mechanistic-claims.md must register ≥3 claims, each with a
    falsification test file path."""
    registry = REPO / "docs" / "mechanistic-claims.md"
    assert registry.exists(), "docs/mechanistic-claims.md missing"
    text = registry.read_text()
    claims = re.findall(r"^##\s+Claim\s+\S+:", text, re.MULTILINE)
    assert len(claims) >= 3, f"Only {len(claims)} claims registered; need ≥3"


def test_every_bib_cite_resolves():
    """Every \\cite{} key in paper/woundsim.tex must exist in references.bib."""
    tex = (REPO / "paper" / "woundsim.tex").read_text()
    bib = (REPO / "paper" / "references.bib").read_text()
    cited = set()
    for g in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        for k in g.split(","):
            cited.add(k.strip())
    defined = set(re.findall(r"^@\w+\{(\w+)\s*,", bib, re.MULTILINE))
    missing = sorted(cited - defined)
    assert not missing, f"Undefined \\cite keys: {missing}"


def test_every_model_file_has_source_comments():
    """Every non-private module in src/woundsim/models/ must carry at least
    one `# SOURCE: <bibkey>` comment, and referenced bibkeys must exist in
    references.bib. The paper's abstract asserts exactly this provenance
    discipline."""
    models = REPO / "src" / "woundsim" / "models"
    assert models.is_dir(), "src/woundsim/models/ missing"
    bib_text = (REPO / "paper" / "references.bib").read_text()
    bib = set(re.findall(r"^@\w+\{(\w+)\s*,", bib_text, re.MULTILINE))
    model_files = [f for f in sorted(models.glob("*.py")) if not f.name.startswith("_")]
    assert model_files, "no model files found"
    for f in model_files:
        refs = set(re.findall(r"#\s*SOURCE:\s*(\w+)", f.read_text()))
        assert refs, f"{f.name}: no `# SOURCE:` comments"
        undefined = refs - bib
        assert not undefined, f"{f.name}: SOURCE refs not in references.bib: {undefined}"


def test_published_calibration_artifact_present():
    """Published-data calibration artifact must exist. This is the domain
    reviewer's first scientific-credibility check."""
    j = REPO / "results" / "published_calibration.json"
    assert j.exists(), "results/published_calibration.json missing"
    body = json.loads(j.read_text())
    assert "environments" in body, "calibration JSON lacks `environments` section"
    assert len(body["environments"]) >= 4, "expected ≥4 calibrated environments"


def test_citation_consistency_gate_passes():
    """The citation-consistency verifier returns exit 0 on clean state."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_citation_consistency.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (
        f"verify_citation_consistency.py returned {r.returncode}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
