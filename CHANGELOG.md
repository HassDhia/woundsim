# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-21

### Added
- Mechanistic-claim registry at `docs/mechanistic-claims.md`. Every mechanism claim in the paper is registered with a primary-source numeric target (state dimensions, Flegg `O_thresh = 40 mmHg`, Zlobina polarization kinetics, etc.) and a falsification test.
- Citation-consistency gate at `scripts/verify_citation_consistency.py` with `--self-test`. Enforces (1) every `\cite{}` in the paper resolves to a `references.bib` entry, (2) every non-derived parameter in `src/woundsim/models/*.py` carries a `# SOURCE: <bibkey>` comment whose bibkey exists in `references.bib`, (3) every registered claim names a falsification test file that exists.
- Integrity-audit test module at `tests/test_integrity_audit.py` — permanent CI gate covering repo hygiene, bib coverage, provenance comments, CHANGELOG tone, claim registry, and calibration-artifact presence.
- Published-data calibration harness at `scripts/run_published_calibration.py`. Validates each environment against its primary-source state dimensions and key threshold parameters. Emits `results/published_calibration.json`.

### Changed
- Moved loose top-level scripts (`train_all.py`, `generate_figures.py`) under `scripts/`.
- `.gitignore` now excludes `.reviews/`, `feedback/`, and `.research-project.json` so internal process artifacts cannot be committed.

### Removed
- `review-certificate.md` at repo root (internal review artifact; not a collaborator-facing document).
- `results/discovery.json` (unreferenced internal workflow output).

## [0.1.0] - 2026-04-21

### Added
- Four Gymnasium-compatible wound healing environments
  - WoundMacrophage-v0: Zlobina macrophage polarization model (5 ODEs)
  - WoundIschemic-v0: Simplified Xue-Friedman ischemic wound model (6 ODEs)
  - WoundHBOT-v0: Flegg HBOT angiogenesis model (4 ODEs)
  - WoundDiabetic-v0: Extended diabetic wound model (7 ODEs)
- ODE-based wound healing models with literature-sourced parameters
- SB3-compatible environment wrappers
- Random, heuristic, and PPO agent implementations
- Training infrastructure with configurable hyperparameters
- Benchmark runner for systematic evaluation
- Publication-quality figure generation
- Comprehensive test suite (100+ tests)
