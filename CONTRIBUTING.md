# Contributing to WoundSim

Thank you for your interest in contributing to WoundSim.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/smarttechinvest/woundsim.git
   cd woundsim
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all]"
   ```

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

## Adding a New Environment

1. Create the ODE model in `src/woundsim/models/`
2. Create the Gymnasium environment in `src/woundsim/envs/`
3. Register the environment in `src/woundsim/envs/__init__.py`
4. Add heuristic baseline in `src/woundsim/agents/heuristic.py`
5. Add training config in `src/woundsim/training/configs.py`
6. Write tests covering the model, environment, and integration

## Parameter Sources

All ODE parameters must include inline `# SOURCE:` comments citing the original publication.
Use BibTeX keys from `paper/references.bib`.

## Code Quality

- Run `ruff check src/ tests/` before submitting
- Ensure all tests pass with `pytest tests/ -v`
- Maintain 100+ test count

## Pull Requests

- One environment or feature per PR
- Include tests for new functionality
- Update CHANGELOG.md
