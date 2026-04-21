# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
