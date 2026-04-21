# Mechanistic Claim Registry

Every mechanistic scientific claim in `paper/woundsim.tex` that asserts a
mechanism, reproduction, or architecture equivalence is registered here with a
primary-source numeric target and a falsification test in `tests/` that runs
the claim under the artifact's default initialization.

Each entry records: paper location, code location, verbatim claim text,
primary source and numeric target, and the path to the falsification test
that enforces it.

---

## Claim macrophage_zlobina: Zlobina-type macrophage polarization (5 ODEs)

- **Location (paper):** paper/woundsim.tex abstract + §3 (Mathematical Models)
- **Location (code):** `src/woundsim/models/zlobina.py`, `src/woundsim/envs/macrophage.py`
- **Claim text:**
  > a Zlobina-type macrophage polarization model with 5 state variables
- **Primary source:** Zlobina KE, et al., *Mathematical modeling of macrophage
  polarization dynamics in wound healing*, 2022.
- **Primary-source target:** 5 state variables (M1, M2, bacteria, ECM
  constituents), with M1→M2 transition kinetics consistent with Zlobina 2022
  Section 2.
- **Falsification test:** `tests/test_srm_macrophage_zlobina.py` asserts
  env.observation_space.shape == (5,) and that the ODE RHS produces M1→M2
  polarization under anti-inflammatory input.

## Claim ischemic_xue_friedman: Xue-Friedman ischemic wound ODE (6 states)

- **Location (paper):** paper/woundsim.tex abstract + §3 (Mathematical Models)
- **Location (code):** `src/woundsim/models/xue_friedman.py`,
  `src/woundsim/envs/ischemic.py`
- **Claim text:**
  > a simplified Xue--Friedman ischemic wound healing model with 6 variables
- **Primary source:** Xue C, Friedman A, Sen CK, *A mathematical model of
  ischemic cutaneous wounds*, PNAS 2009.
- **Primary-source target:** 6 state variables (oxygen, VEGF, inflammatory
  cells, fibroblasts, capillary density, ECM); capillary density rises with
  VEGF above an oxygen-dependent threshold.
- **Falsification test:** `tests/test_srm_ischemic_xue_friedman.py` asserts
  6-variable state, and that capillary density responds monotonically to VEGF
  at fixed oxygen.

## Claim hbot_flegg: Flegg HBOT angiogenesis model (4 states)

- **Location (paper):** paper/woundsim.tex abstract + §3 (Mathematical Models)
- **Location (code):** `src/woundsim/models/flegg.py`, `src/woundsim/envs/hbot.py`
- **Claim text:**
  > a Flegg hyperbaric oxygen therapy (HBOT) angiogenesis model with 4 variables
- **Primary source:** Flegg JA, McElwain DLS, Byrne HM, Turner IW, *A
  three-species model to simulate application of hyperbaric oxygen therapy
  to chronic wounds*, PLoS Comput Biol 2009; Flegg et al. 2010 extension.
- **Primary-source target:** 4 state variables; non-monotonic relationship
  between oxygen and angiogenesis — angiogenesis is suppressed when oxygen
  rises above a threshold (paper §6.1: *O_thresh = 40 mmHg*).
- **Falsification test:** `tests/test_srm_hbot_flegg.py` asserts 4-variable
  state, and that angiogenesis rate is NOT monotonically increasing in oxygen
  under default parameters.

## Claim diabetic_extended: Extended diabetic wound model (7 states, glucose-insulin coupled)

- **Location (paper):** paper/woundsim.tex abstract + §3 (Mathematical Models)
- **Location (code):** `src/woundsim/envs/diabetic.py`
- **Claim text:**
  > an extended diabetic wound model coupling macrophage dynamics with
  > glucose--insulin physiology across 7 variables
- **Primary source:** extension of Zlobina 2022 macrophage dynamics coupled
  to standard glucose–insulin pharmacokinetics (Bergman minimal model lineage).
- **Primary-source target:** 7 state variables; insulin administration lowers
  glucose; glucose ≥ threshold suppresses M1→M2 polarization rate.
- **Falsification test:** `tests/test_srm_diabetic_extended.py` asserts
  7-variable state, insulin→glucose monotone decrease, and glucose-dependent
  polarization penalty.

## Claim param_provenance: All model parameters sourced from peer-reviewed publications

- **Location (paper):** paper/woundsim.tex abstract:
  > All model parameters are sourced from peer-reviewed publications with
  > explicit provenance.
- **Location (code):** inline `# SOURCE:` comments in each
  `src/woundsim/models/*.py` module.
- **Falsification test:** `tests/test_integrity_audit.py::test_all_parameters_have_source_comments`
  scans model files and asserts every non-derived parameter carries a
  `# SOURCE:` comment referring to a bib key defined in
  `paper/references.bib`.
