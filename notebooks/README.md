# Notebook index

Run `python tools/initialize_inputs.py` from the repository root before using
the notebooks. Original notebook cells and saved outputs are retained. Runtime
artifacts are generated locally and excluded from this source release.

The notebooks are ordered by dataset and experimental phase.

## WISDM

- `FedVocabHAR_WISDM_Phase0.ipynb` through
  `FedVocabHAR_WISDM_Phase0D.ipynb`: initial feasibility studies.
- `FedVocabHAR_WISDM_Phase1A.ipynb`,
  `FedVocabHAR_WISDM_Phase1A_R.ipynb`, and
  `FedVocabHAR_WISDM_Phase1A_S.ipynb`: registry/query-policy studies.
- `FedVocabHAR_WISDM_Phase1B.ipynb` and
  `FedVocabHAR_WISDM_Phase1C.ipynb`: integration studies.
- `FedVocabHAR_WISDM_Phase2A.ipynb` through
  `FedVocabHAR_WISDM_Phase2E.ipynb`: encoder, fusion, prototype, and episodic
  representation selection.

## RealWorld HAR

- `FedVocabHAR_RealWorld_Phase3A_FIXED.ipynb`: data and split validation.
- `FedVocabHAR_RealWorld_Phase3B.ipynb`: confirmatory representation test.
- `FedVocabHAR_RealWorld_Phase3C_FIXED.ipynb`: repaired exploratory mechanism
  comparison.
- `FedVocabHAR_RealWorld_Phase3D.ipynb`: selection-only policy lock.

## PAMAP2

- `FedVocabHAR_PAMAP2_Phase4A_FIXED3.ipynb`: data-quality validation.
- `FedVocabHAR_PAMAP2_Phase4B_FIXED.ipynb`: exploratory encoder analysis.
- `FedVocabHAR_PAMAP2_Phase4C.ipynb`: frozen ranked-MNN external analysis.

Earlier superseded `FIXED` variants are intentionally omitted. The included
notebooks are the versions associated with the frozen artifact package.

## Phase 5 (baselines, sensitivity, UCI-HAR confirmatory)

- `FedVocabHAR_Phase5A_Baselines.ipynb`: preregistered baselines
  (random-query, support-only, novelty-top-B, component-mixture,
  centralized-oracle) on the frozen RealWorld and PAMAP2 encoders. Every
  oracle call is counted against the budget.
- `FedVocabHAR_Phase5B_Sensitivity.ipynb`: one-at-a-time and grid sensitivity
  of ranked-MNN, support-only, and random-query over the locked thresholds and
  budget on frozen encoders (descriptive; no parameter selected).
- `FedVocabHAR_Phase5C_A_Data.ipynb`: UCI-HAR download, SHA-256, cache build,
  and gate G1 only.
- `FedVocabHAR_Phase5C_B_Encoders.ipynb`: per-fold frozen-recipe encoders on
  UCI-HAR training subjects only (final-epoch checkpoints; no selection).
- `FedVocabHAR_Phase5C_C_Confirmatory.ipynb`: ranked-MNN and all baselines on
  integration clients, evaluated once on test subjects; gates G2-G6, H1-H4.
- `FedVocabHAR_Phase5D_Cost.ipynb`: uplink bytes, encoder MACs/latency, client and
  server compute of the locked policy on all three datasets (no labels read).
- `src/phase5_common.py`: shared machinery (encoder, calibration, clustering,
  policies, evaluation) imported by the Phase 5B and 5C notebooks.
- `src/`: percent-format source of each Phase 5 notebook; `tools/py2ipynb.ps1`
  regenerates the `.ipynb` from it.
- The Phase 5C UCI-HAR preregistration lives in `artifacts/phase5c/` and was
  retained as an experiment input under `config/`; setup supplies its runtime path.

## Phase 6 (MotionSense second confirmatory dataset)

Run in this order: 6-A, then 6-B, then 6-C.

- `FedVocabHAR_Phase6_A_Data.ipynb`: MotionSense `A_DeviceMotion_data.zip`
  download, SHA-256, cache build (pocket `userAcceleration` XYZ + magnitude,
  g -> m/s^2 scale applied at cache-build time per the manifest), and gate G1
  only.
- `FedVocabHAR_Phase6_B_Encoders.ipynb`: per-fold frozen-recipe encoders on
  MotionSense training subjects only (final-epoch checkpoints, declared in
  advance; identical recipe to Phase 5C).
- `FedVocabHAR_Phase6_C_Confirmatory.ipynb`: ranked-MNN and all baselines on
  integration clients, evaluated once on test subjects; gates G2-G6, H1-H4
  (H1 threshold >= 5/6 folds).
- `src/phase5_common.py` gained a `'motionsense'` entry in `DATASETS`; the
  same shared machinery used by Phase 5B/5C is reused unchanged.
- The Phase 6 MotionSense preregistration lives in `artifacts/phase6/` and
  was committed (`f7ffaba`) before the dataset was downloaded. Its
  `relationship_to_phase5c` field records that this run exists solely to
  test whether the Phase 5C result generalizes to a second dataset; no
  Phase 5A-5D or Phase 5C number may influence any Phase 6 setting.

## Corrected metrics

`src/phase5_metrics_corrected.py` provides queried-label purity. The original
`src/phase5_common.py` retains the historical cluster-homogeneity statistic.
Run `python tools/test_metric_correction.py` from the repository root. Optional
checkpoint replay needs locally generated artifacts and uses `tools/revision_audit.py`.
