# FedVocabHAR

Source code for agreement-triggered federated vocabulary promotion in wearable
human activity recognition.

## Contents

- `notebooks/`: original Google Colab notebooks, including saved outputs.
- `notebooks/src/`: Python implementations and shared model/evaluation modules.
- `notebooks/tools/`: Python-to-notebook conversion helper.
- `config/`: experiment splits, encoder recipe, and promotion-policy inputs.
- `tools/`: setup, metric-correction checks, and optional checkpoint replay.

Manuscripts, LaTeX, figures, submission documents, result tables, datasets, and
checkpoints are excluded. Configuration files contain the small inputs required
by the code, not trained models or result dumps.

## Setup

```sh
python -m pip install -r requirements.txt
python tools/initialize_inputs.py
```

The initializer supplies the untracked runtime configuration and `colab/src`
import path expected by the original notebooks. Existing files are preserved.
It does not train models or generate experiment results.

For Google Colab, place this checkout at
`/content/drive/MyDrive/FedVocabHAR`, run the initializer there, and open the
notebooks from `notebooks/`. They mount Google Drive and use that project path.
Complete earlier data/training phases before running dependent evaluations.
Obtain official datasets using the data notebooks. Data, caches, checkpoints,
and generated results remain local and are ignored by Git.

## Execution order

1. WISDM Phases 0-2: feasibility and representation development.
2. RealWorld 3A, 3B, 3C FIXED, 3D: data, representation, association, policy.
3. PAMAP2 4A FIXED3, 4B FIXED, 4C: external-data experiments.
4. Phase 5A/5B: baseline and sensitivity experiments.
5. UCI-HAR Phase 5C A, B, C: data, encoder training, evaluation.
6. Phase 5D: payload and compute measurements.
7. MotionSense Phase 6 A, B, C: data, encoder training, evaluation.

See `notebooks/README.md` for notebook names. GPU acceleration is recommended
for training. Optional `tools/revision_audit.py` requires generated checkpoints
and saved evaluation payloads from the relevant phases; those files are not
distributed in this source-only repository.

## Metric implementation

`phase5_common.py` preserves the original implementation. Its historical
`entry_purity` measures local cluster homogeneity, not queried-label purity.
`phase5_metrics_corrected.py` computes the queried-label fraction and weights
promoted entries by support. Association and query-ranking rules are unchanged.

```sh
python tools/test_metric_correction.py
```

## Repository history

This source release has one squashed commit. Historical protocol dates and
comments retained in source/configuration are experiment records; this Git
history does not establish their original registration timestamps.

## License

MIT. Dataset and third-party materials retain their respective licenses.
