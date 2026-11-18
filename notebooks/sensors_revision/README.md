# FedVocabHAR Sensors revision source update

This update accompanies the FedVocabHAR Sensors v4.16 study. Extract the
archive at the root of the existing GitHub checkout. All added files are
under notebooks/sensors_revision. The historical experiments and their
setup remain in the repository's existing notebooks and tools.

## Study code

| Directory | Contents |
|---|---|
| d4 | Matched development baseline runner and typicality reference |
| m1 | Frozen MHEALTH transfer runner and protocol |
| c1 | Aggregate-centroid selector, development runner and synthetic tests |
| r2 | Frozen DSADS transfer across two acquisition cohorts |
| exploratory | LoGo adaptation, UCI-HAR high-pass filtering and sequential DSADS notebooks |

The four runner directories preserve their study-specific source snapshots.
Install the existing root requirements and use Linux or Colab for the
runners, which use Unix resource controls. Each runner takes the project
root, a fresh output directory, and its execution contract as arguments.
For M1 and R2, launch from the individual study directory so the
relative protocol and provenance paths in the execution contract resolve.

| Study | Runner | Contract |
|---|---|---|
| D4 | d4_baseline_compare.py | D4_INPUT_CONTRACT.json |
| M1 | m1_mhealth_external.py | M1_EXECUTION_CONTRACT.json |
| C1 | c1_development_run.py | C1_INPUT_CONTRACT.json |
| R2 | r2_dsads_external.py | R2_EXECUTION_CONTRACT.json |

The contracts identify the expected prepared inputs. The original runners
retain their input-hash and runtime checks. Prepared windows, input caches,
embeddings, checkpoints, predictions and run results are excluded. The
notebooks assume the project is mounted at /content/drive/MyDrive/FedVocabHAR.
Dataset access follows the original providers' terms. The experimental
results require the recorded prepared inputs and runtime.

The LoGo notebook is an exploratory restricted-pool adaptation. Its
query counts are descriptive under that setup. The filtering experiment
tests preprocessing sensitivity under a supervised encoder. The sequential
DSADS run reports name recovery separately from recognition F1.
