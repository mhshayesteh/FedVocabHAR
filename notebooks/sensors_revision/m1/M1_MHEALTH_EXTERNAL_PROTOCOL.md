# M1: external MHEALTH transfer evaluation specification

16 November 2026. User reported no prior MHEALTH use in answer to the question covering the user and coauthors, training, evaluation and inspected results in any project. This is a recorded team-history statement, not an independently verified access log. Prior literature discussion is preserved. No raw MHEALTH data has been downloaded or scored for this revision.

## Scope and current status

Specify a fresh, subject-separated external evaluation of the fixed agreement-versus-support interventions and low-budget controls, conditional on a supplied RealWorld-trained encoder. No MHEALTH representation training, checkpoint selection, threshold fitting or known-class prototype adaptation is permitted. This is a cross-dataset transfer study, not a native federated-learning system or a replication of D4's exact eight-class task.

This document locks scientific choices before target outcomes. Implementation and generated conformance checks are the next step. It is not a completed external validation, an executable notebook, a registered protocol at an independent registry, or evidence that the paper's novelty/comparator gates have passed. If implementing these choices reveals an ambiguity or infeasible operation, preserve it and version any necessary specification change before opening target outcomes. Do not silently fill in method choices during a run.

## Data and declared roles

Use the official UCI MHEALTH dataset, DOI 10.24432/C5TW22, https://archive.ics.uci.edu/dataset/319/mhealth+dataset . Official metadata identifies ten subject logs, 50 Hz sampling, acceleration in m/s² and right-lower-arm acceleration columns 15–17 (one-based); setup text also calls the location right wrist. Keep that placement difference from RealWorld forearm explicit. License listed by UCI: CC BY 4.0. Consult the saved POPULATION_FEASIBILITY_SOURCES.json; no new claim about raw-file structure has been validated by downloading it.

Subjects 1–5 are acquisition clients. Subjects 6–10 are evaluation subjects. These numerical assignments are selected before data access, without demographic/performance selection. There is one fixed acquisition episode, not five independent acquisition replications or a subject-rotation search. All target subjects are excluded from source fitting, source calibration and prototype construction. All five evaluation subjects and every mapped activity remain scheduled, including poor or zero-recovery outcomes.

Acquisition may inspect only acquisition-client embeddings and identifiers. Evaluation embeddings are used solely by the fixed decoder after all query plans are committed; they cannot influence clustering, query selection or thresholds. Do not reverse roles after results or introduce another split as fresh confirmation.

## Complete activity mapping

Retain all twelve non-null activities; do not choose four favorable novel classes to imitate the older task. Map integer labels as follows, with fixed known-class order walking, running, sitting, standing:

| Raw label | Canonical label | Role |
|---:|---|---|
| 1 | standing | Known |
| 2 | sitting | Known |
| 3 | lying | Novel |
| 4 | walking | Known |
| 5 | climbing_stairs | Novel |
| 6 | waist_bends_forward | Novel |
| 7 | frontal_elevation_arms | Novel |
| 8 | knees_bending | Novel |
| 9 | cycling | Novel |
| 10 | jogging | Novel |
| 11 | running | Known |
| 12 | jump_front_back | Novel |

Raw label 0 is null/unscored. Unknown label codes or missing required subject logs are structural failures, not opportunities to change the vocabulary or drop subjects. Do not merge jogging with running, generic climbing_stairs with a direction-specific class, or jump_front_back with an old source label. Eight novel classes and B=4 impose a maximum of four newly named classes per arm in this episode. Unnamed classes remain in the F1 denominator; this task difficulty is not a reason to increase B after results.

## Fixed input construction and access boundary

Keep original row order within each subject log. On its 50 Hz grid use starts 0,64,128,... and length128. Retain a window only if all 128 labels equal one nonzero mapped code. Windows crossing null/activity boundaries are excluded by a data-curation component, never passed to a label-aware selector. Do not concatenate activity segments or subjects before windowing, resample, interpolate, remove gravity, rotate axes or normalize using target statistics. This is a curated homogeneous-window task, not unrestricted continuous-stream discovery.

Read XYZ columns15–17, cast XYZ to float32, append float32 Euclidean magnitude computed from those XYZ values, yielding N×4×128. Required selected signals must be finite; malformed rows/shape/label codes stop the run with preserved error. No outlier deletion or class balancing. Keep an immutable stable window ID equal to subject_id×1,000,000,000 + original start row (zero-based), validating uniqueness and bounds. A lookup table maps IDs to subject, start row and original file hash.

Dataset curation necessarily accesses labels to define this evaluation population. Retain its rules and excluded-window counts, but hide label vectors, per-class counts, error plots and performance from selectors and analyst progress until predictions are sealed. Query plans receive only features/IDs/client ownership. The counted oracle answers only purchased IDs; the evaluator receives labels only after all 26 arms' prefix predictions are saved and sealed. This API/process separation is not a cryptographic security guarantee.

## Source-only transfer bundle

Use the existing RealWorld fold1 centralized anchor, artifacts/phase3b/folds/fold_1_models.pt. Fold1 is the fixed numerical reference, chosen without MHEALTH results; do not choose a source fold by target performance. Its recorded checkpoint SHA256 is 824772ae3b1ac09266379d7d6d59cc77fd1cd536a4dcdafeaf13745cb9686bbe. The companion JSON pins source code, source data/manifest, Stage B reference and exact cached embedding bytes.

In Colab, before downloading MHEALTH, reconstruct source-only calibration/prototypes using stage_c_pilot.load_case for RealWorld fold1 and the pinned Stage B embeddings. Save and seal both the calibration prototype matrix returned by that function and the separate recognition known-prototype matrix returned by P.known_prototypes. These use the historical source training/probe recipes, not MHEALTH labels. Their numerical distinction must be preserved. Require exact agreement with the saved source calibration: novelty_threshold 0.638611912727356, local_distance 0.2090341925621032, agreement_similarity 0.8902801990509033.

Load the pinned anchor state and normalization statistics in Colab only, using the existing strict state-key adaptation and encoder architecture. Freeze weights and batch-normalization behavior; model.eval(), no_grad(), CPU, batches512, source input_scale=1.0. Apply the historical Frozen.normalized_input operation ordering and final float32 cast to the four target channels, then the existing encode/unit-normalization function. No target statistic, optimizer, target-label probe, source-fold retry or target-fitted classifier. Seal the target embedding cache and its source/data/runtime identity. Neural inference is required once on MHEALTH in Colab; D4's zero-encoder-inference description does not apply to M1.

## Fixed methods and budgets

There are 26 arms: O_all, O_agreed, H_all, H_agreed, TC_Frozen_Sensor_v1, twenty uniform-random-window seeds20261117..20261136, and a known_only zero-query decoder reference. The four intervention arms, independent TC numerical contract and purchased-window-only decoder retain the pinned D3/D4 code. Do not describe the TC reference as verified native TypiClust or as equal in server-feature access to signature-only agreement.

For each acquisition client select up to600 windows ranked by the unchanged reference_acquisition.construct novelty order, relative to source calibration prototypes. Preserve the pinned NumPy ordering/tie behavior rather than introducing target-dependent changes. The common selected pool is at most3000 windows. Original local cutoff is 0.2090341925621032; H uses exactly half. Both use cosine average linkage, local support minimum12, existing MNN agreement threshold, support/client/window-ID medoid ranking, and no fallback for empty agreement eligibility. TC uses its fixed KMeans/Euclidean typicality convention over the same selected pool, K=min(4,N), eligible cluster size≥5 and KNN≤20, seed42. Random samples uniformly without replacement from sorted selected IDs, independently per seed.

Each non-reference arm has a separate B=4 simulation: known and duplicate-novel answers cost one call; only the first purchased occurrence of a new name adds its purchased-window embedding as a prototype. No known-prototype update, cluster-centroid prototype, majority-label name, hidden novel-count input, Hungarian name mapping or unpaid naming. Up to100 simulated oracle calls total; prefix replay reuses ledgers without additional charges. The known_only arm costs zero. All query plans, including TC diagnostics and random IDs, must seal before any answer; no cross-arm sharing of paid answers as model inputs.

## Endpoints and analysis fixed before outcomes

Primary endpoint: for each arm, mean novel macro-F1 across evaluation subjects6–10 at that arm's actual full-cap endpoint. Each subject score uses the fixed eight-class novel denominator, including unrecognized or absent classes as zero; retain per-class support so this convention is visible. Random first averages its 20 fixed seeds within each subject, then subjects. Report the pooled-window novel F1 separately as a sensitivity analysis, never substitute it after seeing which looks better.

Primary intervention contrasts are O_agreed−O_all, H_agreed−H_all, and the difference between those two contrasts. Estimate their signed descriptive effects without requiring a favorable direction. Also report known four-class and joint twelve-class macro-F1, per-class F1/recall/support, per-subject metrics, actual queries, unused budget, known/duplicate answers, names recovered, candidate coverage and all available prefixes. Spending-matched comparisons use q=min(actual queries of the two arms) and retain zero-query episodes separately; no positive-prefix-only primary conclusion.

All evaluation subjects are disjoint from acquisition, so paid-window exclusion removes none of the evaluation IDs. Compare every method on identical evaluation IDs. Shared acquisition and fixed source imply that subject scores are conditional on one acquisition episode; do not count windows, seeds or the five clients as independent experimental replications. No significance, statistical-power, universal superiority or formal privacy claim is preauthorized by this design. Report all observed known-class losses; a 0.02 loss versus the corresponding all arm is a descriptive review flag, not a reason to discard or rerun. No target accuracy qualification is used to hide a poor transfer result.

## Implementation admission and execution order

Before raw target acquisition, finish the runner and test generated fixtures for column selection, magnitude/dtype, window boundaries/null handling, ID uniqueness, source-normalization parity, subject-role isolation, all twelve labels, eight-class missing-label denominator, paid known/duplicate costs, empty agreement, random ordering and TC selection. Pin all executable sources and verify that they implement this contract without new scientific choices. Source-only prototype/normalization extraction must pass before a target fetch; no local Torch or research inference is permitted.

Then obtain the official archive in Colab, retain raw bytes and record its SHA256 before extraction. No official archive content hash is known at this checkpoint; the source URL/DOI and first-fetched byte identity must not be described as a preexisting verified checksum. Extract only expected safe paths, verify all ten subject files and their hashes, and preserve archive documentation. A source/version discrepancy stops before scientific outcomes and requires a recorded review.

After input checks, encode/cache once; prepare/seal every plan; answer paid queries; build/seal every prefix prediction; then score and expose outcomes. Save structural failures and logs. No automatic rerun, relaxed threshold, extra source fold, new activity mapping or deletion after observing results. Implementation errors require a preserved attempt and a justified correction, not scientific tuning.

The future handoff will specify measured/admitted runtime ceilings and memory safeguards after generated/source-only preflight. No resource estimate or runnable notebook is supplied by this specification. The final budget/population cannot be enlarged merely because a method returns no new activity. M1's findings, if obtained, will concern this fixed external-transfer task; they will not validate the broader end-to-end federated branch or retroactively pass stopped mechanisms.
