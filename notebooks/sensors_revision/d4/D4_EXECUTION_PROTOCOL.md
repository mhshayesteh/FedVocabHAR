# D4 execution and resource addendum

16November2026. Implements the unchanged TC-Frozen-Sensor-v1 numerical profile and random-query schedule from TYPICLUST_SENSOR_ADMISSION_20261116.md/.json. Independent adaptation, not a native TypiClust reproduction. No new encoder or independent evaluation population.

25conditions per fold: four reused D3arms, one TCselector and20new random seeds20261117..20261136. Nine existing development folds;225conditions total. Each has B=4. The old random seed20261116is retained in D3history, excluded from this new seed-average. Fixed seeds are integer identifiers, not calendar dates.

96input byte hashes pin existing dataset/manifests/checkpoints/D1andD3outputs. StageBembedding bytes/IDs/calibration/source/runtime checks are inherited. D3candidate/prediction seals and stored source/ledger identity are verified during preparation and again in Colab before reuse; each reused arm must agree with the current known-only decoder at prefix0. New arms never use old arms' answers to choose queries or construct prototypes.

Reused D3queries are ledger entries, not new oracle calls. At most756new simulated calls (TC9x4plus20randomseeds x9x4), with at most900total calls over all conditions if every reused answer is counted in its own arm. Record both quantities. Cached-ledger prefix replay costs zero additional calls. Known and duplicate answers still consume each initial simulation's budget.

All new query plans for a fold seal before truth-backed oracle simulation. Four D3prediction sets and every new arm's prefixes0..actual spending seal before evaluator scoring. Primary population excludes the union of every arm/seed's full-budget queried evaluation IDs. Full and own-unpaid scores are sensitivity analyses. For own-unpaid prefixes, use a fixed arm-level population excluding its complete full-budget paid set, not a changing prefix denominator. Always keep the fixed class vocabulary and per-class support/recall.

Random seeds are summarized within fold using mean, population standard deviation, min/max and every seed's result; only then may fold means be averaged per dataset. No best-seed selection, significance over windows or independent-subject claim. Rescore the saved D3predictions on the new population instead of comparing their older D3scores directly. Do not average the centralized-feature TCpolicy with signature-policy conditions into an unqualified leaderboard.

Resource ceilings:15minutes overall/outer,60seconds per task, peak RSS2GiB. Progress prints after each arm is ready and each fold is scored. Input checks and generated conformance precede research. Partial outputs, logs and negative/empty/under-spending outcomes remain. The notebook refuses an automatic repeat once STARTED exists. Version mismatch stops without installation/fallback.

NumPy synthetic checks run locally for hand Euclidean distances, self/duplicate exclusion, zero-distance finiteness, size5boundary, selected-count/tie ordering, invalid features and random ID order invariance. Colab additionally checks fixed sklearn KMeans on generated separated/collapsed clusters and the existing oracle/scorer fixtures before research arrays. These tests are conformance, not performance evidence.

Historical Torch definitions import in Colab; model/checkpoint load/encode/embedding routes are disabled. Real inference and optimization never run locally. Direct within-cluster vector subtraction defines ordinary Euclidean distances; no squared-distance shortcut or covariance/normalization transform. Independent evaluation, a complete native implementation and a distributed density protocol remain unresolved as documented in the admission specification.

After completion, inspect all results and audit sources/ledgers/predictions before any next choice. No extra seed, threshold, representation, new dataset, manuscript rewrite or rescue policy is launched automatically.
