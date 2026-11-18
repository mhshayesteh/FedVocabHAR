# R2: DSADS prospective transfer and repeated-acquisition protocol
17 November 2026. Version1. Scientific choices fixed before target signals or outcomes. This is a project protocol, not external preregistration. User reports no prior DSADS use by himself/coauthors in any project. That declaration is not independently verified.

## Scope decision and comparator admission
Proceed with a conditional supplied-encoder external evaluation, separately from the unresolved complete-system baseline branch. P2's full-native-method gap remains OPEN. This is our scoped continuation decision, not evidence that an unavailable/unadmitted method loses.
Keep O_all/O_agreed primary and H_all/H_agreed predeclared sensitivity, TC-Frozen-Sensor-v1 centralized-feature reference, Aggregate-Centroid-Sensor-v1 global-cap reference,20fixed random seeds and known_only:27arms per episode,54conditions. TC and aggregate controls remain independently written adaptations, not native complete learners. The aggregate core is the previously qualified C1 code, without retuning. C1/M1/D1–D4 results remain retained; this is not a rerun of them.
Bi et al.'s Dynamic Active HAR remains a mandatory related-work discussion. Bristol's record/DOI endpoint were checked; no readable full algorithm or verified implementation was retrieved. LoGo retains its audited two-model/warm-start/fixed-head mismatch. Full aggregate-framework SimCLR/FedAvg/learner stages are omitted here and not claimed reproduced. No native algorithm is invented from an abstract.

## Primary sources and physical compatibility
Official dataset: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities ; DOI10.24432/C5C59F; CC BY4.0.
Creator paper: https://www.ee.bilkent.edu.tr/~billur/publ_list/pr10.pdf (DOI10.1016/j.patcog.2010.04.019), section2.
Creator follow-up: https://www.ee.bilkent.edu.tr/~billur/publ_list/IoT20.pdf (DOI10.1109/JIOT.2020.2969840), sectionIV and Fig5.
UCI documents8subjects,19activities,25Hz,60five-second segments per activity/subject,45columns and125rows. Right-arm columns10–12 are acceleration. The creator paper locates these sensors on wrists, despite UCI's broader “arms” wording; this differs from the source RealWorld forearm. Primary plots label acceleration in SI units. The input profile uses unchanged acceleration scale1.0 and retains gravity; archive documentation and source normalization checks must be retained. The source-to-target physical correspondence is an explicit transfer assumption, not proof of orientation/device/placement equivalence. No target statistics, signal-driven unit guessing, axis search or performance-selected scale.

## Roles and independence
EpisodeA acquisition subjects1,2; episodeB acquisition subjects3,4. Both use evaluation subjects5,6,7,8. All target subjects excluded from source fitting/calibration. Numeric split chosen before data access.
Seal BOTH episodes' plans before any oracle answers, and BOTH episodes' predictions before evaluation labels/scoring. No answer sharing across methods or episodes. Evaluation features never enter clustering, acquisition or calibration. Same source encoder and shared evaluation subjects induce dependence: report two disjoint acquisition cohorts with shared-test dependence, not two fully independent studies or eight independent folds. No confidence interval, p-value or power claim. Show both effects and their equal-weight descriptive mean; never select the better episode.

## Fixed canonical vocabulary
Retain every raw activity; use this pre-outcome broad activity ontology. The table is a task definition, not a claim of exact native19-way classification.
| Raw IDs | Canonical name | Role / reason |
|---|---|---|
|1|sitting|known|
|2,7|standing|known; includes stationary elevator context|
|9,10,11|walking|known; outdoor, flat/inclined treadmill contexts|
|12|running|known; treadmill running|
|3,4|lying|novel; back/side posture variants|
|5|stairs_up|novel|
|6|stairs_down|novel|
|8|moving_in_elevator|novel; dataset context label, not proof of a new elementary motion|
|13|stepper|novel|
|14|cross_trainer|novel|
|15,16|cycling|novel; horizontal/vertical exercise-bike variants|
|17|rowing|novel|
|18|jumping|novel|
|19|basketball|novel|
Known order walking,running,sitting,standing. Novel denominator is the TEN names above, always fixed. Do not claim each context constitutes a new fundamental motion. Merging is declared before outcomes to avoid treating already-known walking/standing variants as novel names. Preserve raw IDs and unmerged provenance for audit; no post-outcome remapping.

## Segment-safe adapter
Read each125x45finite segment separately. Select XYZ columns10–12; interpolate each axis in float64 at original-sample coordinates j/2 for j=0..248, corresponding to0..4.96seconds on a50Hz grid. This uses no extrapolation, adds no physical frequency content, and is our linear time-grid adaptation, not the source paper's feature recipe. Cast interpolated XYZ to float32, compute float32 Euclidean magnitude AFTER interpolation, then take128-sample windows at starts0and64. Exactly two windows per segment,18,240total if the documented manifest is complete. Never join files; leftover tail is unused. No target normalizer, gravity removal, rotation, class balancing or outcome-driven exclusions.
Source-only normalization exactly matches historical Frozen.normalized_input. Curator lookup retains subject/rawactivity/segment/start and raw hashes. Selectors receive only fixed randomized opaque IDs,clientownership and embeddings. Identifier obfuscation/process separation is not cryptographic privacy; an informed adversary could reconstruct the permutation. Curator labels necessarily define oracle truth, but are not selector inputs.

## Source and selection lock
Reuse RealWorld fold1 checkpoint SHA824772ae3b1ac09266379d7d6d59cc77fd1cd536a4dcdafeaf13745cb9686bbe and original source/cache/code pins from M1. Fix novelty_threshold0.638611912727356, local_distance0.2090341925621032, agreement_similarity0.8902801990509033. Separate source calibration and recognition prototype matrices. Source fold is reused and conditioned on, not independently retrained or selected on DSADS. Freeze weights/eval/batchnorm, CPU/no_grad/batch512. No target training.
Seal source bundle before official target download; record first official archive byte identity rather than inventing a prior checksum. Source pin/runtime mismatch stops before target download.
Each acquisition client contributes at most600label-blind novelty-ranked windows, at most1200per episode. Min local support12; original/half cosine average-linkage cutoffs as above; unchanged MNN/ranking; no empty-agreement fallback. Both reference selectors use the SAME selected pool. Aggregate profile retains globalB4, arithmetic centroids, seed20261117,max100iterations,tolerance1e-8; nonconvergence stops without seed retry. This relaxes upstream fixed client quotas. TC keeps its pinned existing profile. Random seeds20261117..20261136 fixed. Source-known labels/probe/calibration are supervision OUTSIDE the four additional target queries.

## Oracle and decoder
Four paid answers per arm/episode, including known and repeated-novel answers; at most208simulatedcalls. No target labels from another arm/cohort. First purchased occurrence of a novel name adds ONLY its purchased-window embedding to fixed source-known prototypes. Known answers never update prototypes. No free class counts/names, majority oracle, target supervision, evaluator mapping or cluster-centroid decoder. Prefixes replay the same paid ledger without extra calls.
Evaluation subjects are entirely separate from acquisition, so every arm shares the same unqueried evaluation population. No paid-ID subtraction is needed there. Retain all scheduled classes including zero/absent/unrecovered classes. Novel macro-F1 <= recovered_names/10 <=4/10; report this ceiling rather than interpreting absolute scores as an eight-class task result.

## Endpoints and fixed interpretation
Primary per-episode endpoint: mean of ten-class novel macro-F1 over four evaluation subjects at actual full-cap stopping. Primary contrast O_agreed−O_all. H_agreed−H_all and difference-in-differences are declared sensitivities, not winner-selection options. Report each episode, per-subject deltas, descriptive equal-episode average, known/joint14F1, per-class support/F1, spending, name recovery, known/duplicate costs and every prefix0..actualcalls. Random averages20seeds within episode, then episodes equally. Seeds and overlapping windows are not independent subjects. Pooled-window scores are secondary.
Matched-spending contrast uses q=min(actualcalls_agreed,actualcalls_all) and saved prefixq; full-cap remains primary even if agreement underspends. Retain negative/zero-query results. No success-only filtering, best-cutoff/source-fold/dataset switching or comparison to historical printed denominators.
This evaluates controlled transfer under this ontology/encoder; it neither establishes universal superiority nor closes native-system comparisons, independent encoder training or broader population generalization.

## Qualification and execution
Local generated-array/archive tests only: timing/magnitude/shape/roles/vocabulary/IDmapping, direct/nested archive parsing, complete-manifest enforcement, unsafe/duplicate/missing members, independent score checks and synthetic two-episode plan/prediction flow. No local Torch import or real target access. Colab checks pinned runtime/source bytes, source normalization parity, generated sklearn and qualified selectors before download.
Ceilings, not runtime promises:30minutesoverall,3GiBRSS,220MiBcompressed,650MiBexpanded,10000members; source180s,download300s,inference900s,individualtask120s. No dependency install, fallback profile, extra epoch or automatic retry. Preserve partial/error files and review failures before any new attempt.
Notebook writes immutable timestamped outputs under artifacts/revision_20261113/r2_dsads_external/. No manuscript edit/compile or submission occurs. Scientific performance remains unknown until user-run Colab and independent saved-output review.
