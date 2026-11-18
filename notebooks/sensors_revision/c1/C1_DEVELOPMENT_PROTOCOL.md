# C1: Aggregate-Centroid-Sensor-v1 development protocol

17 November 2026. Independent paper-based selector adaptation from Coordination on a Budget v2, section 4.2. No upstream author code is copied. This is not native FederatedTypiClust or the complete three-phase learner.

## Locked question

On the existing supplied-encoder development episodes, how does global aggregate-centroid coordination compare with the retained agreement/support policies, centralized TC reference and uniform random selection under the same four-call cap and purchased-window decoder?

No MHEALTH access, new target population, threshold search, training or model inference is included. This is inspected development. It cannot establish independent replication or close complete learned-system baseline coverage.

## Algorithm and deliberate adaptation choices

Each simulated client stores only its own selected-pool rows. Distributed Lloyd iterations send cluster sums/counts; server initialization also uses global sums, counts and trace second moments. The server broadcasts centroids and obtains nearest feasible window IDs/distances for each cluster. It ranks feasible clusters by fewest already purchased examples, then largest full cluster population, then numeric cluster ID. It chooses the closest feasible sample, resolving exact distance ties by client ID and window ID. Geometry is fixed during the four-call batch. Every purchased example increments paid cluster count, irrespective of later semantic answer.

The number of clusters is min(4, pool size), with no privileged novel-class count. Initialization is the aggregate mean plus fixed-seed normalized Gaussian directions scaled by aggregate trace variance. Centroids are arithmetic means, not renormalized. Squared Euclidean distances, float64 arithmetic, 100 iterations and absolute maximum-centroid-shift tolerance 1e-8 are fixed. Empty centroids remain in place and cannot be queried without feasible members. An iteration-cap hit stops the research run for review; it does not trigger a replacement seed or initialization. These initialization/stopping/tie choices are our sensor profile, not established author settings.

The research profile uses one global B=4, allowing any client to supply a query while that global cap permits. It **relaxes the paper's fixed local budget allocation** to match the existing D4 cap. The core also tests an explicitly allocated-cap mode, but that mode is not a research arm here. Never claim fixed-quota equivalence. This implementation is an in-process message simulation, not a physically federated or privacy-proven deployment. Aggregate sums can reveal individual information for small groups; metadata and centroid distances also leak geometry. No byte/energy/privacy claims are added.

## Data and decoding

Use the exact nine D4 episodes: five RealWorld development folds and four exploratory PAMAP2 folds. The D4 selected top-N pool IDs and existing frozen Stage-B embeddings are retained. Source known prototypes, labels, calibration and unit conventions remain unchanged. All cache identities, source bytes, runtime versions, manifests and checkpoint byte hashes remain checked; checkpoint loading and encoder routes are disabled.

Reuse all 25 old D4 query plans, oracle ledgers and prefix predictions byte-for-byte. Only C1 issues new simulated queries: at most 4 per fold, 36 total. Known and duplicate answers still cost a call. Only the first purchased novel name adds its representative window prototype; known prototypes remain fixed. Replaying prefixes incurs no additional research-query cost. No unpaid output names are supplied by evaluator mappings.

Add C1 to each fold's plan and seal it before oracle joins. Seal all predictions before evaluation scoring. Recompute all 26 arms' primary scores on one shared unpaid population excluding the union of all 26 full-budget paid ID sets. Do not compare C1 to the old D4 printed table as if its denominator were unchanged. Full and fixed-own-unpaid scores remain sensitivities. PAMAP2 retains disjoint acquisition/evaluation roles and zero-query outcomes. RealWorld remains reused development data.

Primary metrics retain every scheduled known/novel class, assigning zero to unrecovered/absent novel classes. Report all 234 condition rows. Average the 20 random seeds within each fold, then equally average folds separately by dataset. No best-seed selection, cross-dataset grand mean, independent-subject significance or new-episode claim. Preserve selection ledgers, failures, underspending, per-class/prefix scores and known-F1 trade-offs. Existing manuscript numbers remain unchanged pending a separate saved-output review.

## Qualification and execution

Thirteen local synthetic checks passed under Python 3.12.14 / NumPy 2.3.5. Checks cover hand-calculated centroids/order, an independently pooled Lloyd reference, global and allocated caps, empty/short pools, empty/collapsed clusters, deterministic ties, row/client-order invariance, invalid inputs, no semantic-answer API and bounded iterations. These establish scoped implementation behavior, not research performance.

The Colab notebook repeats synthetic checks under the pinned original cache runtime before research. It requires CPU, verifies every pinned input, retains exact D4 source dependencies, and refuses a second attempt after an earlier STARTED or launcher log exists. It does not install or upgrade packages. Runtime/hash mismatch stops without fallback. Limits are 15 minutes overall, 60 seconds per fold/task segment and 2 GiB peak RSS. Partial output and error logs remain. The user starts the separately prepared notebook manually; no research run occurred during preparation.

When FINISHED appears, review results and independently recompute metrics/ledgers before manuscript edits or another experimental decision. A failure is retained without automatic retries. MHEALTH is already inspected and is not used for development here.
