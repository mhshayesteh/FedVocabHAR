# %% [markdown]
# # FedVocabHAR Phase 6-C: Confirmatory evaluation on MotionSense (test labels opened exactly once)
#
# Second confirmatory dataset. Runs the locked `ranked_mnn` promotion policy and every preregistered baseline
# on the **integration clients** of each outer fold, then evaluates on the **test subjects**. Everything below
# is fixed by `artifacts/phase6/phase6_preregistered_manifest.json` (commit `f7ffaba`) and the Phase 3D policy
# lock, identically to Phase 5C:
#
# * calibration from train + selection subjects only (Phase 3D quantiles 0.02 / 0.90 / 0.10);
# * budget B = 4 with every oracle call counted, including known-label answers;
# * decision rule: H1 (ranked_mnn novel macro-F1 > random_query mean and in >= 5/6 folds) plus gates G2-G6.
#
# **This notebook must be run once.** Fold payloads are cached; re-running does not re-open test labels. Any
# gate failure is reported as a failed confirmatory test; MotionSense may not be reused for redesign or tuning.
# Per `relationship_to_phase5c`, this test exists solely to check whether the Phase 5C result generalizes; no
# Phase 5A-5D or Phase 5C number may influence anything below.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import sys; sys.path.insert(0, '/content/drive/MyDrive/FedVocabHAR/colab/src')
import phase5_common as P5
from phase5_common import CFG, PROJECT, Frozen, POLICIES, calibration, local, evaluate, summarize, known_prototypes, boot_ci, assert_locked
from dataclasses import asdict
import json, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')

P6 = PROJECT / 'artifacts' / 'phase6'; FOLD_DIR = P6 / 'folds'
manifest = json.loads((P6 / 'phase6_preregistered_manifest.json').read_text(encoding='utf-8-sig'))
g1 = json.loads((P6 / 'phase6_a_g1_record.json').read_text(encoding='utf-8-sig'))
b_rec = json.loads((P6 / 'phase6_b_record.json').read_text(encoding='utf-8-sig'))
lock = assert_locked()
assert manifest['frozen_promotion_policy']['label_query_budget'] == CFG.label_query_budget == 4
assert b_rec['trained_on'] == 'train_subjects only' and len(b_rec['folds']) == 6
for f in manifest['outer_folds']: assert (FOLD_DIR / f"fold_{f['fold']}_models.pt").exists()
D = Frozen('motionsense'); print(D.ds['status']); print('known', D.known, '| novel', D.novel)
print('Config:', json.dumps(asdict(CFG), indent=1))

# %% [markdown]
# ## Per-fold discovery, promotion, and single test evaluation

# %%
def known_geometry_f1(D, emb, fold):
    """G2: prototype known macro-F1 on test-subject windows of known activities only (Phase 4B definition)."""
    kp = known_prototypes(D, emb, fold)
    idx = np.where(np.isin(D.subjects, fold['test_subjects']) & np.isin(D.activities, D.known))[0]
    y = np.asarray([D.known_to_index[a] for a in D.activities[idx]]); pred = (emb[idx] @ kp.T).argmax(1)
    return float(precision_recall_fscore_support(y, pred, labels=list(range(len(D.known))), zero_division=0)[2].mean())

for fold in D.folds:
    fid = int(fold['fold']); path = FOLD_DIR / f'motionsense_fold_{fid}_complete.json'
    if path.exists(): print('fold', fid, 'already complete (test labels not re-opened)'); continue
    clients = tuple(int(s) for s in fold['integration_subjects']); test = tuple(int(s) for s in fold['test_subjects'])
    assert set(test).isdisjoint(clients) and set(test).isdisjoint(fold['train_subjects']) and set(test).isdisjoint(fold['selection_subjects'])
    t0 = time.time(); emb = D.embedding(fold); p, c = calibration(D, emb, fold); rows = local(D, emb, clients, p, c)
    results = []; audit = []
    for name, fn in POLICIES.items():
        seeds = range(CFG.random_query_seeds) if name == 'random_query' else [0]
        for s in seeds:
            entries, queries, queried_labels = fn(D, emb, clients, p, c, rows, seed=CFG.seed + 1000 * fid + s)
            assert queries <= CFG.label_query_budget, 'G6 violated: budget exceeded'
            row = {'dataset': 'motionsense', 'fold': fid, 'policy': name, 'seed': s, 'local_signatures': len(rows)}
            row.update(summarize(D, entries, queries, queried_labels, evaluate(D, emb, fold, entries)))  # eval on test_subjects
            results.append(row)
            for k, e in enumerate(entries, 1):
                audit.append({'fold': fid, 'policy': name, 'seed': s, 'entry': k, 'label': e['label'], 'clients': ','.join(map(str, sorted(e['clients']))),
                              'support': e['support'], 'purity': e['purity'], 'queried_labels': '|'.join(queried_labels)})
    payload = {'fold': fold, 'calibration': c, 'known_geometry_test_f1': known_geometry_f1(D, emb, fold), 'local_signatures': len(rows),
               'results': results, 'entries': audit, 'seconds': time.time() - t0}
    path.write_text(json.dumps(payload, indent=2)); print(f'completed fold {fid} in {payload["seconds"]:.0f}s; local signatures {len(rows)}')

# %% [markdown]
# ## Aggregation, intervals, paired comparisons

# %%
payloads = [json.loads((FOLD_DIR / f"motionsense_fold_{f['fold']}_complete.json").read_text()) for f in D.folds]
results = pd.DataFrame([r for p in payloads for r in p['results']]); entries = pd.DataFrame([r for p in payloads for r in p['entries']])
results.to_csv(P6 / 'phase6_results_per_seed.csv', index=False); entries.to_csv(P6 / 'phase6_promoted_entries.csv', index=False)
geometry = pd.DataFrame([{'fold': int(p['fold']['fold']), 'known_geometry_test_f1': p['known_geometry_test_f1'], 'local_signatures': p['local_signatures'], **p['calibration']} for p in payloads])
geometry.to_csv(P6 / 'phase6_fold_geometry_calibration.csv', index=False)

METRICS = ['novel_macro_f1', 'combined_macro_f1', 'known_macro_f1', 'novel_recovery', 'entry_purity', 'label_queries', 'promoted_entries', 'known_label_queries']
per_class_cols = [c for c in results.columns if c.startswith('f1_')]
fold_level = results.groupby(['fold', 'policy'])[METRICS + per_class_cols].mean().reset_index(); fold_level.to_csv(P6 / 'phase6_fold_results.csv', index=False)

summary_rows = []
for pol, g in fold_level.groupby('policy'):
    row = {'policy': pol, 'folds': len(g)}
    for m in METRICS + per_class_cols:
        lo, hi = boot_ci(g[m]); row[m] = float(np.nanmean(g[m])); row[m + '_ci_lo'] = lo; row[m + '_ci_hi'] = hi
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows).sort_values('novel_macro_f1', ascending=False); summary.to_csv(P6 / 'phase6_summary.csv', index=False)

base = fold_level[fold_level.policy == 'ranked_mnn'].set_index('fold'); paired_rows = []
for pol, h in fold_level.groupby('policy'):
    if pol == 'ranked_mnn': continue
    h = h.set_index('fold').loc[base.index]
    for m in ('novel_macro_f1', 'combined_macro_f1', 'novel_recovery', 'entry_purity', 'label_queries'):
        d = (base[m] - h[m]).values; wins = int((d > 0).sum()); ties = int((d == 0).sum()); n = len(d) - ties
        pval = float(binomtest(wins, n, .5, alternative='greater').pvalue) if n > 0 else float('nan'); lo, hi = boot_ci(d)
        paired_rows.append({'baseline': pol, 'metric': m, 'mean_diff_ranked_minus_baseline': float(d.mean()), 'ci_lo': lo, 'ci_hi': hi,
                            'folds_ranked_better': wins, 'folds_tied': ties, 'folds': len(d), 'sign_test_p_greater': pval})
paired = pd.DataFrame(paired_rows); paired.to_csv(P6 / 'phase6_paired_vs_ranked_mnn.csv', index=False)
display(summary[['policy'] + METRICS].round(4)); display(paired.round(4)); display(geometry.round(4))

# %% [markdown]
# ## Preregistered gates and hypotheses (evaluated exactly as written in the manifest)

# %%
S = summary.set_index('policy'); rk = base; rnd = fold_level[fold_level.policy == 'random_query'].set_index('fold').loc[rk.index]
sup = fold_level[fold_level.policy == 'support_only'].set_index('fold').loc[rk.index]
h1_folds = int((rk.novel_macro_f1 > rnd.novel_macro_f1).sum())
H = {'H1': bool(S.loc['ranked_mnn', 'novel_macro_f1'] > S.loc['random_query', 'novel_macro_f1'] and h1_folds >= 5),
     'H2': bool(S.loc['ranked_mnn', 'entry_purity'] > S.loc['support_only', 'entry_purity']),
     'H3': bool(S.loc['ranked_mnn', 'label_queries'] <= S.loc['support_only', 'label_queries']),
     'H4': bool(S.loc['centralized_oracle', 'novel_macro_f1'] - S.loc['ranked_mnn', 'novel_macro_f1'] < .15)}
G = {'G1_cache_integrity': bool(g1['pass']),
     'G2_known_geometry': bool(geometry.known_geometry_test_f1.mean() >= .75),
     'G3_primary': H['H1'],
     'G4_novel_f1': bool(S.loc['ranked_mnn', 'novel_macro_f1'] >= .30),
     'G5_purity': bool(S.loc['ranked_mnn', 'entry_purity'] >= .60),
     'G6_budget': bool((rk.label_queries <= 4).all() and (results.label_queries <= 4).all())}
confirmatory_pass = all(G.values())
detail = {'H1_folds_ranked_beats_random': f'{h1_folds}/6', 'ranked_mnn_novel_f1': float(S.loc['ranked_mnn', 'novel_macro_f1']),
          'random_query_novel_f1': float(S.loc['random_query', 'novel_macro_f1']), 'centralized_oracle_novel_f1': float(S.loc['centralized_oracle', 'novel_macro_f1']),
          'federation_gap': float(S.loc['centralized_oracle', 'novel_macro_f1'] - S.loc['ranked_mnn', 'novel_macro_f1']),
          'ranked_mnn_entry_purity': float(S.loc['ranked_mnn', 'entry_purity']), 'support_only_entry_purity': float(S.loc['support_only', 'entry_purity']),
          'ranked_mnn_label_queries': float(S.loc['ranked_mnn', 'label_queries']), 'support_only_label_queries': float(S.loc['support_only', 'label_queries']),
          'mean_known_geometry_test_f1': float(geometry.known_geometry_test_f1.mean()), 'g1_failed_checks': [k for k, v in g1['checks'].items() if not v]}
print('Gates:', json.dumps(G, indent=1)); print('Hypotheses:', json.dumps(H, indent=1)); print('CONFIRMATORY PASS:', confirmatory_pass); print(json.dumps(detail, indent=1))

# %% [markdown]
# ## Figure and decision record

# %%
PLOT = ['novel_macro_f1', 'novel_recovery', 'entry_purity', 'label_queries', 'combined_macro_f1', 'known_label_queries']
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for ax, m in zip(axes.ravel(), PLOT):
    fold_level.pivot(index='fold', columns='policy', values=m).plot(kind='bar', ax=ax, width=.85)
    ax.set_title(f'MotionSense test: {m.replace("_", " ")}'); ax.grid(axis='y', alpha=.25); ax.set_xlabel('fold')
    ax.legend(fontsize=6) if m == PLOT[0] else ax.get_legend().remove()
fig.tight_layout(); fig.savefig(P6 / 'phase6_motionsense_confirmatory.png', dpi=180, bbox_inches='tight'); plt.show()

decision = {'phase': 'Phase 6-C', 'dataset': 'MotionSense (pocket userAcceleration, 128/64 windows, g -> m/s^2 scaled)', 'preregistration_commit': 'f7ffaba',
            'confirmatory_pass': confirmatory_pass, 'gates': G, 'hypotheses': H, 'detail': detail,
            'deviations': 'artifacts/phase6/phase6_deviations.md (only if G1 or another check failed; empty/absent if none)',
            'rule': manifest['rule'], 'relationship_to_phase5c': manifest['relationship_to_phase5c'],
            'summary': json.loads(summary.set_index('policy').to_json(orient='index')),
            'paired_vs_ranked_mnn': json.loads(paired.to_json(orient='records')), 'fold_geometry_calibration': json.loads(geometry.to_json(orient='records')),
            'config': asdict(CFG), 'policy_lock': lock}
(P6 / 'phase6_decision.json').write_text(json.dumps(decision, indent=2))

report = (f"# FedVocabHAR Phase 6 Confirmatory Report (MotionSense)\n\n**Confirmatory pass: {confirmatory_pass}**\n\n"
          f"Preregistered in commit `f7ffaba` before download. Test labels were opened once. This is the second confirmatory\n"
          f"dataset, run to test whether the Phase 5C (UCI-HAR) result generalizes; see `relationship_to_phase5c` in the manifest.\n\n"
          "## Gates\n\n| Gate | Pass |\n|---|---|\n" + ''.join(f'| {k} | {v} |\n' for k, v in G.items()) +
          "\n## Hypotheses\n\n| Hypothesis | Holds |\n|---|---|\n" + ''.join(f'| {k} | {v} |\n' for k, v in H.items()) +
          f"\n- H1 fold count: ranked_mnn beats random_query in {detail['H1_folds_ranked_beats_random']} folds.\n"
          f"- Mean known-geometry test F1 (G2): {detail['mean_known_geometry_test_f1']:.4f}.\n"
          f"- Federation gap centralized - ranked (H4): {detail['federation_gap']:+.4f}.\n\n"
          "## Test-subject results (mean over 6 folds, 95% bootstrap CI)\n\n| Policy | Novel F1 [95% CI] | Combined F1 | Known F1 | Recovery | Purity | Queries | Known-label queries |\n|---|---|---|---|---|---|---|---|\n")
for _, r in summary.iterrows():
    report += (f"| {r.policy} | {r.novel_macro_f1:.4f} [{r.novel_macro_f1_ci_lo:.3f}, {r.novel_macro_f1_ci_hi:.3f}] | {r.combined_macro_f1:.4f} | {r.known_macro_f1:.4f} | "
               f"{r.novel_recovery:.3f} | {r.entry_purity:.3f} | {r.label_queries:.2f} | {r.known_label_queries:.2f} |\n")
report += "\n## Per-activity F1 (ranked_mnn)\n\n| Activity | F1 |\n|---|---|\n" + ''.join(f"| {c[3:]} | {S.loc['ranked_mnn', c]:.4f} |\n" for c in per_class_cols)
report += "\n## Paired differences (ranked_mnn minus baseline)\n\n| Baseline | Metric | Mean diff [95% CI] | Folds ranked better | Sign-test p |\n|---|---|---|---|---|\n"
for _, r in paired.iterrows():
    report += f"| {r.baseline} | {r.metric} | {r.mean_diff_ranked_minus_baseline:+.4f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.folds_ranked_better}/{r.folds} | {r.sign_test_p_greater:.3f} |\n"
report += "\n## Per-fold calibration and known geometry\n\n" + geometry.round(4).to_markdown(index=False) + "\n"
(P6 / 'FedVocabHAR_Phase6_Confirmatory.md').write_text(report); print(report)

# %% [markdown]
# ## Closing rule
#
# Whatever the outcome, MotionSense is now spent as a confirmatory dataset. Results are reported as-is in the
# manuscript alongside Phase 5C; no threshold, recipe, or policy may be changed in response to them.
