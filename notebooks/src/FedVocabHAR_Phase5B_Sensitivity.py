# %% [markdown]
# # FedVocabHAR Phase 5B
#
# ## Sensitivity of ranked-MNN promotion to its locked hyperparameters
#
# Phase 3D locked five numbers without reporting how results move around them. Phase 5B sweeps each one, and the
# two-way agreement × cluster-distance grid, on **frozen encoders** with **no test contact**:
#
# * **RealWorld**: clients and evaluation are the selection subjects, as in Phase 3D / 5A.
# * **PAMAP2** (diagnostic): clients and evaluation are the *selection* subjects, so this sweep does not touch the
#   test subjects again. It is labelled exploratory and exists only to explain why the RealWorld-calibrated
#   thresholds under-fire on PAMAP2 (Phase 5A).
#
# At every grid point `ranked_mnn`, `support_only`, and `random_query` (20 seeds) are run with the **same** budget,
# so the plots show the *advantage over chance* as a function of each parameter, not just the raw score.
#
# **Nothing in this notebook may be used to change the Phase 5C preregistration.** The locked point is
# reproduced and asserted first; the sweep is descriptive.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import sys, json, itertools
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '/content/drive/MyDrive/FedVocabHAR/colab/src')
import phase5_common as P5

P5.seed_all(P5.CFG.seed); lock = P5.assert_locked()
OUT = P5.PROJECT / 'artifacts' / 'phase5b'; OUT.mkdir(parents=True, exist_ok=True)
print('device:', P5.DEVICE); print(json.dumps(asdict(P5.CFG), indent=2))

# %% [markdown]
# ## Sweep definition
#
# One-at-a-time sweeps around the locked configuration, plus a two-way grid. The locked value is always included
# so every curve passes through the Phase 3D operating point.

# %%
BASE = P5.CFG
ONE_AT_A_TIME = {
    'agreement_similarity_quantile': [.02, .05, .10, .20, .35, .50, .75],
    'local_distance_quantile': [.70, .80, .90, .95, .98],
    'label_query_budget': [1, 2, 4, 6, 8],
    'min_local_cluster_size': [6, 12, 24, 48],
    'max_candidates_per_client': [300, 600, 1200],
    'novelty_false_alarm_quantile': [.01, .02, .05, .10],
}
GRID = {'agreement_similarity_quantile': [.02, .05, .10, .20, .35, .50], 'local_distance_quantile': [.70, .80, .90, .95, .98]}
for k, v in ONE_AT_A_TIME.items(): assert getattr(BASE, k) in v, k

points = [('locked', {})]
for k, vals in ONE_AT_A_TIME.items():
    points += [(f'oat:{k}', {k: v}) for v in vals if v != getattr(BASE, k)]
for a, l in itertools.product(*GRID.values()):
    if (a, l) != (BASE.agreement_similarity_quantile, BASE.local_distance_quantile):
        points.append(('grid', {'agreement_similarity_quantile': a, 'local_distance_quantile': l}))
print(len(points), 'configurations')

SWEEP_DATASETS = {
    'realworld': dict(client_key='selection_subjects', eval_key='selection_subjects', status='Selection-only; matches Phase 3D/5A protocol.'),
    'pamap2': dict(client_key='selection_subjects', eval_key='selection_subjects', status='Exploratory diagnostic on selection subjects; test subjects not indexed.'),
}
POLICIES = {'ranked_mnn': P5.policy_ranked_mnn, 'support_only': P5.policy_support_only, 'random_query': P5.policy_random_query}

# %% [markdown]
# ## Run
#
# Embeddings are computed once per fold and reused for every configuration. Per-dataset/fold JSON files make the
# run resumable.

# %%
FOLD_DIR = OUT / 'folds'; FOLD_DIR.mkdir(exist_ok=True)
for dname, spec in SWEEP_DATASETS.items():
    D = P5.Frozen(dname); print(f'== {dname}: {spec["status"]}')
    for fold in D.folds:
        fid = int(fold['fold']); path = FOLD_DIR / f'{dname}_fold_{fid}_complete.json'
        if path.exists(): print(' fold', fid, 'already complete'); continue
        clients = tuple(fold[spec['client_key']]); eval_subjects = tuple(fold[spec['eval_key']])
        assert set(fold['test_subjects']).isdisjoint(clients) and set(fold['test_subjects']).isdisjoint(eval_subjects)
        emb = D.embedding(fold); rows_out = []
        for tag, over in points:
            cfg = replace(BASE, **over); p, c = P5.calibration(D, emb, fold, cfg); rows = P5.local(D, emb, clients, p, c, cfg=cfg)
            for name, fn in POLICIES.items():
                seeds = range(cfg.random_query_seeds) if name == 'random_query' else [0]
                acc = []
                for s in seeds:
                    entries, queries, ql = fn(D, emb, clients, p, c, rows, seed=cfg.seed + 1000 * fid + s, cfg=cfg)
                    assert queries <= cfg.label_query_budget
                    m = P5.evaluate(D, emb, fold, entries, cfg, eval_subjects=eval_subjects); m.pop('per_class_f1', None)
                    acc.append(P5.summarize(D, entries, queries, ql, m))
                mean = pd.DataFrame(acc).mean(numeric_only=True).to_dict()
                row = {'dataset': dname, 'fold': fid, 'sweep': tag, 'policy': name, 'local_signatures': len(rows), **{k: getattr(cfg, k) for k in ONE_AT_A_TIME},
                       'realized_local_distance': c['local_distance'], 'realized_agreement_similarity': c['agreement_similarity'], 'realized_novelty_threshold': c['novelty_threshold']}
                row.update(mean); rows_out.append(row)
        path.write_text(json.dumps({'fold': fold, 'results': rows_out}, indent=2)); print(' completed fold', fid, f'({len(rows_out)} rows)')

# %% [markdown]
# ## Reproduction guard against Phase 5A

# %%
payloads = [json.loads(p.read_text()) for p in sorted(FOLD_DIR.glob('*_complete.json'))]
res = pd.DataFrame([r for p in payloads for r in p['results']]); res.to_csv(OUT / 'phase5b_sweep_fold_results.csv', index=False)
ref = pd.read_csv(P5.PROJECT / 'artifacts' / 'phase5a' / 'phase5a_fold_results.csv')
ref = ref[(ref.dataset == 'realworld') & (ref.policy == 'ranked_mnn')].set_index('fold').sort_index()
ours = res[(res.dataset == 'realworld') & (res.sweep == 'locked') & (res.policy == 'ranked_mnn')].set_index('fold').sort_index()
for col in ('label_queries', 'novel_recovery', 'entry_purity', 'novel_macro_f1'):
    assert np.allclose(ours[col], ref[col], atol=1e-6), f'locked point drifted from Phase 5A on {col}'
print('Locked configuration reproduces Phase 5A / Phase 3D.')

# %% [markdown]
# ## Aggregation: fold means, bootstrap CIs, and advantage over random

# %%
KEYS = ['dataset', 'sweep', 'policy'] + list(ONE_AT_A_TIME)
METRICS = ['novel_macro_f1', 'combined_macro_f1', 'novel_recovery', 'entry_purity', 'label_queries', 'promoted_entries', 'known_label_queries',
           'realized_local_distance', 'realized_agreement_similarity', 'local_signatures']
agg = res.groupby(KEYS)[METRICS].mean().reset_index()
ci = res.groupby(KEYS)['novel_macro_f1'].apply(lambda v: pd.Series(P5.boot_ci(v), index=['novel_f1_ci_lo', 'novel_f1_ci_hi'])).unstack().reset_index()
agg = agg.merge(ci, on=KEYS)
rk = agg[agg.policy == 'ranked_mnn'].drop(columns='policy'); rq = agg[agg.policy == 'random_query'][KEYS[:2] + list(ONE_AT_A_TIME) + ['novel_macro_f1', 'novel_recovery']]
rq = rq.rename(columns={'novel_macro_f1': 'random_novel_macro_f1', 'novel_recovery': 'random_novel_recovery'})
adv = rk.merge(rq, on=KEYS[:2] + list(ONE_AT_A_TIME)); adv['advantage_novel_f1'] = adv.novel_macro_f1 - adv.random_novel_macro_f1
# fold-level win count vs random at each point
pf = res.pivot_table(index=['dataset', 'sweep', 'fold'] + list(ONE_AT_A_TIME), columns='policy', values='novel_macro_f1').reset_index()
pf['ranked_wins'] = (pf.ranked_mnn > pf.random_query).astype(int)
wins = pf.groupby(['dataset', 'sweep'] + list(ONE_AT_A_TIME))['ranked_wins'].agg(['sum', 'count']).reset_index().rename(columns={'sum': 'folds_ranked_beats_random', 'count': 'folds'})
adv = adv.merge(wins, on=['dataset', 'sweep'] + list(ONE_AT_A_TIME))
agg.to_csv(OUT / 'phase5b_sweep_summary.csv', index=False); adv.to_csv(OUT / 'phase5b_ranked_vs_random.csv', index=False)
display(adv[adv.sweep == 'locked'].round(4))

# %% [markdown]
# ## Figures
#
# One-at-a-time curves (ranked-MNN with 95 % CI band, support-only, random) and the two-way heat maps of ranked
# novel F1 and its advantage over random.

# %%
for dname in SWEEP_DATASETS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (param, vals) in zip(axes.ravel(), ONE_AT_A_TIME.items()):
        sel = agg[(agg.dataset == dname) & (agg.sweep.isin(['locked', f'oat:{param}']))].sort_values(param)
        for pol, style in (('ranked_mnn', '-o'), ('support_only', '--s'), ('random_query', ':^')):
            s = sel[sel.policy == pol]; ax.plot(s[param], s.novel_macro_f1, style, label=pol)
            if pol == 'ranked_mnn': ax.fill_between(s[param], s.novel_f1_ci_lo, s.novel_f1_ci_hi, alpha=.15)
        ax.axvline(getattr(BASE, param), color='k', lw=.8, ls='--'); ax.set_xlabel(param.replace('_', ' ')); ax.set_ylabel('novel macro-F1'); ax.grid(alpha=.25)
        if param in ('max_candidates_per_client',): ax.set_xscale('log', base=2)
    axes[0, 0].legend(fontsize=8); fig.suptitle(f'{dname}: one-at-a-time sensitivity (dashed = locked)'); fig.tight_layout()
    fig.savefig(OUT / f'phase5b_{dname}_one_at_a_time.png', dpi=180, bbox_inches='tight'); plt.show()

    g = adv[(adv.dataset == dname) & (adv.sweep.isin(['locked', 'grid']))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (col, title) in zip(axes, (('novel_macro_f1', 'ranked-MNN novel F1'), ('advantage_novel_f1', 'ranked minus random novel F1'), ('label_queries', 'ranked-MNN label queries'))):
        piv = g.pivot(index='agreement_similarity_quantile', columns='local_distance_quantile', values=col)
        im = ax.imshow(piv.values, origin='lower', aspect='auto', cmap='RdBu_r' if 'advantage' in col else 'viridis',
                       vmin=-abs(piv.values).max() if 'advantage' in col else None, vmax=abs(piv.values).max() if 'advantage' in col else None)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel('local distance quantile'); ax.set_ylabel('agreement similarity quantile'); ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]): ax.text(j, i, f'{piv.values[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if 'advantage' not in col else 'k')
        ax.scatter([list(piv.columns).index(BASE.local_distance_quantile)], [list(piv.index).index(BASE.agreement_similarity_quantile)], marker='*', s=160, c='r')
        plt.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f'{dname}: agreement × cluster-distance grid (star = locked)'); fig.tight_layout()
    fig.savefig(OUT / f'phase5b_{dname}_grid.png', dpi=180, bbox_inches='tight'); plt.show()

# Budget Pareto: queries vs novel F1 for each policy on RealWorld
fig, ax = plt.subplots(figsize=(6, 4.2))
sel = agg[(agg.dataset == 'realworld') & (agg.sweep.isin(['locked', 'oat:label_query_budget']))]
for pol, style in (('ranked_mnn', '-o'), ('support_only', '--s'), ('random_query', ':^')):
    s = sel[sel.policy == pol].sort_values('label_query_budget'); ax.plot(s.label_queries, s.novel_macro_f1, style, label=pol)
    for _, r in s.iterrows(): ax.annotate(f'B={int(r.label_query_budget)}', (r.label_queries, r.novel_macro_f1), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('mean label queries used'); ax.set_ylabel('novel macro-F1'); ax.grid(alpha=.25); ax.legend(); ax.set_title('RealWorld: query budget trade-off')
fig.tight_layout(); fig.savefig(OUT / 'phase5b_realworld_budget_pareto.png', dpi=180, bbox_inches='tight'); plt.show()

# %% [markdown]
# ## Decision record

# %%
def best_row(df, col): return df.loc[df[col].idxmax()]
record = {'phase': 'Phase 5B', 'status': 'Descriptive sensitivity analysis on frozen encoders; no test subject indexed; no parameter selected or changed.',
          'locked_config': asdict(BASE), 'datasets': {k: v['status'] for k, v in SWEEP_DATASETS.items()}, 'configurations': len(points), 'findings': {}}
report = '# FedVocabHAR Phase 5B Sensitivity\n\nRanked-MNN, support-only, and random-query at every configuration with equal budget. No parameter was selected.\n\n'
for dname in SWEEP_DATASETS:
    g = adv[adv.dataset == dname]; lockedrow = g[g.sweep == 'locked'].iloc[0]; b = best_row(g, 'novel_macro_f1'); ba = best_row(g, 'advantage_novel_f1')
    robust = g[(g.sweep.isin(['locked', 'grid'])) & (g.advantage_novel_f1 > 0)]
    f = {'locked_novel_f1': float(lockedrow.novel_macro_f1), 'locked_advantage_over_random': float(lockedrow.advantage_novel_f1),
         'locked_folds_beating_random': f'{int(lockedrow.folds_ranked_beats_random)}/{int(lockedrow.folds)}',
         'grid_points_with_positive_advantage': f'{len(robust)}/{int((g.sweep.isin(["locked", "grid"])).sum())}',
         'best_novel_f1_in_sweep': {'value': float(b.novel_macro_f1), 'sweep': b.sweep, **{k: float(b[k]) for k in ONE_AT_A_TIME}},
         'best_advantage_in_sweep': {'value': float(ba.advantage_novel_f1), 'sweep': ba.sweep, **{k: float(ba[k]) for k in ONE_AT_A_TIME}}}
    record['findings'][dname] = f
    report += f'## {dname}\n\n{SWEEP_DATASETS[dname]["status"]}\n\n'
    report += f"- Locked point: novel F1 {f['locked_novel_f1']:.4f}, advantage over random {f['locked_advantage_over_random']:+.4f}, beats random in {f['locked_folds_beating_random']} folds.\n"
    report += f"- Grid points where ranked-MNN beats random on mean novel F1: {f['grid_points_with_positive_advantage']}.\n"
    report += f"- Best novel F1 anywhere in the sweep: {b.novel_macro_f1:.4f} ({b.sweep}; agreement q={b.agreement_similarity_quantile}, distance q={b.local_distance_quantile}, B={int(b.label_query_budget)}, min={int(b.min_local_cluster_size)}, top={int(b.max_candidates_per_client)}).\n\n"
    report += '### One-at-a-time (ranked-MNN novel F1 / advantage over random / queries)\n\n| Parameter | Value | Novel F1 [95% CI] | Advantage | Queries | Signatures |\n|---|---|---|---|---|---|\n'
    for param, vals in ONE_AT_A_TIME.items():
        s = g[g.sweep.isin(['locked', f'oat:{param}'])].sort_values(param)
        for _, r in s.iterrows():
            mark = ' **(locked)**' if r.sweep == 'locked' else ''
            report += f"| {param} | {r[param]}{mark} | {r.novel_macro_f1:.4f} [{r.novel_f1_ci_lo:.3f}, {r.novel_f1_ci_hi:.3f}] | {r.advantage_novel_f1:+.4f} | {r.label_queries:.2f} | {r.local_signatures:.1f} |\n"
    report += '\n'
(OUT / 'phase5b_decision.json').write_text(json.dumps(record, indent=2)); (OUT / 'FedVocabHAR_Phase5B_Sensitivity.md').write_text(report); print(report)

# %% [markdown]
# ## Next step
#
# Report the sweep as a robustness figure. Do **not** move the Phase 5C operating point: the preregistration fixes the
# Phase 3D values, and the sweep exists to characterize, not to tune.
