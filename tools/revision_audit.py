"""Post hoc metric correction on frozen checkpoints; never trains or selects a policy.

Run from the project root. Original payloads and preregistration files are read-only.
The original module is archived alongside this audit for traceability.
"""
from pathlib import Path
import sys, json, csv, hashlib, importlib.util
import numpy as np
import platform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'revision_20261015'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / ('colab' if (ROOT/'colab').exists() else 'notebooks') / 'src'))
import phase5_metrics_corrected as P
import torch
torch.set_num_threads(4)
P.PROJECT = ROOT
for ds in P.DATASETS.values():
    for key in ('cache', 'manifest', 'models'):
        ds[key] = ROOT / ds[key].relative_to(Path(P.CFG.project))

def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(name):
    D = P.Frozen(name)
    all_rows, checks, controls = [], [], []
    phase = 'phase5c' if name == 'ucihar' else 'phase6'
    prefix = 'ucihar' if name == 'ucihar' else 'motionsense'
    for fold in D.folds:
        fid = fold['fold']; print(name, fid, 'encoding frozen checkpoint', flush=True)
        original = json.loads((ROOT/'artifacts'/phase/'folds'/f'{prefix}_fold_{fid}_complete.json').read_text())
        emb = D.embedding(fold); p, c = P.calibration(D, emb, fold)
        clients = tuple(fold['integration_subjects']); local = P.local(D, emb, clients, p, c)
        for policy, fn in P.POLICIES.items():
            seeds = range(P.CFG.random_query_seeds) if policy == 'random_query' else [0]
            for seed in seeds:
                entries, queries, labels = fn(D, emb, clients, p, c, local, P.CFG.seed + 1000*fid + seed)
                old = next(r for r in original['results'] if r['policy']==policy and r['seed']==seed)
                # Check the query path without reopening any held-out test windows.
                audit_old = [r for r in original['entries'] if r['policy']==policy and r['seed']==seed]
                old_entries = [(r['label'], r['support']) for r in audit_old]
                new_entries = [(r['label'], r['support']) for r in entries]
                match = old_entries == new_entries and queries == old['label_queries']
                checks.append(dict(dataset=name,fold=fid,policy=policy,seed=seed,query_path_matches=match))
                weights = [r['support'] for r in entries]
                purity = float(np.average([r['purity'] for r in entries],weights=weights)) if entries else 0.
                all_rows.append(dict(dataset=name,fold=fid,policy=policy,seed=seed,
                    legacy_entry_purity=old['entry_purity'], corrected_entry_purity=purity,
                    promoted_entries=len(entries),label_queries=queries,query_path_matches=match))
        # Post hoc mechanism controls: identical local clusters and pooled-prototype
        # construction; change only the association rule. No tuning or selection.
        sim=np.asarray([r['signature'] for r in local]) @ np.asarray([r['signature'] for r in local]).T
        directed=set()
        for i,r in enumerate(local):
            eligible=[j for j,s in enumerate(local) if s['client']!=r['client'] and sim[i,j]>=c['agreement_similarity']]
            if eligible: directed.add(tuple(sorted((i,max(eligible,key=lambda j:sim[i,j])))))
        hungarian=[]
        for ai,a in enumerate(clients):
            for b in clients[ai+1:]:
                ia=[i for i,r in enumerate(local) if r['client']==a]
                ib=[i for i,r in enumerate(local) if r['client']==b]
                if not ia or not ib: continue
                ri,ci=P.linear_sum_assignment(-sim[np.ix_(ia,ib)])
                hungarian += [[ia[i],ib[j]] for i,j in zip(ri,ci) if sim[ia[i],ib[j]]>=c['agreement_similarity']]
        for policy,groups in [('ranked_mnn',P.mnn_groups(local,c['agreement_similarity'])),
                              ('oneway_nearest',sorted(directed)),('hungarian_same_clusters',hungarian)]:
            cands=P.group_candidates(D,local,groups)
            entries,queries,labels=P.spend_budget(D,cands,lambda r:r['support'])
            row=dict(dataset=name,fold=fid,policy=policy,agreed_candidates=len(cands))
            row.update(P.summarize(D,entries,queries,labels,P.evaluate(D,emb,fold,entries)))
            controls.append(row)
        write_csv(OUT/f'{name}_association_controls.csv',controls)
        write_csv(OUT/f'{name}_metric_audit.csv',all_rows)
        print(name, fid, 'saved', flush=True)
    # Random seeds are averaged within folds before folds are averaged.
    summaries=[]
    for policy in P.POLICIES:
        values=[]; legacy=[]
        for fid in [f['fold'] for f in D.folds]:
            rs=[r for r in all_rows if r['policy']==policy and r['fold']==fid]
            values.append(float(np.mean([r['corrected_entry_purity'] for r in rs])))
            legacy.append(float(np.mean([r['legacy_entry_purity'] for r in rs])))
        lo,hi=P.boot_ci(values)
        summaries.append(dict(dataset=name,policy=policy,legacy_entry_purity=float(np.mean(legacy)),
            corrected_entry_purity=float(np.mean(values)),ci_lo=lo,ci_hi=hi,
            query_paths_match=all(r['query_path_matches'] for r in all_rows if r['policy']==policy)))
    write_csv(OUT/f'{name}_purity_summary.csv',summaries)
    write_csv(OUT/f'{name}_query_path_checks.csv',checks)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    import sklearn, scipy
    (OUT/'audit_environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
        device=str(P.DEVICE),threads=torch.get_num_threads(),training='none; frozen checkpoints only',
        role='post hoc replay; do not replace original confirmatory records'),indent=2))
    for name in sys.argv[1:] or ['ucihar','motionsense']: run(name)
