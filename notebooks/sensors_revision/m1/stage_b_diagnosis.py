"""Development-only failure accounting, outside the label-free acquisition API."""
from pathlib import Path
from types import SimpleNamespace
import csv, hashlib, json, platform, time
import numpy as np
import torch, scipy, sklearn
from data_access import CandidateView, BudgetedOracle
import reference_acquisition as A

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def runtime():
    return dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                sklearn=sklearn.__version__,scipy=scipy.__version__,device='cpu',torch_threads=4)

def scope_dataset(P,name,fold):
    D=P.Frozen(name)
    train=np.flatnonzero(np.isin(D.subjects,fold['train_subjects']))
    known_train=train[np.isin(D.activities[train],D.known)]
    roles=['selection_subjects'] if name=='realworld' else ['selection_subjects','integration_subjects','test_subjects']
    subjects=sum([fold[k] for k in roles],[])
    other=np.flatnonzero(np.isin(D.subjects,subjects))
    ids=np.sort(np.concatenate([known_train,other]))
    assert len(ids)==len(set(map(int,ids)))
    D.x=D.x[ids];D.subjects=D.subjects[ids];D.activities=D.activities[ids]
    if name=='realworld':assert not np.isin(D.subjects,fold['test_subjects']+fold['integration_subjects']).any()
    return D,ids

def embedding(P,D,ids,fold,project,cache_dir):
    checkpoint=D.ds['models']/f"fold_{fold['fold']}_models.pt"
    identity=dict(dataset=D.name,fold=fold['fold'],cache_sha256=sha(D.ds['cache']),
        checkpoint_sha256=sha(checkpoint),manifest_sha256=sha(D.ds['manifest']),
        source_sha256=sha(P.__file__),scope_sha256=A.array_sha(ids),runtime=runtime())
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    meta=cache_dir/f'{D.name}_fold_{fold["fold"]}_{key}.json'
    path=meta.with_suffix('.npz')
    if meta.exists() and path.exists():
        saved=json.loads(meta.read_text());assert saved['identity']==identity and saved['npz_sha256']==sha(path)
        with np.load(path,allow_pickle=False) as data:
            assert np.array_equal(data['window_ids'],ids)
            emb=data['embeddings']
        assert A.array_sha(emb)==saved['embedding_sha256']
        return emb,saved
    assert not meta.exists() and not path.exists(), 'Partial embedding cache; preserve and investigate'
    emb=D.embedding(fold)
    assert emb.shape==(len(ids),64) and np.isfinite(emb).all()
    np.savez_compressed(path,window_ids=ids,embeddings=emb)
    saved=dict(identity=identity,embedding_sha256=A.array_sha(emb),npz_sha256=sha(path),path=str(path))
    meta.write_text(json.dumps(saved,indent=2),encoding='utf-8')
    return emb,saved

def diagnose_fold(P,name,fold,project,cache_dir):
    started=time.perf_counter();P.seed_all(P.CFG.seed)
    D,ids=scope_dataset(P,name,fold);emb,cache=embedding(P,D,ids,fold,project,cache_dir)
    p,cal=P.calibration(D,emb,fold)
    clients=tuple(fold[D.ds['client_key']]);mask=np.isin(D.subjects,clients)
    view=CandidateView(ids[mask],D.subjects[mask],emb[mask])
    rows,pools,discarded=A.construct(view,clients,p,cal,P.cluster)
    groups,margins=A.associate(rows,cal['agreement_similarity']);cands=A.candidates(rows,groups)
    lookup={int(i):str(a) for i,a in zip(ids,D.activities)}
    oracle=BudgetedOracle(lookup.__getitem__,view.window_ids,4)
    promoted,trace=A.acquire(cands,frozenset(D.known),oracle,4)
    # Verify original mechanism on the identical scoped features before auditing labels.
    legacy_rows=P.local(D,emb,clients,p,cal)
    old,queries,labels=P.policy_ranked_mnn(D,emb,clients,p,cal,legacy_rows,0)
    equivalent=(labels==[r['label'] for r in trace] and queries==len(trace)
                and [(r['label'],r['support']) for r in old]==[(r['label'],r['support']) for r in promoted]
                and len(rows)==len(legacy_rows)
                and all(np.array_equal(a['signature'],b['signature']) and a['support']==b['support'] for a,b in zip(rows,legacy_rows)))
    assert equivalent,'Label-free acquisition deviated from legacy reference'
    def hist(windows):
        v,n=np.unique([lookup[int(i)] for i in windows],return_counts=True)
        return dict(zip(map(str,v),map(int,n)))
    cluster_audit=[]
    associated={i for g in groups for i in g};eligible={r['cluster'] for r in margins}
    for i,r in enumerate(rows):
        h=hist(r['window_ids']);dominant=max(h,key=h.get)
        cluster_audit.append(dict(dataset=name,fold=fold['fold'],cluster=i,client=r['client'],support=r['support'],
            medoid=r['medoid'],medoid_label=lookup[r['medoid']],majority_label=dominant,
            majority_fraction=h[dominant]/r['support'],medoid_label_purity=h[lookup[r['medoid']]]/r['support'],
            threshold_eligible=i in eligible,mnn_associated=i in associated,label_histogram=json.dumps(h,sort_keys=True)))
    pair_audit=[]
    for i,g in enumerate(groups):
        a,b=[cluster_audit[j] for j in g]
        h=hist(cands[i]['window_ids'])
        pair_audit.append(dict(dataset=name,fold=fold['fold'],candidate=i,member_clusters=json.dumps(g),support=cands[i]['support'],
            medoid_labels_agree=a['medoid_label']==b['medoid_label'],majority_labels_agree=a['majority_label']==b['majority_label'],
            representative=cands[i]['medoid'],representative_label=lookup[cands[i]['medoid']],
            representative_margin=cands[i]['representative_margin'],label_histogram=json.dumps(h,sort_keys=True)))
    selected={t['candidate'] for t in trace}
    breakdown=[]
    for client in clients:
        raw=set(map(int,view.window_ids[view.client_ids==client]))
        pool=set(next(r['window_ids'] for r in pools if r['client']==client))
        def row_union(indices):return set(i for j in indices if rows[j]['client']==client for i in rows[j]['window_ids'])
        retained=row_union(range(len(rows)));elig=row_union(eligible);agreed=row_union(associated)
        chosen=set(i for j in selected for i in cands[j]['window_ids']) & raw
        promoted_ids=set(i for r in promoted for i in r['window_ids']) & raw
        for activity in D.vocabulary:
            activity_ids={i for i in raw if lookup[i]==activity}
            fragments=[len(activity_ids & set(r['window_ids'])) for r in rows if r['client']==client and activity_ids & set(r['window_ids'])]
            counts=dict(raw_windows=len(activity_ids),topn_windows=len(activity_ids&pool),retained_windows=len(activity_ids&retained),
                threshold_eligible_windows=len(activity_ids&elig),mnn_windows=len(activity_ids&agreed),
                selected_pool_windows=len(activity_ids&chosen),promoted_pool_windows=len(activity_ids&promoted_ids))
            breakdown.append(dict(dataset=name,fold=fold['fold'],client=int(client),activity=activity,is_novel=activity in D.novel,
                **counts,retained_fragments=len(fragments),largest_fragment_windows=max(fragments,default=0),
                known_contamination_topn=sum(lookup[i] in D.known for i in pool)/len(pool) if pool else 0.))
    query_rows=[];local_entries=[];pooled_correct=0;local_correct=0;pooled_n=0;local_n=0
    promoted_labels={r['label'] for r in promoted}
    for t in trace:
        cand=cands[t['candidate']]
        local=next(rows[i] for i in cand['member_clusters'] if rows[i]['medoid']==cand['medoid'])
        ph,lh=hist(cand['window_ids']),hist(local['window_ids'])
        query_rows.append(dict(dataset=name,fold=fold['fold'],**t,pooled_queried_label_purity=ph.get(t['label'],0)/cand['support'],
            queried_member_support=local['support'],queried_member_label_purity=lh.get(t['label'],0)/local['support'],
            pooled_histogram=json.dumps(ph,sort_keys=True),queried_member_histogram=json.dumps(lh,sort_keys=True)))
        if t['status']=='promoted':
            local_entries.append(dict(signature=local['signature'],label=t['label']))
            pooled_correct+=ph[t['label']];pooled_n+=cand['support'];local_correct+=lh[t['label']];local_n+=local['support']
    # Same purchased labels and known prototypes. No labels are supplied for unqueried members.
    pooled_metrics=P.evaluate(D,emb,fold,promoted);local_metrics=P.evaluate(D,emb,fold,local_entries)
    novel=[r for r in breakdown if r['is_novel']]
    stages=('raw_windows','topn_windows','retained_windows','threshold_eligible_windows','mnn_windows','selected_pool_windows')
    survival={stage:sum(any(r[stage]>0 for r in novel if r['activity']==a) for a in D.novel) for stage in stages}
    summary=dict(dataset=name,fold=fold['fold'],clients=len(clients),scoped_windows=len(ids),
        local_clusters=len(rows),mnn_candidates=len(cands),label_queries=len(trace),unused_budget=4-len(trace),
        known_answers=sum(t['status']=='known' for t in trace),duplicate_novel_answers=sum(t['status']=='duplicate_novel' for t in trace),
        distinct_novel_recovered=len(promoted_labels),novel_class_count=len(D.novel),
        unmatched_novel_medoid_clusters=sum(i not in associated and lookup[r['medoid']] not in D.known for i,r in enumerate(rows)),
        associated_pairs_majority_disagree=sum(not r['majority_labels_agree'] for r in pair_audit),
        associated_pairs_medoid_disagree=sum(not r['medoid_labels_agree'] for r in pair_audit),
        pooled_queried_label_purity=pooled_correct/pooled_n if pooled_n else 0.,
        queried_member_label_purity=local_correct/local_n if local_n else 0.,
        pooled_novel_f1=pooled_metrics['novel_macro_f1'],queried_member_novel_f1=local_metrics['novel_macro_f1'],
        pooled_known_f1=pooled_metrics['known_macro_f1'],queried_member_known_f1=local_metrics['known_macro_f1'],
        acquisition_equivalent=equivalent,**{f'novel_classes_with_{k}':v for k,v in survival.items()},
        elapsed_seconds=time.perf_counter()-started,
        recognition_role='Reused selection-window diagnostic; not independent evaluation' if name=='realworld' else 'Already inspected subject-disjoint exploratory test diagnostic')
    detail=dict(summary=summary,calibration=cal,cache=cache,reference=A.membership_reference(rows,pools,discarded,groups,cands),
        oracle_trace=list(oracle.trace),association_margins=margins,query_trace=query_rows,
        pooled_metrics=pooled_metrics,queried_member_metrics=local_metrics,
        information_access='All histograms and activity survival are offline audit-only; acquisition sees no full labels or novel vocabulary')
    return summary,breakdown,query_rows,cluster_audit,pair_audit,detail

def run(P,project,out,cache_dir):
    torch.set_num_threads(4);P.DEVICE=torch.device('cpu');P.PROJECT=project
    for ds in P.DATASETS.values():
        for key in ('cache','manifest','models'):ds[key]=project/ds[key].relative_to(Path(P.CFG.project))
    P.assert_locked()
    out.mkdir(parents=True,exist_ok=True);cache_dir.mkdir(parents=True,exist_ok=True)
    (out/'environment.json').write_text(json.dumps(runtime(),indent=2))
    tables=[[],[],[],[],[]]
    for name in ('realworld','pamap2'):
        manifest=json.loads(P.DATASETS[name]['manifest'].read_text(encoding='utf-8-sig'))
        for fold in manifest['outer_folds']:
            target=out/f"{name}_fold_{fold['fold']}_diagnosis.json"
            if target.exists():raise FileExistsError(target)
            result=diagnose_fold(P,name,fold,project,cache_dir)
            target.write_text(json.dumps(result[-1],indent=2),encoding='utf-8')
            for table,rows in zip(tables,result[:-1]):table.extend(rows if isinstance(rows,list) else [rows])
            for filename,table in zip(('fold_summary.csv','failure_breakdown.csv','query_trace.csv','cluster_audit.csv','association_audit.csv'),tables):write_csv(out/filename,table)
            print('FOLD_COMPLETE',json.dumps(result[0]),flush=True)
    (out/'summary.json').write_text(json.dumps(tables[0],indent=2),encoding='utf-8')
    print('STAGE_B_COMPLETE',out,flush=True)
    return tables[0]
