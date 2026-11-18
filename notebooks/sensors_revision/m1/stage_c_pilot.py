"""Fixed two-case structural pilot; exact Stage B embeddings, no encoder inference."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import torch
from data_access import CandidateView,BudgetedOracle
import reference_acquisition as A
import stage_b_diagnosis as B
import stage_c_mechanisms as C

CASES=(('realworld',2),('pamap2',1))
BASE_RUN='artifacts/revision_20261113/stage_b_runs/20261113T153209825609Z'

def load_case(P,project,name,fid):
    prior_path=project/BASE_RUN/f'{name}_fold_{fid}_diagnosis.json'
    prior=json.loads(prior_path.read_text())
    manifest=json.loads(P.DATASETS[name]['manifest'].read_text(encoding='utf-8-sig'))
    fold=next(f for f in manifest['outer_folds'] if f['fold']==fid)
    D,ids=B.scope_dataset(P,name,fold)
    saved=prior['cache'];path=Path(saved['path'])
    assert path.is_relative_to(project/'artifacts/revision_20261113/embeddings')
    assert path.exists() and path.with_suffix('.json').exists(),'Missing Stage B cache: do not silently encode'
    assert json.loads(path.with_suffix('.json').read_text())==saved
    identity=saved['identity']
    assert identity['runtime']==B.runtime() and identity['source_sha256']==B.sha(P.__file__)
    assert identity['cache_sha256']==B.sha(D.ds['cache'])
    assert identity['manifest_sha256']==B.sha(D.ds['manifest'])
    assert identity['checkpoint_sha256']==B.sha(D.ds['models']/f'fold_{fid}_models.pt')
    assert B.sha(path)==saved['npz_sha256'] and identity['scope_sha256']==A.array_sha(ids)
    with np.load(path,allow_pickle=False) as data:
        assert np.array_equal(data['window_ids'],ids)
        embeddings=data['embeddings']
    assert A.array_sha(embeddings)==saved['embedding_sha256']
    prototypes,calibration=P.calibration(D,embeddings,fold)
    assert calibration==prior['calibration'],'Initial calibration differs from Stage B reference'
    return D,ids,fold,embeddings,prototypes,calibration,prior

def run_method(P,D,ids,fold,embeddings,prototypes,calibration,prior,method):
    start=time.perf_counter();clients=tuple(fold[D.ds['client_key']]);mask=np.isin(D.subjects,clients)
    view=CandidateView(ids[mask],D.subjects[mask],embeddings[mask])
    start_construction=time.perf_counter()
    built=C.construct(method,view,clients,prototypes,calibration,P.cluster)
    construction_seconds=time.perf_counter()-start_construction
    lookup={int(i):str(a) for i,a in zip(ids,D.activities)}
    oracle=BudgetedOracle(lookup.__getitem__,view.window_ids,4)
    entries,trace=C.purchase(built['candidates'],built['order'],frozenset(D.known),oracle)
    assert len(trace)==len(oracle.trace)<=4
    if method=='original_mnn':
        assert [r['window_id'] for r in trace]==[r['window_id'] for r in prior['oracle_trace']]
        assert [r['label'] for r in trace]==[r['label'] for r in prior['oracle_trace']]
    if method=='original_support':
        local=P.local(D,embeddings,clients,prototypes,calibration)
        old,q,labels=P.policy_support_only(D,embeddings,clients,prototypes,calibration,local,0)
        assert q==len(trace) and labels==[r['label'] for r in trace]
        assert [(r['label'],r['support']) for r in old]==[(r['label'],r['support']) for r in entries]
    metrics=P.evaluate(D,embeddings,fold,entries)
    if method=='original_mnn':
        assert abs(metrics['novel_macro_f1']-prior['summary']['pooled_novel_f1'])<1e-12
        assert abs(metrics['known_macro_f1']-prior['summary']['pooled_known_f1'])<1e-12
    def histogram(windows):
        labels,counts=np.unique([lookup[int(w)] for w in windows],return_counts=True)
        return dict(zip(map(str,labels),map(int,counts)))
    query_rows=[];correct=0;support=0
    for t in trace:
        h=histogram(built['candidates'][t['candidate']]['window_ids'])
        query_rows.append(dict(dataset=D.name,fold=fold['fold'],method=method,**t,
            queried_label_purity=h.get(t['label'],0)/t['support'],label_histogram=json.dumps(h,sort_keys=True)))
        if t['status']=='promoted':correct+=h[t['label']];support+=t['support']
    local_labels={lookup[r['medoid']] for r in built['rows']}
    candidate_labels={lookup[r['medoid']] for r in built['candidates']}
    row=dict(dataset=D.name,fold=fold['fold'],method=method,local_clusters=len(built['rows']),
        candidates=len(built['candidates']),queries=len(trace),unused_budget=4-len(trace),
        known_answers=sum(t['status']=='known' for t in trace),duplicate_novel_answers=sum(t['status']=='duplicate_novel' for t in trace),
        novel_recovered=len(entries),local_novel_representative_coverage=len(set(D.novel)&local_labels),
        candidate_novel_representative_coverage=len(set(D.novel)&candidate_labels),
        queried_label_purity=correct/support if support else 0.,
        novel_f1=metrics['novel_macro_f1'],known_f1=metrics['known_macro_f1'],joint_f1=metrics['combined_macro_f1'],
        triggered_parents=built.get('triggered_parents',0),refinement_discarded_windows=built.get('refinement_discarded_windows',0),
        proposal_records=built['proposal_records'],proposal_payload_bytes=built['proposal_payload_bytes'],
        construction_seconds=construction_seconds,total_seconds=time.perf_counter()-start,
        encoder_training_seconds=0,encoder_inference_seconds=0)
    detail=dict(result=row,query_trace=query_rows,oracle_trace=list(oracle.trace),metrics=metrics,
        calibration=calibration,embedding_cache=prior['cache'],
        reference=A.membership_reference(built['rows'],built['pools'],built['discarded'],built['groups'],built['candidates']),
        acquisition_order=built['order'],
        information_access='Class coverage and histograms are offline diagnostic fields only. They are not inputs to construction, diversity selection, or acquisition.',
        payload_convention='268 bytes/proposal: 64 float32 + client/support/window IDs int32. Simulated application payload estimate; excludes initialization, downlink, transport, annotation access; not the historical 264-byte convention.')
    return row,query_rows,detail

def screen(results):
    by_method={m:[r for r in results if r['method']==m] for m in C.METHODS}
    assert all(len(rows)==2 for rows in by_method.values())
    candidate=by_method['parent_refinement_mnn'];original=by_method['original_mnn']
    controls=[m for m in C.METHODS if m not in ('original_mnn','parent_refinement_mnn')]
    strongest=min(controls,key=lambda m:(-sum(r['novel_f1'] for r in by_method[m])/2,m))
    comp=by_method[strongest]
    checks=dict(original_gain_both=all(c['novel_f1']-o['novel_f1']>=.01-1e-12 for c,o in zip(candidate,original)),
        strongest_control_mean_gain=sum(c['novel_f1']-b['novel_f1'] for c,b in zip(candidate,comp))/2>=.005-1e-12,
        strongest_control_case_tolerance=all(c['novel_f1']-b['novel_f1']>=-.01-1e-12 for c,b in zip(candidate,comp)),
        known_class_tolerance=all(c['known_f1']-o['known_f1']>=-.02-1e-12 for c,o in zip(candidate,original)),
        recovery_not_reduced=all(c['novel_recovered']>=o['novel_recovered'] for c,o in zip(candidate,original)),
        pamap2_extra_activity=next(r for r in candidate if r['dataset']=='pamap2')['novel_recovered']>=next(r for r in original if r['dataset']=='pamap2')['novel_recovered']+1)
    return dict(cases=list(CASES),strongest_simple_comparator=strongest,checks=checks,
        eligible_for_expanded_development=all(checks.values()),
        conclusion='Eligible for expanded development only; not confirmation or completed broad Gate C' if all(checks.values()) else 'Screen failed: stop this candidate branch; no automatic revision or broad training',
        mean_novel_f1={m:sum(r['novel_f1'] for r in rows)/2 for m,rows in by_method.items()})

def run(P,project,out):
    torch.set_num_threads(4);P.DEVICE=torch.device('cpu');P.PROJECT=project
    for ds in P.DATASETS.values():
        for key in ('cache','manifest','models'):ds[key]=project/ds[key].relative_to(Path(P.CFG.project))
    P.assert_locked();out.mkdir(parents=True,exist_ok=True)
    (out/'environment.json').write_text(json.dumps(B.runtime(),indent=2))
    results=[];queries=[]
    for name,fid in CASES:
        context=load_case(P,project,name,fid)
        for method in C.METHODS:
            row,trace,detail=run_method(P,*context,method)
            path=out/f'{name}_fold_{fid}_{method}.json'
            if path.exists():raise FileExistsError(path)
            path.write_text(json.dumps(detail,indent=2),encoding='utf-8')
            results.append(row);queries.extend(trace)
            B.write_csv(out/'results.csv',results);B.write_csv(out/'query_trace.csv',queries)
            print('PILOT_EPISODE',json.dumps(row),flush=True)
    decision=screen(results)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'screening_decision.json').write_text(json.dumps(decision,indent=2))
    print('PILOT_SCREEN',json.dumps(decision),flush=True)
    return decision
